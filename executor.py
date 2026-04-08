"""
Executor — places trades in paper or live mode.
Mode is read once at startup and never changes mid-session.

Phase 4 (implemented):
- _fetch_resolution(): calls Gamma API to confirm YES/NO outcome before closing
- monitor_open_positions(): waits for real resolution on time-exits; position stays
  open if market not yet resolved (no more exit_price=0 on every time-exit)
- _record_to_evaluator(): writes ResolvedTrade to evaluation_log.jsonl on confirmed close
- Paper mode stop-loss: _get_current_price() now runs in both paper and live modes
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import Config
from forecaster import ForecastResult
from risk_manager import RiskDecision

logger = logging.getLogger(__name__)

PAPER_PORTFOLIO_FILE = Config.PAPER_PORTFOLIO_FILE
TRADES_LOG_FILE = Config.TRADES_LOG_FILE

IS_PAPER_TRADING = Config.PAPER_TRADING


def _load_portfolio() -> dict:
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(PAPER_PORTFOLIO_FILE):
        try:
            with open(PAPER_PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception as exc:
            logger.error(
                "Portfolio file corrupted or unreadable (%s) — refusing to reset silently. "
                "Inspect %s before continuing.",
                exc,
                PAPER_PORTFOLIO_FILE,
            )
            raise RuntimeError(
                f"Portfolio file unreadable: {exc}. Fix or delete {PAPER_PORTFOLIO_FILE} manually."
            ) from exc
    return {
        "balance": 1000.0,
        "peak_balance": 1000.0,
        "open_positions": {},
        "trade_history": [],
    }


def _save_portfolio(portfolio: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    tmp = PAPER_PORTFOLIO_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(portfolio, f, indent=2)
    os.replace(tmp, PAPER_PORTFOLIO_FILE)


def _log_trade(trade_record: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(TRADES_LOG_FILE, "a") as f:
        f.write(json.dumps(trade_record) + "\n")


def _build_trade_record(decision: RiskDecision, status: str) -> dict:
    f: ForecastResult = decision.forecast
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": f.market_id,
        "question": f.question,
        "side": f.side,
        "size_usdc": decision.position_size_usdc,
        "yes_price": f.yes_price,
        "no_price": f.no_price,
        "probability": f.probability,
        "edge": f.edge,
        "confidence": f.confidence,
        "evidence_quality": f.evidence_quality,
        "rationale": f.rationale,
        "status": status,
        "paper_trading": IS_PAPER_TRADING,
        "actual_outcome": None,      # filled in on resolution
        "brier_contribution": None,  # filled in by evaluator
    }


def execute_trade(decision: RiskDecision) -> dict | None:
    """
    Execute an approved trade. Logs the rationale BEFORE placing any order.
    Returns the trade record on success, None on failure.
    """
    if not decision.approved or decision.forecast is None:
        logger.warning("execute_trade called with unapproved decision — skipping.")
        return None

    f: ForecastResult = decision.forecast

    logger.info(
        "EXECUTING TRADE | Market: %s | Side: %s | Size: $%.2f | "
        "Probability: %.3f | Edge: %.4f | Confidence: %s | Rationale: %s",
        f.question[:60],
        f.side,
        decision.position_size_usdc,
        f.probability,
        f.edge,
        f.confidence,
        f.rationale,
    )

    trade_record = _build_trade_record(decision, "pending")

    if IS_PAPER_TRADING:
        result = _execute_paper(decision, f)
    else:
        result = _execute_live(decision, f)

    if result:
        trade_record["status"] = "executed"
        _log_trade(trade_record)

    return trade_record if result else None


def _execute_paper(decision: RiskDecision, f: ForecastResult) -> bool:
    """Simulate order execution and update the paper portfolio."""
    portfolio = _load_portfolio()

    entry_price = f.yes_price if f.side == "YES" else f.no_price
    shares = round(decision.position_size_usdc / entry_price, 4) if entry_price > 0 else 0.0

    now = datetime.now(timezone.utc)
    portfolio["balance"] -= decision.position_size_usdc

    position = {
        "market_id": f.market_id,
        "question": f.question,
        "side": f.side,
        "size_usdc": decision.position_size_usdc,
        "shares": shares,
        "entry_price": entry_price,
        "entry_time": now.isoformat(),
        "expiry_time": (now + timedelta(days=f.days_to_expiry)).isoformat(),
        "stop_loss_price": entry_price * 0.70,
        "condition_id": f.condition_id,
        "token_ids": f.token_ids,
        # Phase 4: needed to build ResolvedTrade on close
        "probability": f.probability,
        "category": f.category,
        "edge": f.edge,
        "confidence": f.confidence,
        "evidence_quality": f.evidence_quality,
    }
    portfolio["open_positions"][f.market_id] = position

    _save_portfolio(portfolio)

    logger.info(
        "[PAPER] Trade simulated: %s %s @ %.3f | Size: $%.2f | Shares: %.2f",
        f.side,
        f.question[:40],
        entry_price,
        decision.position_size_usdc,
        shares,
    )
    return True


def _execute_live(decision: RiskDecision, f: ForecastResult) -> bool:
    """Place a real order via py-clob-client."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs

        client = ClobClient(
            host=Config.CLOB_API_BASE,
            key=Config.POLYMARKET_PRIVATE_KEY,
            chain_id=Config.POLYMARKET_CHAIN_ID,
            funder=Config.POLYMARKET_FUNDER_ADDRESS,
            signature_type=2,
        )

        if not f.token_ids or len(f.token_ids) < 2:
            logger.error(
                "[LIVE] Market %s has incomplete token_ids (%s) — aborting order.",
                f.market_id,
                f.token_ids,
            )
            return False

        if f.side == "YES":
            token_id = f.token_ids[0]
            price = f.yes_price
        else:
            token_id = f.token_ids[1]
            price = f.no_price

        size = round(decision.position_size_usdc / price, 4) if price > 0 else 0

        order_args = OrderArgs(token_id=token_id, price=price, size=size)

        import time as _t, random as _r
        resp = None
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                resp = client.create_and_post_order(order_args)
                break
            except Exception as _exc:
                last_exc = _exc
                if _attempt < 2:
                    _wait = 1.5 ** _attempt + _r.uniform(0, 0.5)
                    logger.warning("[LIVE] Order attempt %d failed: %s - retrying in %.1fs", _attempt + 1, _exc, _wait)
                    _t.sleep(_wait)
        if resp is None:
            raise RuntimeError(f"Order placement failed after 3 attempts: {last_exc}")
        logger.info("[LIVE] Order placed: %s", resp)

        portfolio = _load_portfolio()
        position = {
            "market_id": f.market_id,
            "question": f.question,
            "side": f.side,
            "size_usdc": decision.position_size_usdc,
            "shares": size,
            "entry_price": price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "days_to_expiry_at_entry": f.days_to_expiry,
            "stop_loss_price": price * 0.70,
            "order_id": str(resp),
            "condition_id": f.condition_id,
            "token_ids": f.token_ids,
            # Phase 4: needed to build ResolvedTrade on close
            "probability": f.probability,
            "category": f.category,
            "edge": f.edge,
            "confidence": f.confidence,
            "evidence_quality": f.evidence_quality,
        }
        portfolio["open_positions"][f.market_id] = position
        portfolio["balance"] -= decision.position_size_usdc
        if portfolio["balance"] > portfolio.get("peak_balance", 0):
            portfolio["peak_balance"] = portfolio["balance"]
        _save_portfolio(portfolio)

        return True

    except Exception as exc:
        logger.error("[LIVE] Order placement failed: %s", exc)
        return False


def _fetch_resolution(condition_id: str) -> Optional[bool]:
    """
    Fetch market resolution from Gamma API.
    Returns True = YES won, False = NO won, None = not yet resolved.
    Uses same outcomePrices parsing logic as backtester._parse_resolution().
    """
    try:
        import requests
        resp = requests.get(
            f"{Config.GAMMA_API_BASE}/markets/{condition_id}",
            timeout=10,
        )
        resp.raise_for_status()
        m = resp.json()

        raw = m.get("outcomePrices")
        if raw:
            prices = json.loads(raw) if isinstance(raw, str) else raw
            yes_final = float(prices[0])
            if yes_final > 0.9:
                return True
            if yes_final < 0.1:
                return False

        # Fallback: token price list
        for token in (m.get("tokens") or []):
            if (token.get("outcome") or "").upper() == "YES":
                price = float(token.get("price", 0) or 0)
                if price > 0.9:
                    return True
                if price < 0.1:
                    return False

    except Exception as exc:
        logger.debug("Resolution fetch failed for %s: %s", condition_id, exc)

    return None


def _record_to_evaluator(pos: dict, pnl: float, resolved_yes: bool) -> None:
    """Write a ResolvedTrade to evaluation_log.jsonl. Non-fatal on any error."""
    try:
        from evaluator import record_resolved_trade, ResolvedTrade

        entry_time = pos.get("entry_time", "")
        days_held = 0.0
        if entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                days_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
            except Exception:
                pass

        trade = ResolvedTrade(
            timestamp_resolved=datetime.now(timezone.utc).isoformat(),
            market_id=pos.get("market_id", ""),
            question=pos.get("question", ""),
            category=pos.get("category", "UNKNOWN"),
            side=pos.get("side", "YES"),
            size_usdc=pos.get("size_usdc", 0.0),
            entry_price=pos.get("entry_price", 0.0),
            predicted_probability=pos.get("probability", 0.5),
            actual_outcome=resolved_yes,
            pnl=round(pnl, 4),
            edge_at_entry=pos.get("edge", 0.0),
            confidence=pos.get("confidence", "MEDIUM"),
            evidence_quality=pos.get("evidence_quality", 0.5),
            days_held=round(days_held, 2),
            cross_ref_metaculus=pos.get("cross_ref_metaculus"),
            cross_ref_manifold=pos.get("cross_ref_manifold"),
        )
        record_resolved_trade(trade)
        logger.info(
            "Evaluator: recorded resolved trade for %s (outcome=%s pnl=%.2f)",
            pos.get("market_id"), resolved_yes, pnl,
        )
    except Exception as exc:
        logger.warning("Failed to record resolved trade to evaluator: %s", exc)


def monitor_open_positions() -> list[dict]:
    """
    Check open positions for stop-loss triggers and time-based exits.

    Time-exits: polls Gamma API for real resolution before closing.
    If market not yet resolved, position stays open for the next cycle —
    no more forced 100% loss assumption on every time-exit.

    Stop-loss: checks live CLOB price in both paper and live modes.

    Returns list of closed position dicts, each including resolved_yes and
    predicted_prob for Telegram alert enrichment.
    """
    portfolio = _load_portfolio()
    open_positions: dict = portfolio.get("open_positions", {})
    closed = []

    for market_id, pos in list(open_positions.items()):
        try:
            close_reason = None
            exit_price = None
            resolved_yes = None

            # Time-based exit: 4h before expiry
            now = datetime.now(timezone.utc)
            expiry_str = pos.get("expiry_time") or ""
            if expiry_str:
                expiry_dt = datetime.fromisoformat(expiry_str)
                if expiry_dt.tzinfo is None:
                    expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                remaining_seconds = (expiry_dt - now).total_seconds()
                if remaining_seconds < 4 * 3600:
                    condition_id = pos.get("condition_id", "")
                    if condition_id:
                        resolved_yes = _fetch_resolution(condition_id)
                    if resolved_yes is not None:
                        close_reason = "time_exit_4h_before_expiry"
                        # exit_price: 1.0 if we bet the winning side, 0.0 if we bet the loser
                        won = (pos.get("side") == "YES") == resolved_yes
                        exit_price = 1.0 if won else 0.0
                    else:
                        logger.info(
                            "Position %s near expiry but not yet resolved — holding for next cycle.",
                            market_id,
                        )

            # Stop-loss: check live CLOB price in both paper and live modes
            if close_reason is None:
                live_price = _get_current_price(pos)
                if live_price is not None and live_price < pos.get("stop_loss_price", 0):
                    close_reason = f"stop_loss_triggered (price={live_price:.3f})"
                    exit_price = live_price

            if close_reason:
                pnl = _close_position(portfolio, market_id, pos, close_reason, exit_price=exit_price)

                if resolved_yes is not None:
                    _record_to_evaluator(pos, pnl, resolved_yes)

                closed.append({
                    "market_id": market_id,
                    "question": pos["question"],
                    "side": pos.get("side"),
                    "pnl": pnl,
                    "reason": close_reason,
                    "resolved_yes": resolved_yes,
                    "predicted_prob": pos.get("probability"),
                })
                logger.info("Closed position %s: reason=%s pnl=%.2f", market_id, close_reason, pnl)

        except Exception as exc:
            logger.warning("Error monitoring position %s: %s", market_id, exc)

    if closed:
        _save_portfolio(portfolio)

    return closed


def _get_current_price(pos: dict) -> float | None:
    """Fetch current market price via CLOB API. Works in both paper and live modes."""
    try:
        import requests
        token_ids = pos.get("token_ids") or []
        side = pos.get("side")
        if side == "YES":
            token_id = token_ids[0] if len(token_ids) >= 1 else None
        else:
            token_id = token_ids[1] if len(token_ids) >= 2 else None
        if not token_id:
            logger.warning("Cannot fetch price for position %s: missing token_id", pos.get("market_id"))
            return None
        resp = requests.get(
            "https://clob.polymarket.com/price",
            params={"token_id": token_id, "side": "buy"},
            timeout=10,
        )
        resp.raise_for_status()
        price = float(resp.json().get("price", 0))
        return price if price > 0 else None
    except Exception as exc:
        logger.warning("Price fetch failed for %s: %s", pos.get("market_id"), exc)
        return None


def _close_position(
    portfolio: dict,
    market_id: str,
    pos: dict,
    reason: str,
    exit_price: float | None = None,
) -> float:
    """
    Remove position from portfolio and calculate P&L.

    exit_price = 1.0 (win) or 0.0 (loss) for time-exits with confirmed resolution.
    exit_price = live CLOB price for stop-loss exits.
    Fallback if neither: conservative 0.0 (full loss assumption).
    """
    entry_price = pos.get("entry_price", 0)
    shares = pos.get("shares", 0)
    size_usdc = pos.get("size_usdc", 0)

    if exit_price is None:
        if "stop_loss" in reason:
            exit_price = pos.get("stop_loss_price", 0.0)
        else:
            exit_price = 0.0  # conservative fallback

    pnl = (exit_price - entry_price) * shares

    portfolio["balance"] = portfolio.get("balance", 0) + size_usdc + pnl
    if portfolio["balance"] > portfolio.get("peak_balance", 0):
        portfolio["peak_balance"] = portfolio["balance"]

    del portfolio["open_positions"][market_id]

    record = {
        **pos,
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "exit_reason": reason,
        "pnl": round(pnl, 4),
    }
    portfolio.setdefault("trade_history", []).append(record)
    _log_trade({**record, "status": "closed"})
    return round(pnl, 4)


def get_portfolio_summary() -> dict:
    """Return a summary of current portfolio state."""
    portfolio = _load_portfolio()
    history = portfolio.get("trade_history", [])
    wins = [t for t in history if t.get("pnl", 0) > 0]
    losses = [t for t in history if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in history)

    return {
        "balance": portfolio.get("balance", 0),
        "peak_balance": portfolio.get("peak_balance", 0),
        "open_positions": len(portfolio.get("open_positions", {})),
        "total_trades": len(history),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl": round(total_pnl, 2),
    }
