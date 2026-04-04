"""
Executor — places trades in paper or live mode.
Mode is read once at startup and never changes mid-session.

PLANNED UPGRADES (do not implement until approved):

Phase 4 — Real Outcome Resolution (CRITICAL prerequisite for evaluation)
  Problem: _close_position() currently uses exit_price = entry_price (neutral/wrong).
  The bot cannot compute real P&L or Brier scores without knowing the actual outcome.

  Fix for _close_position():
    When close_reason == "time_exit_4h_before_expiry":
      1. Call Gamma API: GET https://gamma-api.polymarket.com/markets/{condition_id}
      2. Parse outcomePrices field (same logic as backtester._parse_resolution())
      3. resolved_yes = True if yes_final > 0.9, False if yes_final < 0.1, None if ambiguous
      4. If resolved_yes is not None:
           Compute binary P&L:
             if pos["side"] == "YES" and resolved_yes:
                 pnl = size_usdc * (1.0 / entry_price - 1.0)   # correct YES bet
             elif pos["side"] == "YES" and not resolved_yes:
                 pnl = -size_usdc                                # wrong YES bet
             elif pos["side"] == "NO" and not resolved_yes:
                 pnl = size_usdc * (1.0 / entry_price - 1.0)   # correct NO bet
             elif pos["side"] == "NO" and resolved_yes:
                 pnl = -size_usdc                                # wrong NO bet
         else:
             pnl = 0.0  # market not yet resolved — check again next cycle
      5. Add to trade record: actual_outcome=resolved_yes, predicted_prob=pos["probability"]
      6. Only close the position if resolved_yes is not None. Otherwise leave open for next scan.

  Fix for _build_trade_record():
    Add fields to every trade record for evaluation:
      "predicted_probability": f.probability,   # already there as "probability"
      "actual_outcome": None,                    # filled in by _close_position()
      "brier_contribution": None,                # filled in by evaluator.py
      "cross_ref_metaculus": None,               # filled in if Phase 2A is active
      "cross_ref_manifold": None,                # filled in if Phase 2A is active

Phase 4 — Evaluation Log Write
  After _close_position() computes real P&L and actual_outcome:
    from evaluator import record_resolved_trade
    record_resolved_trade(trade_record)
  This writes one line to logs/evaluation_log.jsonl for the weekly Sonnet review.
"""

import json
import logging
import os
from datetime import datetime, timezone

from config import Config
from forecaster import ForecastResult
from risk_manager import RiskDecision

logger = logging.getLogger(__name__)

PAPER_PORTFOLIO_FILE = Config.PAPER_PORTFOLIO_FILE
TRADES_LOG_FILE = Config.TRADES_LOG_FILE

# Read PAPER_TRADING once at module import — never changes mid-session
IS_PAPER_TRADING = Config.PAPER_TRADING


def _load_portfolio() -> dict:
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(PAPER_PORTFOLIO_FILE):
        try:
            with open(PAPER_PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "balance": 1000.0,
        "peak_balance": 1000.0,
        "open_positions": {},
        "trade_history": [],
    }


def _save_portfolio(portfolio: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(PAPER_PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


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

    # Log rationale BEFORE executing
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
        _log_trade(trade_record)  # log once only, after confirmed execution

    return trade_record if result else None


def _execute_paper(decision: RiskDecision, f: ForecastResult) -> bool:
    """Simulate order execution and update the paper portfolio."""
    portfolio = _load_portfolio()

    entry_price = f.yes_price if f.side == "YES" else f.no_price
    shares = decision.position_size_usdc / entry_price if entry_price > 0 else 0

    portfolio["balance"] -= decision.position_size_usdc

    position = {
        "market_id": f.market_id,
        "question": f.question,
        "side": f.side,
        "size_usdc": decision.position_size_usdc,
        "shares": shares,
        "entry_price": entry_price,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "days_to_expiry_at_entry": f.days_to_expiry,
        "condition_id": f.condition_id,
        "token_ids": f.token_ids,
    }
    portfolio["open_positions"][f.market_id] = position
    # peak_balance is updated in _close_position() when cash returns after a win

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
        from py_clob_client.clob_types import OrderArgs, OrderType

        client = ClobClient(
            host=Config.CLOB_API_BASE,
            key=Config.POLYMARKET_PRIVATE_KEY,
            chain_id=137,  # Polygon
            funder=Config.POLYMARKET_FUNDER_ADDRESS,
            signature_type=2,  # EIP-712
        )

        if f.side == "YES":
            token_id = f.token_ids[0] if f.token_ids else ""
            price = f.yes_price
        else:
            token_id = f.token_ids[1] if len(f.token_ids) > 1 else ""
            price = f.no_price

        size = round(decision.position_size_usdc / price, 2) if price > 0 else 0

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
        )

        resp = client.create_and_post_order(order_args)
        logger.info("[LIVE] Order placed: %s", resp)

        # Update paper portfolio for tracking (even in live mode we track positions)
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
        }
        portfolio["open_positions"][f.market_id] = position
        portfolio["balance"] -= decision.position_size_usdc
        _save_portfolio(portfolio)

        return True

    except Exception as exc:
        logger.error("[LIVE] Order placement failed: %s", exc)
        return False


def monitor_open_positions() -> list[dict]:
    """
    Check open positions for stop-loss triggers and time-based exits.
    Returns list of closed position records.
    """
    portfolio = _load_portfolio()
    open_positions: dict = portfolio.get("open_positions", {})
    closed = []

    for market_id, pos in list(open_positions.items()):
        try:
            close_reason = None

            # Time-based exit: close positions 4 hours before expiry (or already past expiry)
            entry_time_raw = datetime.fromisoformat(pos["entry_time"])
            # Ensure timezone-aware comparison regardless of how entry_time was stored
            if entry_time_raw.tzinfo is None:
                entry_time_raw = entry_time_raw.replace(tzinfo=timezone.utc)
            days_at_entry = pos.get("days_to_expiry_at_entry", 999)
            now = datetime.now(timezone.utc)
            elapsed_days = (now - entry_time_raw).total_seconds() / 86400
            remaining_days = days_at_entry - elapsed_days

            if remaining_days <= (4 / 24):
                # Covers both "4h before expiry" and "already past expiry" (bot was offline)
                close_reason = "time_exit_4h_before_expiry" if remaining_days > 0 else "past_expiry"

            # For paper trading, we can't check live price easily — rely on expiry
            # In live mode: check current price vs stop-loss
            if not IS_PAPER_TRADING and close_reason is None:
                current_price = _get_current_price(pos)
                if current_price is not None and current_price < pos.get("stop_loss_price", 0):
                    close_reason = f"stop_loss_triggered (price={current_price:.3f})"

            if close_reason:
                pnl = _close_position(portfolio, market_id, pos, close_reason)
                closed.append({"market_id": market_id, "question": pos["question"], "pnl": pnl, "reason": close_reason})
                logger.info("Closed position %s: reason=%s pnl=%.2f", market_id, close_reason, pnl)

        except Exception as exc:
            logger.warning("Error monitoring position %s: %s", market_id, exc)

    if closed:
        _save_portfolio(portfolio)

    return closed


def _get_current_price(pos: dict) -> float | None:
    """Fetch current market price for a live position."""
    try:
        import requests
        token_id = pos.get("token_ids", [None])[0] if pos.get("side") == "YES" else pos.get("token_ids", [None, None])[1]
        if not token_id:
            return None
        resp = requests.get(
            f"https://clob.polymarket.com/price",
            params={"token_id": token_id, "side": "buy"},
            timeout=10,
        )
        return float(resp.json().get("price", 0))
    except Exception:
        return None


def _close_position(portfolio: dict, market_id: str, pos: dict, reason: str) -> float:
    """Remove position from portfolio and calculate P&L."""
    entry_price = pos.get("entry_price", 0)
    shares = pos.get("shares", 0)
    size_usdc = pos.get("size_usdc", 0)

    # For paper mode, assume we exit at current market-implied price (simplified)
    # A real exit would fetch the current price and sell
    # BUG NOTE: exit_price = entry_price means pnl is always 0.
    # Every daily summary will show $0 P&L and every win/loss metric is meaningless.
    # Fix: Phase 4 — fetch actual outcomePrices from Gamma API and compute real binary P&L.
    # Binary P&L formula:
    #   WIN:  size_usdc * (1.0 / entry_price - 1.0)
    #   LOSS: -size_usdc
    # Do not close position until Gamma confirms resolution (outcomePrices > 0.9 or < 0.1).
    exit_price = entry_price  # neutral assumption — actual outcome determines real P&L
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
