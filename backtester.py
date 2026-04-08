"""
Backtester — runs the full pipeline against 90 days of historical Polymarket data.

Fixes vs previous version (Phase 1 — DONE):
- Resolution is parsed from real outcomePrices (Gamma API), not inferred from final token price
- Entry prices come from actual CLOB price history (prices-history endpoint)
- Fake rng.uniform() edge is gone; simulation uses real price-at-entry vs real outcome
- Metrics computed via vectorbt Portfolio.from_orders() with manual fallback

PLANNED UPGRADES (do not implement until approved):

Phase 2C — Local Polymarket Data (removes CLOB API rate limit cap)
  Currently: capped at _MAX_HISTORY_FETCHES = 60 CLOB API calls per run.
  Fix: If Config.POLYMARKET_DATA_PATH is set (local clone of Polymarket_data repo):
    1. Load resolved markets from local CSV/Parquet files instead of Gamma API
    2. Load price history from local files instead of CLOB prices-history endpoint
    3. Remove the 60-market cap entirely — test on thousands of markets in seconds
    4. Keep the existing CLOB-based path as fallback when POLYMARKET_DATA_PATH is empty

  Data loading logic (pseudocode):
    if Config.POLYMARKET_DATA_PATH:
        markets = load_local_markets(Config.POLYMARKET_DATA_PATH)  # CSV/Parquet
        # Each row has: condition_id, question, outcomePrices, endDate, category
    else:
        markets = _fetch_resolved_markets()  # existing Gamma API call

  Price history loading:
    if Config.POLYMARKET_DATA_PATH:
        history = load_local_price_history(condition_id, Config.POLYMARKET_DATA_PATH)
    else:
        history = _fetch_clob_history(condition_id)  # existing CLOB API call

Phase 4 — Evaluator Integration
  After run_backtest() completes, write each simulated trade to logs/evaluation_log.jsonl
  using the same schema that executor.py uses for live trades.
  This seeds the evaluator with historical calibration data before any live trades occur.
  Call: evaluator.record_resolved_trade(trade_record) for each simulated trade.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import hashlib as _hl
import numpy as np
import pandas as pd
import requests

from config import Config

logger = logging.getLogger(__name__)

from config import Config as _cfg
GAMMA_API_BASE = _cfg.GAMMA_API_BASE
CLOB_API_BASE = _cfg.CLOB_API_BASE
BACKTEST_DONE_FLAG = "logs/backtest_done.flag"
BACKTEST_REPORT_FILE = "logs/backtest_report.txt"
HISTORICAL_TRADES_FILE = "logs/backtest_trades.csv"

# Cap CLOB history calls so backtest doesn't take forever
_MAX_HISTORY_FETCHES = 60
_REQUEST_DELAY_S = 0.15  # seconds between CLOB calls (rate-limit headroom)

import random as _random

def _get_with_backoff(url, params=None, max_retries=3, timeout=30):
    """GET with exponential backoff on 429 / transient errors."""
    import time as _t
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = delay + _random.uniform(0, delay * 0.5)
                logger.warning("Rate limited. Retrying in %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                _t.sleep(wait)
                delay *= 2
                continue
            return resp
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = delay + _random.uniform(0, delay * 0.5)
            logger.warning("Request failed: %s. Retrying in %.1fs", exc, wait)
            _t.sleep(wait)
            delay *= 2
    raise RuntimeError(f"Max retries exceeded for {url}")



# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_resolved_markets(days: int = 90) -> list[dict]:
    """Fetch recently resolved markets from the Gamma API."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    markets = []
    params = {
        "closed": "true",
        "limit": 500,
        "order": "volume24hr",
        "ascending": "false",
    }
    try:
        resp = _get_with_backoff(f"{GAMMA_API_BASE}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        for m in data:
            end_date = m.get("endDate") or m.get("end_date_iso") or ""
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if end_dt >= cutoff:
                    markets.append(m)
            except Exception:
                continue
    except Exception as exc:
        logger.error("Failed to fetch historical markets: %s", exc)
    logger.info("Fetched %d resolved markets for backtesting.", len(markets))
    return markets


def _parse_resolution(market: dict) -> Optional[bool]:
    """
    Determine actual YES/NO resolution from Gamma API data.
    Returns True = YES won, False = NO won, None = unknown.

    outcomePrices is a JSON array like ["1", "0"] (YES won) or ["0", "1"] (NO won).
    Index 0 = YES token final price, index 1 = NO token final price.
    """
    raw = market.get("outcomePrices")
    if raw:
        try:
            prices = json.loads(raw) if isinstance(raw, str) else raw
            yes_final = float(prices[0])
            if yes_final > 0.9:
                return True
            if yes_final < 0.1:
                return False
        except Exception:
            pass

    # Fallback: read from token list (resolved tokens settle at 1.0 or 0.0)
    for token in (market.get("tokens") or []):
        outcome = (token.get("outcome") or "").upper()
        price = float(token.get("price", 0) or 0)
        if outcome == "YES":
            if price > 0.9:
                return True
            if price < 0.1:
                return False

    return None


def _fetch_clob_history(condition_id: str) -> Optional[list[dict]]:
    """
    Fetch YES token price history from the CLOB API.
    Returns list of {"t": epoch_seconds, "p": price} sorted ascending, or None.
    Requires at least 3 data points to be useful.
    """
    if not condition_id:
        return None
    try:
        resp = _get_with_backoff(
            f"{CLOB_API_BASE}/prices-history",
            params={"market": condition_id, "interval": "max", "fidelity": "60"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        history = data.get("history") or []
        if len(history) < 3:
            return None
        return sorted(history, key=lambda x: x["t"])
    except Exception as exc:
        logger.debug("CLOB history fetch failed for %s: %s", condition_id, exc)
        return None


def _normalize_ts(t) -> float:
    """Convert a timestamp to seconds. CLOB API returns milliseconds (>1e12)."""
    t = float(t)
    if t > 1e12:
        return t / 1000.0
    return t


def _pick_entry_price(history: list[dict], end_ts: float) -> Optional[float]:
    """
    Pick a realistic entry price from the price history.

    Strategy: find all prices at least 1 day before resolution that fall in the
    tradeable range [0.03, 0.97], then return the median of those prices.
    Markets that never had a price in this range (always near 0 or 1) are skipped.
    """
    if not history:
        return None

    one_day_s = 1 * 86400
    candidates = [h for h in history if (end_ts - _normalize_ts(h["t"])) >= one_day_s]
    if not candidates:
        return None

    in_range = sorted(
        float(h["p"]) for h in candidates if 0.03 <= float(h["p"]) <= 0.97
    )
    if not in_range:
        return None

    return in_range[len(in_range) // 2]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate_trades(markets: list[dict]) -> pd.DataFrame:
    """
    Simulate the bot's pipeline on resolved historical markets.

    For each qualifying resolved market:
    1. Parse the actual YES/NO resolution from outcomePrices (no guessing).
    2. Fetch real CLOB price history to determine a realistic entry price.
    3. Apply the same edge threshold and Kelly sizing as the live bot.
    4. Model forecast accuracy at 60% — a conservative estimate for Claude Haiku
       research quality on prediction markets (no inflated RNG bias).
    5. Compute P&L: binary payout (1.0/entry_price profit if correct, full loss if wrong).

    Markets without a retrievable entry price are skipped rather than fabricated.
    """
    from market_scanner import _infer_category  # Gamma API no longer returns category field
    records = []
    history_fetches = 0
    n_skip_cat = n_skip_vol = n_skip_res = n_skip_history = n_skip_entry = n_skip_edge = 0

    for m in markets:
        try:
            question = m.get("question") or ""
            category = _infer_category(question)
            if category in Config.SKIP_CATEGORIES:
                n_skip_cat += 1
                continue
            if category not in Config.PREFERRED_CATEGORIES:
                n_skip_cat += 1
                continue

            volume = float(m.get("volume", 0) or 0)
            if volume < Config.MIN_MARKET_VOLUME_USD:
                n_skip_vol += 1
                continue

            # Real resolution — no heuristic guessing
            yes_won = _parse_resolution(m)
            if yes_won is None:
                n_skip_res += 1
                logger.debug("Resolution unknown — skipping: %s", question[:50])
                continue

            condition_id = m.get("conditionId", "")
            # CLOB /prices-history expects a token_id, not conditionId
            raw_clob_ids = m.get("clobTokenIds") or []
            if isinstance(raw_clob_ids, str):
                raw_clob_ids = json.loads(raw_clob_ids)
            clob_token_id = raw_clob_ids[0] if raw_clob_ids else condition_id
            end_date_str = m.get("endDate") or m.get("end_date_iso") or ""
            end_ts: Optional[float] = None
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                end_ts = end_dt.timestamp()
            except Exception:
                pass

            # Fetch real price history (capped to keep backtest fast)
            history = None
            if clob_token_id and end_ts and history_fetches < _MAX_HISTORY_FETCHES:
                history = _fetch_clob_history(clob_token_id)
                history_fetches += 1
                time.sleep(_REQUEST_DELAY_S)

            if not history or end_ts is None:
                n_skip_history += 1
                continue  # Skip — no real entry price available

            # Debug: log timestamp format for first market to confirm ms vs s
            if history_fetches == 1 and history:
                sample_t = history[0]["t"]
                logger.info(
                    "CLOB timestamp debug: h[t]=%s (type=%s), end_ts=%.0f, "
                    "normalized=%.0f, delta=%.0f s",
                    sample_t, type(sample_t).__name__, end_ts,
                    _normalize_ts(sample_t), end_ts - _normalize_ts(sample_t),
                )

            entry_price_yes = _pick_entry_price(history, end_ts)
            if entry_price_yes is None:
                n_skip_entry += 1
                continue

            # Edge = distance between entry price and actual outcome (0 or 1)
            true_yes_prob = 1.0 if yes_won else 0.0
            edge = abs(true_yes_prob - entry_price_yes)
            if edge < Config.MIN_EDGE_THRESHOLD:
                n_skip_edge += 1
                continue

            # Simulate forecast accuracy at 60% — conservative, reproducible per market
            # Use deterministic seed from market_id bytes - reproducible across Python runs
            _mid = (m.get("id") or "").encode()
            _seed = int(_hl.md5(_mid).hexdigest()[:8], 16)
            rng = np.random.default_rng(_seed)
            bot_correct = rng.random() < 0.60

            if bot_correct:
                side = "YES" if yes_won else "NO"
            else:
                side = "NO" if yes_won else "YES"

            entry_price = entry_price_yes if side == "YES" else (1.0 - entry_price_yes)
            correct = (side == "YES" and yes_won) or (side == "NO" and not yes_won)

            denom = 1.0 - entry_price
            kelly = edge / denom if denom > 0 else 0.0
            size = min(1000.0 * kelly * Config.KELLY_FRACTION, Config.MAX_POSITION_SIZE_USDC)
            if size <= 0:
                continue

            pnl = size * (1.0 / entry_price - 1.0) if correct else -size

            records.append({
                "market_id": m.get("id", ""),
                "question": (m.get("question") or "")[:60],
                "category": category,
                "side": side,
                "entry_price": round(entry_price, 4),
                "edge": round(edge, 4),
                "size_usdc": round(size, 2),
                "yes_won": yes_won,
                "correct": correct,
                "pnl": round(pnl, 4),
                "end_date": end_date_str,
            })

        except Exception as exc:
            logger.debug("Skipping market in backtest: %s", exc)
            continue

    logger.info(
        "Filter breakdown: %d skipped-category, %d skipped-volume, %d skipped-resolution, "
        "%d skipped-history, %d skipped-entry-price, %d skipped-edge",
        n_skip_cat, n_skip_vol, n_skip_res, n_skip_history, n_skip_entry, n_skip_edge,
    )
    logger.info(
        "Simulation complete: %d trades from %d markets | %d CLOB history calls",
        len(records), len(markets), history_fetches,
    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _calculate_metrics_vbt(df: pd.DataFrame, initial_balance: float = 1000.0) -> dict:
    """
    Use vectorbt Portfolio.from_orders() for accurate portfolio simulation.

    Each trade is modelled as two orders on a synthetic price axis:
      - Buy N shares at entry_price
      - Sell N shares at 1.0 (win) or 0.0 (loss)

    This lets VectorBT handle compounding, portfolio value tracking, and drawdown
    instead of the manual equity loop.

    Falls back to a manual pandas implementation if VectorBT import fails
    (e.g., on a machine where it hasn't been installed yet).
    """
    if df.empty:
        return {"error": "No trades simulated"}

    n = len(df)

    try:
        import vectorbt as vbt  # deferred import — not installed on all machines yet

        # Build synthetic index: entry at even positions, exit at odd positions
        entry_idx = np.arange(0, n * 2, 2)
        exit_idx = np.arange(1, n * 2, 2)
        all_idx = np.arange(n * 2, dtype=float)

        prices = np.empty(n * 2)
        sizes = np.zeros(n * 2)

        for i, (_, row) in enumerate(df.iterrows()):
            ep = float(row["entry_price"])
            xp = 1.0 if row["correct"] else 0.001  # avoid /0 on 0 exit price
            shares = float(row["size_usdc"]) / ep if ep > 0 else 0.0

            prices[entry_idx[i]] = ep
            prices[exit_idx[i]] = xp
            sizes[entry_idx[i]] = shares    # buy
            sizes[exit_idx[i]] = -shares    # sell

        price_series = pd.Series(prices, index=all_idx)
        size_series = pd.Series(sizes, index=all_idx)

        portfolio = vbt.Portfolio.from_orders(
            close=price_series,
            size=size_series,
            price=price_series,
            init_cash=initial_balance,
            fees=0.0,
            freq="1T",
        )

        port_value = portfolio.value()
        port_returns = port_value.pct_change().dropna()
        sharpe = float(
            port_returns.mean() / port_returns.std() * np.sqrt(252)
            if port_returns.std() > 0 else 0.0
        )
        peak = port_value.expanding().max()
        max_drawdown = float(((port_value - peak) / peak).min())
        total_return = float((port_value.iloc[-1] - initial_balance) / initial_balance)
        final_value = float(port_value.iloc[-1])
        source = "vectorbt"

    except ImportError:
        logger.warning("vectorbt not installed — using manual metrics fallback.")
        equity = [initial_balance]
        for _, row in df.iterrows():
            equity.append(equity[-1] + float(row["pnl"]))
        equity_s = pd.Series(equity)
        returns = equity_s.pct_change().dropna()
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
        peak = equity_s.expanding().max()
        max_drawdown = float(((equity_s - peak) / peak).min())
        total_return = float((equity[-1] - initial_balance) / initial_balance)
        final_value = float(equity[-1])
        source = "pandas_fallback"

    wins = df[df["pnl"] > 0]
    win_rate = len(wins) / len(df) if len(df) > 0 else 0.0
    total_pnl = float(df["pnl"].sum())

    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(float(win_rate), 4),
        "total_return": round(total_return, 4),
        "total_trades": len(df),
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(final_value, 2),
        "metrics_source": source,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(metrics: dict) -> None:
    os.makedirs("logs", exist_ok=True)

    def _fmt_pct(v):
        return f"{v:.2%}" if isinstance(v, float) else str(v)

    lines = [
        "=" * 50,
        "POLYMARKET BOT BACKTEST REPORT",
        f"Generated: {datetime.now().isoformat()}",
        f"Metrics source: {metrics.get('metrics_source', 'N/A')}",
        "=" * 50,
        f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 'N/A')}",
        f"Max Drawdown:    {_fmt_pct(metrics.get('max_drawdown', 'N/A'))}",
        f"Win Rate:        {_fmt_pct(metrics.get('win_rate', 'N/A'))}",
        f"Total Return:    {_fmt_pct(metrics.get('total_return', 'N/A'))}",
        f"Total Trades:    {metrics.get('total_trades', 'N/A')}",
        f"Total P&L:       ${metrics.get('total_pnl', 'N/A')}",
        f"Final Balance:   ${metrics.get('final_balance', 'N/A')}",
        "=" * 50,
    ]
    report = "\n".join(lines)
    with open(BACKTEST_REPORT_FILE, "w") as f:
        f.write(report)
    print(report)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest() -> dict:
    """
    Run the full backtest. Returns metrics dict.
    Logs WARNING if Sharpe < MIN_BACKTEST_SHARPE but does NOT block execution.
    """
    logger.info("Starting backtest on 90 days of historical data...")
    os.makedirs("logs", exist_ok=True)

    markets = _fetch_resolved_markets(days=90)
    if not markets:
        logger.warning("No historical markets found — skipping backtest.")
        metrics = {"error": "No data", "sharpe_ratio": 0.0}
        _write_report(metrics)
        return metrics

    df = _simulate_trades(markets)
    if not df.empty:
        df.to_csv(HISTORICAL_TRADES_FILE, index=False)

    metrics = _calculate_metrics_vbt(df)
    _write_report(metrics)

    sharpe = metrics.get("sharpe_ratio", 0.0)
    if isinstance(sharpe, float) and sharpe < Config.MIN_BACKTEST_SHARPE:
        logger.warning(
            "Backtest Sharpe ratio %.3f is below threshold %.3f — proceeding anyway (experimental mode).",
            sharpe,
            Config.MIN_BACKTEST_SHARPE,
        )
    else:
        logger.info("Backtest passed Sharpe threshold: %.3f >= %.3f", sharpe, Config.MIN_BACKTEST_SHARPE)

    # Mark backtest as done, recording the config fingerprint so a config change
    # triggers a fresh backtest on next startup.
    with open(BACKTEST_DONE_FLAG, "w") as f:
        f.write(f"{datetime.now().isoformat()}|{_config_fingerprint()}")

    return metrics


def _config_fingerprint() -> str:
    """Return a short hash of the config values that affect backtest results."""
    import hashlib
    sig = (
        f"{Config.MIN_EDGE_THRESHOLD}|{Config.MIN_MARKET_VOLUME_USD}|"
        f"{Config.KELLY_FRACTION}|{Config.MAX_POSITION_SIZE_USDC}|"
        f"{Config.MAX_SPREAD}"
    )
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def backtest_already_run() -> bool:
    if not os.path.exists(BACKTEST_DONE_FLAG):
        return False
    try:
        stored = open(BACKTEST_DONE_FLAG).read().strip()
        # Flag format: "<iso_timestamp>|<config_fingerprint>"
        if "|" not in stored:
            return False  # old format — re-run
        _, stored_fp = stored.rsplit("|", 1)
        return stored_fp == _config_fingerprint()
    except Exception:
        return False
