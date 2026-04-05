"""
Calibration tracker — logs every approved forecast and records resolution outcomes.

Why this matters: Claude's probability estimates need empirical validation.
This module tracks every forecast we act on, then polls Polymarket for when
those markets resolve, so we can measure per-category accuracy over time.

Log file: logs/calibration_log.json (a JSON array, fully rewritten on each update)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

CALIBRATION_LOG_FILE = "logs/calibration_log.json"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def _load_log() -> list:
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(CALIBRATION_LOG_FILE):
        return []
    try:
        with open(CALIBRATION_LOG_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.error("Calibration log unreadable (%s) — returning empty list.", exc)
        return []


def _save_log(entries: list) -> None:
    os.makedirs("logs", exist_ok=True)
    try:
        with open(CALIBRATION_LOG_FILE, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception as exc:
        logger.error("Failed to save calibration log: %s", exc)


def log_forecast(forecast) -> None:
    """
    Append a new entry for an approved forecast.
    Call this in main.py immediately after execute_trade() returns a non-None record.

    Duplicate entries for the same market_id are silently skipped — safe to call
    multiple times across restarts.
    """
    entries = _load_log()

    # Skip if already logged and unresolved
    if any(e["market_id"] == forecast.market_id for e in entries):
        logger.debug("Calibration: skipping duplicate entry for %s", forecast.market_id)
        return

    entries.append({
        "market_id": forecast.market_id,
        "question": forecast.question,
        "category": getattr(forecast, "category", "UNKNOWN"),
        "our_probability": forecast.probability,
        "market_price": forecast.yes_price,
        "edge": forecast.edge,
        "side": forecast.side,
        "confidence": forecast.confidence,
        "evidence_quality": forecast.evidence_quality,
        "forecasted_at": datetime.now(timezone.utc).isoformat(),
        "resolved_yes": None,
        "resolved_at": None,
    })

    _save_log(entries)
    logger.info("Calibration: logged forecast for market %s", forecast.market_id)


def update_resolution(market_id: str, resolved_yes: bool) -> None:
    """
    Manually record the resolution outcome for a previously logged forecast.
    Used when resolution is detected through a source other than the polling loop.
    """
    entries = _load_log()
    for entry in entries:
        if entry["market_id"] == market_id and entry["resolved_yes"] is None:
            entry["resolved_yes"] = resolved_yes
            entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
            _save_log(entries)
            logger.info(
                "Calibration: manually updated resolution for %s — resolved_yes=%s",
                market_id,
                resolved_yes,
            )
            return
    logger.debug("Calibration: no pending entry found for market_id %s", market_id)


def check_and_update_resolutions() -> int:
    """
    Poll Gamma API for markets that closed since we logged them.
    Determines resolution by whether the YES token settled at >= 0.99 (YES won)
    or <= 0.01 (NO won). Markets settled at intermediate prices are skipped.

    Returns the count of newly resolved entries.
    Called once per trading cycle in main.py.
    """
    entries = _load_log()
    pending = [e for e in entries if e["resolved_yes"] is None]
    if not pending:
        return 0

    pending_ids = {e["market_id"] for e in pending}

    try:
        resp = requests.get(
            f"{GAMMA_API_BASE}/markets",
            params={"closed": "true", "limit": 500, "order": "end_date_iso", "ascending": "false"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
    except Exception as exc:
        logger.error("check_and_update_resolutions: Gamma API fetch failed: %s", exc)
        return 0

    # Determine resolution for each closed market we care about
    resolutions: Dict[str, bool] = {}
    for m in data:
        mid = str(m.get("id", ""))
        if mid not in pending_ids:
            continue
        tokens = m.get("tokens") or []
        for t in tokens:
            if (t.get("outcome") or "").upper() == "YES":
                price = float(t.get("price") or -1)
                if price >= 0.99:
                    resolutions[mid] = True
                elif price <= 0.01:
                    resolutions[mid] = False
                # If price is between 0.01 and 0.99, market may not be fully settled
                break

    updated = 0
    for entry in entries:
        mid = entry["market_id"]
        if entry["resolved_yes"] is None and mid in resolutions:
            entry["resolved_yes"] = resolutions[mid]
            entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1
            logger.info(
                "Calibration: market %s resolved YES=%s", mid, resolutions[mid]
            )

    if updated:
        _save_log(entries)
        logger.info("Calibration: resolved %d market(s) in this poll.", updated)

    return updated


def get_category_stats() -> Dict[str, dict]:
    """
    Compute per-category accuracy stats for all resolved forecasts.

    A forecast is "correct" if the side we bet on (YES or NO) turned out to
    be the winning side. This measures actual trading accuracy, not just
    whether our probability was directionally right.

    Returns: {category: {n_resolved, n_correct, accuracy, mean_edge, mean_probability}}
    """
    entries = _load_log()
    resolved = [e for e in entries if e["resolved_yes"] is not None]

    buckets: Dict[str, list] = {}
    for e in resolved:
        cat = e.get("category", "UNKNOWN")
        buckets.setdefault(cat, []).append(e)

    result = {}
    for cat, cat_entries in buckets.items():
        n = len(cat_entries)
        n_correct = sum(
            1 for e in cat_entries
            if (e["side"] == "YES" and e["resolved_yes"])
            or (e["side"] == "NO" and not e["resolved_yes"])
        )
        result[cat] = {
            "n_resolved": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 4) if n > 0 else None,
            "mean_edge": round(sum(e["edge"] for e in cat_entries) / n, 4) if n > 0 else None,
            "mean_probability": round(sum(e["our_probability"] for e in cat_entries) / n, 4) if n > 0 else None,
        }

    return result
