"""
FRED Macro Context — fetches 5 macro indicators with 24h file cache.

Requires:
  pip install fredapi        (uncomment fredapi in requirements.txt)
  FRED_API_KEY in .env      (free at fred.stlouisfed.org)
  ENABLE_MACRO_CONTEXT=true in .env

Integration point: researcher.py injects the formatted snapshot into the system
prompt for MACRO category markets when ENABLE_MACRO_CONTEXT=true.

Graceful degradation: returns None if fredapi not installed, key missing, or any
fetch fails. Never raises — macro context is an optional enrichment signal.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)

_CACHE_FILE = "logs/macro_cache.json"
_CACHE_TTL_SECONDS = 86400  # 24 hours

_FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "sp500": "SP500",
    "yield_10y": "DGS10",
}


@dataclass
class MacroSnapshot:
    fed_funds: Optional[float]    # Fed Funds Rate (%)
    cpi: Optional[float]          # CPI index level
    unemployment: Optional[float] # Unemployment Rate (%)
    sp500: Optional[float]        # S&P 500 level
    yield_10y: Optional[float]    # 10-Year Treasury Yield (%)
    fetched_at: str               # ISO UTC timestamp


def _load_cache() -> Optional[MacroSnapshot]:
    """Load cached snapshot if it exists and is < 24h old."""
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE) as f:
            data = json.load(f)
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
        if age < _CACHE_TTL_SECONDS:
            return MacroSnapshot(**data)
    except Exception as exc:
        logger.debug("Macro cache unreadable: %s", exc)
    return None


def _save_cache(snapshot: MacroSnapshot) -> None:
    os.makedirs("logs", exist_ok=True)
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(asdict(snapshot), f)
    except Exception as exc:
        logger.debug("Failed to save macro cache: %s", exc)


def _fetch_from_fred() -> Optional[MacroSnapshot]:
    """Fetch all 5 series from FRED. Returns None if fredapi unavailable or key missing."""
    if not Config.FRED_API_KEY:
        logger.debug("FRED_API_KEY not set — macro context disabled.")
        return None
    try:
        import fredapi
        fred = fredapi.Fred(api_key=Config.FRED_API_KEY)

        values: dict = {}
        for field, series_id in _FRED_SERIES.items():
            try:
                series = fred.get_series(series_id)
                last_val = series.dropna().iloc[-1] if not series.dropna().empty else None
                values[field] = round(float(last_val), 4) if last_val is not None else None
            except Exception as exc:
                logger.debug("FRED series %s failed: %s", series_id, exc)
                values[field] = None

        snapshot = MacroSnapshot(**values, fetched_at=datetime.now(timezone.utc).isoformat())
        _save_cache(snapshot)
        logger.info("FRED macro snapshot fetched: %s", values)
        return snapshot

    except ImportError:
        logger.debug("fredapi not installed. Run: pip install fredapi")
        return None
    except Exception as exc:
        logger.warning("FRED fetch failed: %s", exc)
        return None


def get_macro_snapshot() -> Optional[MacroSnapshot]:
    """
    Return current macro snapshot. Uses 24h file cache to avoid repeated API calls.
    Returns None if fredapi not installed, API key missing, or all fetches fail.
    """
    if not Config.ENABLE_MACRO_CONTEXT or not Config.FRED_API_KEY:
        return None

    cached = _load_cache()
    if cached:
        logger.debug("Using cached macro snapshot (age < 24h).")
        return cached

    return _fetch_from_fred()


def format_macro_for_prompt(snapshot: MacroSnapshot) -> str:
    """Format a MacroSnapshot as a concise string for injection into research prompts."""
    parts = []
    if snapshot.fed_funds is not None:
        parts.append(f"Fed Funds Rate: {snapshot.fed_funds:.2f}%")
    if snapshot.cpi is not None:
        parts.append(f"CPI: {snapshot.cpi:.1f}")
    if snapshot.unemployment is not None:
        parts.append(f"Unemployment: {snapshot.unemployment:.1f}%")
    if snapshot.sp500 is not None:
        parts.append(f"S&P 500: {snapshot.sp500:,.0f}")
    if snapshot.yield_10y is not None:
        parts.append(f"10Y Treasury: {snapshot.yield_10y:.2f}%")
    if not parts:
        return ""
    return "Current macro environment: " + ", ".join(parts)
