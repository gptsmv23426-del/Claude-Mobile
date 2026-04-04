"""
Market scanner — connects to Polymarket CLOB API and returns filtered opportunities.
"""

import logging
from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel

from config import Config

logger = logging.getLogger(__name__)

# Polymarket Gamma API base URL for market data
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"


class MarketOpportunity(BaseModel):
    market_id: str
    question: str
    category: str
    yes_price: float
    no_price: float
    volume_usd: float
    days_to_expiry: float
    spread: float
    condition_id: str = ""
    token_ids: List[str] = []


def _get_days_to_expiry(end_date_iso: str) -> float:
    """Return days remaining until market expiry from an ISO timestamp string."""
    try:
        end_dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = end_dt - now
        return max(delta.total_seconds() / 86400, 0.0)
    except Exception:
        return 0.0


def scan_markets() -> List[MarketOpportunity]:
    """
    Fetch active markets from Polymarket Gamma API, apply filters, and return
    up to 20 MarketOpportunity objects.
    """
    import requests

    opportunities: List[MarketOpportunity] = []

    params = {
        "active": "true",
        "closed": "false",
        "limit": 200,
        "order": "volume24hr",
        "ascending": "false",
    }

    try:
        resp = requests.get(f"{GAMMA_API_BASE}/markets", params=params, timeout=30)
        resp.raise_for_status()
        markets = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch markets from Gamma API: %s", exc)
        return []

    if isinstance(markets, dict) and "data" in markets:
        markets = markets["data"]

    for m in markets:
        if len(opportunities) >= 20:
            break

        try:
            category = (m.get("category") or "UNKNOWN").upper()
            if category in Config.SKIP_CATEGORIES:
                continue
            if category not in Config.PREFERRED_CATEGORIES:
                continue

            volume = float(m.get("volume", 0) or 0)
            if volume < Config.MIN_MARKET_VOLUME_USD:
                continue

            end_date = m.get("endDate") or m.get("end_date_iso") or ""
            days_to_expiry = _get_days_to_expiry(end_date)
            if not (1 <= days_to_expiry <= 120):
                continue

            # Parse orderbook prices from tokens
            tokens = m.get("tokens") or []
            yes_price, no_price = None, None
            token_ids = []
            for token in tokens:
                outcome = (token.get("outcome") or "").upper()
                price = float(token.get("price", 0) or 0)
                token_ids.append(token.get("token_id", ""))
                if outcome == "YES":
                    yes_price = price
                elif outcome == "NO":
                    no_price = price

            if yes_price is None or no_price is None:
                logger.debug(
                    "Skipping market %s: missing YES or NO token price in API response.",
                    m.get("id", ""),
                )
                continue

            if yes_price <= 0 or yes_price >= 1:
                continue

            spread = abs(1.0 - yes_price - no_price)
            if spread > Config.MAX_SPREAD:
                continue

            opp = MarketOpportunity(
                market_id=str(m.get("id", "")),
                question=m.get("question", ""),
                category=category,
                yes_price=yes_price,
                no_price=no_price,
                volume_usd=volume,
                days_to_expiry=round(days_to_expiry, 2),
                spread=round(spread, 4),
                condition_id=m.get("conditionId", ""),
                token_ids=token_ids,
            )
            opportunities.append(opp)
            logger.debug("Accepted market: %s (vol=%.0f, spread=%.3f)", opp.question[:60], volume, spread)

        except Exception as exc:
            logger.warning("Skipping malformed market entry: %s", exc)
            continue

    logger.info("Scanner found %d qualifying markets.", len(opportunities))
    return opportunities


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    results = scan_markets()
    for r in results[:5]:
        print(f"\n{r.question}")
        print(f"  Category: {r.category}  Volume: ${r.volume_usd:,.0f}  Spread: {r.spread:.3f}")
        print(f"  YES: {r.yes_price:.3f}  NO: {r.no_price:.3f}  Days: {r.days_to_expiry:.1f}")
