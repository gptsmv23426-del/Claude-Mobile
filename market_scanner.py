"""
Market scanner — connects to Polymarket Gamma API and returns filtered opportunities.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

from pydantic import BaseModel, Field

from config import Config
from monitor import alert_error
from cross_platform import get_cross_platform_prices

logger = logging.getLogger(__name__)

GAMMA_API_BASE = Config.GAMMA_API_BASE
CLOB_API_BASE = Config.CLOB_API_BASE


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
    cross_platform_divergence: float = 0.0
    cross_platform_prices: Dict[str, float] = Field(default_factory=dict)


def _get_days_to_expiry(end_date_iso: str) -> float:
    """Return days remaining until market expiry from an ISO timestamp string."""
    try:
        end_dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = end_dt - now
        return max(delta.total_seconds() / 86400, 0.0)
    except Exception:
        return 0.0


def _infer_category(question: str) -> str:
    """
    Infer market category from question text.
    The Gamma API no longer returns a category field, so we derive it from keywords.
    """
    q = question.lower()
    if any(w in q for w in ["bitcoin", "btc", "eth", "ethereum", "crypto", "solana", "sol", "coin", "token", "blockchain", "defi"]):
        return "CRYPTO"
    if any(w in q for w in ["nba", "nfl", "nhl", "mlb", " vs ", "vs.", "celtics", "lakers", "warriors", "knicks",
                             "yankees", "dodgers", "oilers", "bucks", "hornets", "nets", "heat", "bulls",
                             "championship", "super bowl", "world cup", "playoff", "tournament", "soccer",
                             "football", "basketball", "baseball", "hockey", "tennis", "golf", "ufc", "boxing"]):
        return "SPORTS"
    if any(w in q for w in ["fed", "federal reserve", "inflation", "cpi", "gdp", "interest rate",
                             "unemployment", "recession", "economy", "treasury", "debt ceiling",
                             "tariff", "trade war", "s&p", "nasdaq", "dow", "stock market"]):
        return "MACRO"
    if any(w in q for w in ["ai", "openai", "chatgpt", "gpt", "apple", "google", "microsoft", "meta",
                             "tesla", "amazon", "nvidia", "technology", "tech", "iphone", "android"]):
        return "TECHNOLOGY"
    if any(w in q for w in ["fda", "drug", "vaccine", "clinical", "cancer", "science", "nasa",
                             "space", "climate", "research", "study", "medical"]):
        return "SCIENCE"
    # Default: geopolitical/political questions
    return "POLITICS"


import time as _time
import random as _random


def _get_with_backoff(url, params=None, max_retries=3, timeout=30):
    """GET with exponential backoff on 429 / transient errors."""
    import requests
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = delay + _random.uniform(0, delay * 0.5)
                logger.warning("Rate limited. Retrying in %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                _time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = delay + _random.uniform(0, delay * 0.5)
            logger.warning("Request failed: %s. Retrying in %.1fs", exc, wait)
            _time.sleep(wait)
            delay *= 2
    raise RuntimeError(f"Max retries exceeded for {url}")


def scan_markets() -> List[MarketOpportunity]:
    """
    Fetch active markets from Polymarket Gamma API, apply filters, and return
    up to 20 MarketOpportunity objects sorted by cross-platform divergence.
    """
    opportunities: List[MarketOpportunity] = []

    params = {
        "active": "true",
        "closed": "false",
        "limit": 30,
        "order": "volume24hr",
        "ascending": "false",
    }

    try:
        resp = _get_with_backoff(f"{GAMMA_API_BASE}/markets", params=params)
        markets = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch markets from Gamma API: %s", exc)
        alert_error("ScannerFailure", str(exc)[:200])
        return []

    if isinstance(markets, dict) and "data" in markets:
        markets = markets["data"]

    for m in markets:
        if len(opportunities) >= 20:
            break

        try:
            # Infer category since Gamma API no longer returns it
            question = m.get("question", "")
            category = _infer_category(question)

            if category in Config.SKIP_CATEGORIES:
                continue
            if category not in Config.PREFERRED_CATEGORIES:
                continue

            volume = float(m.get("volume", 0) or 0)
            if volume < Config.MIN_MARKET_VOLUME_USD:
                continue

            end_date = m.get("endDateIso") or m.get("endDate") or ""
            days_to_expiry = _get_days_to_expiry(end_date)
            if not (1 <= days_to_expiry <= 120):
                continue

            condition_id = m.get("conditionId", "")
            if not condition_id:
                logger.debug("Skipping market with missing conditionId: %s", question[:60])
                continue

            # Parse prices from outcomePrices array (Gamma API current format)
            # outcomePrices: ["yes_price", "no_price"] as strings
            raw_prices = m.get("outcomePrices") or []
            if isinstance(raw_prices, str):
                raw_prices = json.loads(raw_prices)
            if len(raw_prices) < 2:
                continue
            yes_price = float(raw_prices[0])
            no_price = float(raw_prices[1])

            if yes_price <= 0 or yes_price >= 1:
                continue

            # Parse token IDs from clobTokenIds (Gamma API current format)
            raw_token_ids = m.get("clobTokenIds") or []
            if isinstance(raw_token_ids, str):
                raw_token_ids = json.loads(raw_token_ids)
            token_ids = list(raw_token_ids)

            # Use spread from API directly if available, otherwise compute
            spread = float(m.get("spread") or abs(1.0 - yes_price - no_price))
            if spread > Config.MAX_SPREAD:
                continue

            opp = MarketOpportunity(
                market_id=str(m.get("id", "")),
                question=question,
                category=category,
                yes_price=yes_price,
                no_price=no_price,
                volume_usd=volume,
                days_to_expiry=round(days_to_expiry, 2),
                spread=round(spread, 4),
                condition_id=condition_id,
                token_ids=token_ids,
            )
            opportunities.append(opp)
            logger.debug("Accepted: %s [%s] vol=$%.0f spread=%.3f", question[:60], category, volume, spread)

        except Exception as exc:
            logger.warning("Skipping malformed market entry: %s", exc)
            continue

    # Enrich with cross-platform prices and compute divergence
    for opp in opportunities:
        prices = get_cross_platform_prices(opp.question)
        if prices:
            opp.cross_platform_prices = prices
            opp.cross_platform_divergence = round(
                max(abs(opp.yes_price - p) for p in prices.values()), 4
            )

    # Sort by divergence descending: highest cross-platform disagreement first
    opportunities.sort(key=lambda o: o.cross_platform_divergence, reverse=True)

    logger.info(
        "Scanner found %d qualifying markets (%d with cross-platform data).",
        len(opportunities),
        sum(1 for o in opportunities if o.cross_platform_prices),
    )
    return opportunities


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    results = scan_markets()
    for r in results[:5]:
        print(f"\n{r.question}")
        print(f"  Category: {r.category}  Volume: ${r.volume_usd:,.0f}  Spread: {r.spread:.3f}")
        print(f"  YES: {r.yes_price:.3f}  NO: {r.no_price:.3f}  Days: {r.days_to_expiry:.1f}")
