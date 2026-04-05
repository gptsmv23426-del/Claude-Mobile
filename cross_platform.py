"""
Cross-platform price discovery.

Queries Kalshi, Manifold Markets, and Metaculus for markets matching a Polymarket
question by keyword overlap. Returns {platform: yes_probability} for any matches.

Used to find cross-platform divergence — a stronger edge signal than comparing
Claude's estimate to the Polymarket price alone, because it surfaces cases where
multiple independent markets disagree on the same outcome.

All platform fetches are best-effort: any failure returns None and is silently
skipped. The calling code never has a hard dependency on this module succeeding.
"""

import logging
import re
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds; short per-request to keep the enrichment pass fast


def _keyword_overlap(question: str, candidate: str) -> float:
    """
    Return the fraction of non-stopword question tokens that appear in the
    candidate title. A score of 0.4 means 40% of the question's meaningful
    words appear in the candidate.
    """
    STOPWORDS = {
        "the", "a", "an", "is", "will", "be", "in", "of", "to", "by", "on",
        "at", "or", "and", "for", "with", "this", "that", "which", "what",
        "when", "how", "who", "was", "were", "has", "have", "had", "do",
        "does", "did", "are", "not", "its", "it",
    }

    def tokenize(s: str) -> set:
        return {
            w for w in re.findall(r"[a-z]+", s.lower())
            if w not in STOPWORDS and len(w) > 2
        }

    q_tok = tokenize(question)
    c_tok = tokenize(candidate)
    if not q_tok:
        return 0.0
    return len(q_tok & c_tok) / len(q_tok)


def _fetch_kalshi(question: str) -> Optional[float]:
    """
    Search Kalshi for a binary market matching the question.
    Kalshi prices are in cents (0–100); this function divides by 100.
    Returns YES probability (0–1) or None if no adequate match found.
    No auth is required for read-only market data.
    """
    try:
        resp = requests.get(
            "https://trading-api.kalshi.com/trade-api/v2/markets",
            params={"status": "open", "limit": 200},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        markets = resp.json().get("markets", [])

        best_score, best_price = 0.0, None
        for m in markets:
            title = m.get("title", "") or m.get("question", "")
            score = _keyword_overlap(question, title)
            if score > best_score:
                # Prefer last_price; fall back to yes_bid mid
                raw = m.get("last_price")
                if raw is None:
                    yes_ask = m.get("yes_ask")
                    yes_bid = m.get("yes_bid")
                    if yes_ask is not None and yes_bid is not None:
                        raw = (yes_ask + yes_bid) / 2.0
                if raw is not None and 0 < float(raw) < 100:
                    best_score = score
                    best_price = float(raw) / 100.0

        if best_score >= 0.40 and best_price is not None:
            logger.debug("Kalshi match (score=%.2f) for '%s': %.3f", best_score, question[:50], best_price)
            return round(max(0.01, min(0.99, best_price)), 4)

    except Exception as exc:
        logger.debug("Kalshi fetch failed: %s", exc)

    return None


def _fetch_manifold(question: str) -> Optional[float]:
    """
    Search Manifold Markets for a BINARY market matching the question.
    The `probability` field is already 0–1.
    Returns YES probability or None if no adequate match found.
    """
    try:
        resp = requests.get(
            "https://api.manifold.markets/v0/search-markets",
            params={"term": question[:120], "filter": "open", "contractType": "BINARY", "limit": 10},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        markets = resp.json()  # list of market objects
        if not isinstance(markets, list):
            return None

        best_score, best_price = 0.0, None
        for m in markets:
            title = m.get("question", "")
            score = _keyword_overlap(question, title)
            prob = m.get("probability")
            if score > best_score and prob is not None:
                best_score = score
                best_price = float(prob)

        if best_score >= 0.40 and best_price is not None:
            logger.debug("Manifold match (score=%.2f) for '%s': %.3f", best_score, question[:50], best_price)
            return round(max(0.01, min(0.99, best_price)), 4)

    except Exception as exc:
        logger.debug("Manifold fetch failed: %s", exc)

    return None


def _fetch_metaculus(question: str) -> Optional[float]:
    """
    Search Metaculus for a matching question.
    community_prediction.full.q2 is the community median (0–1).
    Returns the community median or None if no adequate match found.
    """
    try:
        resp = requests.get(
            "https://www.metaculus.com/api2/questions/",
            params={
                "search": question[:120],
                "status": "open",
                "type": "forecast",
                "limit": 10,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        best_score, best_price = 0.0, None
        for m in results:
            title = m.get("title", "")
            score = _keyword_overlap(question, title)
            if score > best_score:
                cp = m.get("community_prediction") or {}
                q2 = (cp.get("full") or {}).get("q2")
                if q2 is not None:
                    best_score = score
                    best_price = float(q2)

        if best_score >= 0.40 and best_price is not None:
            logger.debug("Metaculus match (score=%.2f) for '%s': %.3f", best_score, question[:50], best_price)
            return round(max(0.01, min(0.99, best_price)), 4)

    except Exception as exc:
        logger.debug("Metaculus fetch failed: %s", exc)

    return None


def get_cross_platform_prices(question: str) -> Dict[str, float]:
    """
    Query Kalshi, Manifold, and Metaculus for a market matching the question.
    Returns {platform_name: yes_probability} for each platform where a match
    was found. Returns an empty dict when no matches are found anywhere.

    All platform failures are silently swallowed — callers must treat this
    data as an optional enrichment signal, not a required data source.
    """
    prices: Dict[str, float] = {}

    kalshi_price = _fetch_kalshi(question)
    if kalshi_price is not None:
        prices["kalshi"] = kalshi_price

    manifold_price = _fetch_manifold(question)
    if manifold_price is not None:
        prices["manifold"] = manifold_price

    metaculus_price = _fetch_metaculus(question)
    if metaculus_price is not None:
        prices["metaculus"] = metaculus_price

    if prices:
        logger.debug("Cross-platform prices for '%s': %s", question[:50], prices)

    return prices
