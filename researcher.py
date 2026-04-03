"""
Researcher — uses claude-haiku-4-5 with web search to gather evidence for each market.
"""

import logging
from typing import List

import anthropic
from pydantic import BaseModel

from config import Config
from market_scanner import MarketOpportunity

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    return _client


class ResearchResult(BaseModel):
    market_id: str
    question: str
    evidence_summary: str
    evidence_quality: float  # 0.0 – 1.0
    key_facts: List[str]
    category: str
    yes_price: float
    no_price: float
    volume_usd: float
    days_to_expiry: float
    spread: float
    condition_id: str = ""
    token_ids: List[str] = []


def _build_search_query(market: MarketOpportunity) -> str:
    category_hints = {
        "CRYPTO": "cryptocurrency blockchain price prediction",
        "MACRO": "economic policy Federal Reserve inflation GDP",
        "TECHNOLOGY": "technology company product launch AI",
        "SCIENCE": "scientific research study results",
        "POLITICS": "election vote political outcome",
    }
    hint = category_hints.get(market.category, "")
    return f"{market.question} {hint} latest news 2025 2026"


def research_market(market: MarketOpportunity) -> ResearchResult | None:
    """
    Research a single market using Claude haiku with web search.
    Returns None if evidence quality is below threshold.
    """
    client = _get_client()
    query = _build_search_query(market)

    system_prompt = (
        "You are a research analyst for a prediction market trading firm. "
        "Your job is to gather factual evidence relevant to a binary market question "
        "and rate the quality of that evidence. "
        "Be objective. Focus on credible sources, recency, and direct relevance. "
        "Do NOT make a trading recommendation — only report facts."
    )

    user_prompt = f"""Market question: {market.question}
Category: {market.category}
Current YES price: {market.yes_price:.3f}
Days to expiry: {market.days_to_expiry:.1f}

Please search the web for the latest relevant information and then provide:
1. A concise evidence summary (3-5 sentences)
2. An evidence quality score from 0.0 to 1.0 based on:
   - Source credibility (official sources, reputable outlets = higher score)
   - Recency (last 30 days = highest, older = lower)
   - Direct relevance to the market question
3. Up to 5 key facts as bullet points

Respond in this exact format:
EVIDENCE_QUALITY: <float 0.0-1.0>
SUMMARY: <3-5 sentence summary>
KEY_FACTS:
- <fact 1>
- <fact 2>
- <fact 3>"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from response
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        if not text.strip():
            logger.warning("Empty response for market: %s", market.question[:60])
            return None

        # Parse response
        evidence_quality = 0.0
        summary = ""
        key_facts: List[str] = []

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("EVIDENCE_QUALITY:"):
                try:
                    evidence_quality = float(line.split(":", 1)[1].strip())
                    evidence_quality = max(0.0, min(1.0, evidence_quality))
                except ValueError:
                    pass
            elif line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
            elif line.startswith("- ") and summary:
                key_facts.append(line[2:].strip())

        if not summary:
            # fallback: use whole text as summary
            summary = text[:500]

        if evidence_quality < Config.MIN_EVIDENCE_QUALITY:
            logger.info(
                "Skipping market (low evidence quality %.2f): %s",
                evidence_quality,
                market.question[:60],
            )
            return None

        return ResearchResult(
            market_id=market.market_id,
            question=market.question,
            evidence_summary=summary,
            evidence_quality=evidence_quality,
            key_facts=key_facts[:5],
            category=market.category,
            yes_price=market.yes_price,
            no_price=market.no_price,
            volume_usd=market.volume_usd,
            days_to_expiry=market.days_to_expiry,
            spread=market.spread,
            condition_id=market.condition_id,
            token_ids=market.token_ids,
        )

    except Exception as exc:
        logger.error("Research failed for market %s: %s", market.market_id, exc)
        return None


def research_markets(markets: List[MarketOpportunity]) -> List[ResearchResult]:
    """Research all markets and return those that pass the evidence quality threshold."""
    results = []
    for market in markets:
        logger.info("Researching: %s", market.question[:60])
        result = research_market(market)
        if result:
            results.append(result)
    logger.info("Research complete: %d/%d markets passed evidence threshold.", len(results), len(markets))
    return results
