"""
Forecaster — uses claude-haiku-4-5 to produce calibrated probability estimates.

PLANNED UPGRADES (do not implement until approved):

Phase 3 — Ensemble Forecasting (toggle: ENABLE_ENSEMBLE_FORECAST in .env)
  Replace single Haiku call in forecast_market() with 3 independent calls:

  Frame 1 — Base Rate Anchor:
    system: "Start from the historical base rate for this type of event before
             updating on the specific evidence. Anchor on what usually happens."

  Frame 2 — Bull Case:
    system: "Steelman the strongest case for YES. What probability does the
             evidence support if you assume the most favorable interpretation?"

  Frame 3 — Bear Case:
    system: "Steelman the strongest case for NO. What probability does the
             evidence support if you assume the most skeptical interpretation?"

  Aggregation rules:
    final_probability = mean([p1, p2, p3])
    std_dev = stdev([p1, p2, p3])
    if std_dev > 0.12:
        confidence = "LOW"   # models disagree too much — not tradeable
    elif std_dev < 0.05:
        confidence = "HIGH"  # strong agreement across frames
    else:
        confidence = "MEDIUM"

  Only run ensemble if Config.ENABLE_ENSEMBLE_FORECAST is True.
  Fall back to current single-call logic otherwise.

Phase 4 — Calibration-Adjusted Shrinkage
  After computing probability, apply an additional shrinkage based on the bot's
  recent Brier score. If Brier > 0.20 (poorly calibrated), shrink harder toward 0.5:
    from evaluator import get_calibration_summary
    cal = get_calibration_summary()
    if cal and cal.brier_score > 0.20:
        extra_shrinkage = (cal.brier_score - 0.20) * 2.0  # up to 0.6 extra at worst
        probability = probability * (1 - extra_shrinkage) + 0.5 * extra_shrinkage
  This automatically makes the bot more conservative when its past forecasts have been poor.

Phase 4 — Per-Category Learned Thresholds
  Read Config.LEARNED_THRESHOLDS_PATH on startup (if file exists).
  If it contains per-category MIN_EDGE_THRESHOLD overrides, apply them:
    learned = load_learned_thresholds()
    category_min_edge = learned.get(research.category, {}).get("min_edge", Config.MIN_EDGE_THRESHOLD)
  This allows Sonnet's weekly strategy review to automatically tighten or loosen
  the edge requirement per category based on observed performance.
"""

import json
import logging
import re
from typing import Dict, Literal

import anthropic
from pydantic import BaseModel, Field

from config import Config
from researcher import ResearchResult

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

FORECASTER_SYSTEM_PROMPT = (
    "You are a calibrated probability forecaster for prediction markets. "
    "Given a market question and research evidence, output ONLY a JSON object. "
    "No preamble. No markdown. Raw JSON only. "
    'Format: {"probability": 0.0, "confidence": "LOW|MEDIUM|HIGH", "rationale": "..."} '
    "Be conservative. If uncertain, move probability toward 0.5."
)


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    return _client


class ForecastResult(BaseModel):
    market_id: str
    question: str
    probability: float  # probability of YES
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    rationale: str
    edge: float  # abs(probability - yes_price)
    side: Literal["YES", "NO"]  # which side has the edge
    yes_price: float
    no_price: float
    evidence_quality: float
    volume_usd: float
    days_to_expiry: float
    spread: float
    condition_id: str = ""
    token_ids: list = []
    category: str = ""
    cross_platform_divergence: float = 0.0
    cross_platform_prices: Dict[str, float] = Field(default_factory=dict)


def _apply_calibration_penalty(probability: float, evidence_quality: float) -> float:
    """Widen probability toward 0.5 when evidence quality is below 0.75."""
    if evidence_quality >= 0.75:
        return probability
    # Linear shrinkage toward 0.5 proportional to quality gap
    penalty_factor = (0.75 - evidence_quality) / 0.75
    shrinkage = penalty_factor * 0.5
    return probability * (1 - shrinkage) + 0.5 * shrinkage


def forecast_market(research: ResearchResult) -> ForecastResult | None:
    """
    Produce a probability forecast for a single market.
    Returns None on parse failure.
    """
    client = _get_client()

    # When other platforms have priced the same event, include their prices as
    # additional signal. If your estimate diverges from these, you should have
    # an explicit reason — not just different web search results.
    cross_platform_section = ""
    if research.cross_platform_prices:
        lines = [
            f"  {platform}: {prob:.3f}"
            for platform, prob in research.cross_platform_prices.items()
        ]
        cross_platform_section = (
            "\n\nCross-platform prices for the same event (independent markets):\n"
            + "\n".join(lines)
            + "\nIf your probability diverges from these, explain the discrepancy in your rationale."
        )

    user_prompt = f"""Market question: {research.question}
Category: {research.category}
Current YES price (market implied probability): {research.yes_price:.3f}
Current NO price: {research.no_price:.3f}
Days to expiry: {research.days_to_expiry:.1f}
Evidence quality score: {research.evidence_quality:.2f}

Evidence summary:
{research.evidence_summary}

Key facts:
{chr(10).join(f"- {f}" for f in research.key_facts)}{cross_platform_section}

Based on this evidence, what is the true probability that the YES outcome occurs?
Remember: output ONLY a JSON object, no other text."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=FORECASTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        text = text.strip()

        # Strip markdown fences if present (e.g. ```json ... ``` or ``` ... ```)
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        data = json.loads(text)
        raw_prob = float(data["probability"])
        raw_prob = max(0.01, min(0.99, raw_prob))

        confidence = str(data.get("confidence", "LOW")).upper()
        if confidence not in ("LOW", "MEDIUM", "HIGH"):
            confidence = "LOW"

        rationale = str(data.get("rationale", ""))

        # Apply calibration penalty for low-quality evidence
        probability = _apply_calibration_penalty(raw_prob, research.evidence_quality)

        # Determine edge and side
        yes_edge = probability - research.yes_price
        no_edge = (1 - probability) - research.no_price

        if yes_edge >= no_edge:
            edge = yes_edge
            side = "YES"
        else:
            edge = no_edge
            side = "NO"

        logger.info(
            "Forecast for '%s': prob=%.3f conf=%s edge=%.3f side=%s",
            research.question[:50],
            probability,
            confidence,
            edge,
            side,
        )

        return ForecastResult(
            market_id=research.market_id,
            question=research.question,
            probability=round(probability, 4),
            confidence=confidence,
            rationale=rationale,
            edge=round(edge, 4),
            side=side,
            yes_price=research.yes_price,
            no_price=research.no_price,
            evidence_quality=research.evidence_quality,
            volume_usd=research.volume_usd,
            days_to_expiry=research.days_to_expiry,
            spread=research.spread,
            condition_id=research.condition_id,
            token_ids=research.token_ids,
            category=research.category,
            cross_platform_divergence=research.cross_platform_divergence,
            cross_platform_prices=research.cross_platform_prices,
        )

    except Exception as exc:
        logger.error("Forecast failed for market %s: %s", research.market_id, exc)
        return None


def forecast_markets(research_results: list[ResearchResult]) -> list[ForecastResult]:
    """Forecast all researched markets."""
    results = []
    for research in research_results:
        result = forecast_market(research)
        if result:
            results.append(result)
    logger.info("Forecasting complete: %d forecasts produced.", len(results))
    return results
