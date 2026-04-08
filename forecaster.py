"""
Forecaster — uses claude-haiku-4-5 to produce calibrated probability estimates.

Phase 3 (toggle ENABLE_ENSEMBLE_FORECAST=true in .env):
  Three independent Haiku calls per market (base rate, bull case, bear case).
  Aggregation: final = mean; std_dev > 0.12 -> LOW, < 0.05 -> HIGH, else MEDIUM.

Phase 2A (toggle ENABLE_CROSS_REFERENCE=true in .env):
  After generating probability, apply consensus shrinkage from signals/cross_reference.py.
  If any platform diverges by > 0.15 from consensus, shrink probability 20% toward 0.5.
"""

import json
import logging
import re
import statistics
import time
from typing import Dict, Literal, Optional, Tuple

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

_ENSEMBLE_SYSTEMS = [
    # Frame 1: Base Rate Anchor
    (
        "You are a calibrated probability forecaster for prediction markets. "
        "Start from the historical base rate for this type of event before updating on specific evidence. "
        "Anchor on what usually happens, then adjust for the current evidence. "
        'Output ONLY JSON: {"probability": 0.0, "confidence": "LOW|MEDIUM|HIGH", "rationale": "..."}'
    ),
    # Frame 2: Bull Case
    (
        "You are a calibrated probability forecaster for prediction markets. "
        "Steelman the strongest case for YES. What probability does the evidence support "
        "if you assume the most favorable interpretation? "
        'Output ONLY JSON: {"probability": 0.0, "confidence": "LOW|MEDIUM|HIGH", "rationale": "..."}'
    ),
    # Frame 3: Bear Case
    (
        "You are a calibrated probability forecaster for prediction markets. "
        "Steelman the strongest case for NO. What probability does the evidence support "
        "if you assume the most skeptical interpretation? "
        'Output ONLY JSON: {"probability": 0.0, "confidence": "LOW|MEDIUM|HIGH", "rationale": "..."}'
    ),
]


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    return _client


class ForecastResult(BaseModel):
    market_id: str
    question: str
    probability: float
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    rationale: str
    edge: float
    side: Literal["YES", "NO"]
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
    penalty_factor = (0.75 - evidence_quality) / 0.75
    shrinkage = penalty_factor * 0.5
    return probability * (1 - shrinkage) + 0.5 * shrinkage


def _build_user_prompt(research: ResearchResult) -> str:
    """Build the shared user prompt used by both single-call and ensemble paths."""
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

    return (
        f"Market question: {research.question}\n"
        f"Category: {research.category}\n"
        f"Current YES price (market implied probability): {research.yes_price:.3f}\n"
        f"Current NO price: {research.no_price:.3f}\n"
        f"Days to expiry: {research.days_to_expiry:.1f}\n"
        f"Evidence quality score: {research.evidence_quality:.2f}\n\n"
        f"Evidence summary:\n{research.evidence_summary}\n\n"
        f"Key facts:\n"
        + "\n".join(f"- {fact}" for fact in research.key_facts)
        + cross_platform_section
        + "\n\nBased on this evidence, what is the true probability that the YES outcome occurs?\n"
        "Remember: output ONLY a JSON object, no other text."
    )


def _single_haiku_call(
    system_prompt: str,
    user_prompt: str,
) -> Optional[Tuple[float, str, str]]:
    """One Haiku API call. Returns (probability, confidence, rationale) or None on failure."""
    try:
        client = _get_client()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        data = json.loads(text)
        prob = max(0.01, min(0.99, float(data["probability"])))
        conf = str(data.get("confidence", "LOW")).upper()
        if conf not in ("LOW", "MEDIUM", "HIGH"):
            conf = "LOW"
        rationale = str(data.get("rationale", ""))
        return prob, conf, rationale
    except Exception as exc:
        logger.warning("Haiku call failed: %s", exc)
        return None


def _build_forecast_result(
    research: ResearchResult,
    probability: float,
    confidence: str,
    rationale: str,
) -> ForecastResult:
    """Assemble ForecastResult from a probability, computing edge and side."""
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
        research.question[:50], probability, confidence, edge, side,
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


def _forecast_single(research: ResearchResult, user_prompt: str) -> Optional[ForecastResult]:
    """Single Haiku call path."""
    out = _single_haiku_call(FORECASTER_SYSTEM_PROMPT, user_prompt)
    if out is None:
        logger.error("Forecast failed for market %s", research.market_id)
        return None
    prob, conf, rationale = out
    prob = _apply_calibration_penalty(prob, research.evidence_quality)
    return _build_forecast_result(research, prob, conf, rationale)


def _forecast_ensemble(research: ResearchResult, user_prompt: str) -> Optional[ForecastResult]:
    """
    Three independent Haiku calls with different framing prompts.
    Aggregation: mean probability, stdev determines confidence.
    Falls back to single-call if fewer than 2 valid responses.
    """
    probs = []
    rationales = []

    for system in _ENSEMBLE_SYSTEMS:
        out = _single_haiku_call(system, user_prompt)
        if out is not None:
            prob, _, rationale = out
            probs.append(prob)
            rationales.append(rationale)

    if len(probs) < 2:
        logger.warning(
            "Ensemble: only %d valid calls for '%s' — falling back to single.",
            len(probs), research.question[:50],
        )
        return _forecast_single(research, user_prompt)

    final_prob = statistics.mean(probs)
    std_dev = statistics.stdev(probs) if len(probs) >= 2 else 0.0

    if std_dev > 0.12:
        confidence = "LOW"
    elif std_dev < 0.05:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    final_prob = _apply_calibration_penalty(final_prob, research.evidence_quality)
    combined_rationale = " | ".join(r[:120] for r in rationales if r)[:360]

    logger.info(
        "Ensemble for '%s': probs=%s std=%.3f conf=%s",
        research.question[:50], [round(p, 3) for p in probs], std_dev, confidence,
    )
    return _build_forecast_result(research, final_prob, confidence, combined_rationale)


def _consistency_check(result: ForecastResult) -> ForecastResult | None:
    """
    Verify the forecaster's reasoning direction matches its bet direction.
    Uses a single cheap Haiku call (~200 tokens) to detect contradictions
    like 'argues collapse is unlikely' + 'bets YES on collapse'.
    Returns None if contradiction found (kills the forecast).
    """
    try:
        client = _get_client()
        check_prompt = (
            f"A forecaster analyzed this market: '{result.question}'\n"
            f"Their reasoning: {result.rationale[:500]}\n"
            f"Their bet: {result.side} at probability {result.probability:.3f}\n\n"
            f"Does the reasoning SUPPORT or CONTRADICT the bet direction? "
            f"Answer exactly one word: SUPPORT or CONTRADICT"
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": check_prompt}],
        )
        answer = "".join(b.text for b in response.content if hasattr(b, "text")).strip().upper()
        if "CONTRADICT" in answer:
            logger.warning(
                "Consistency gate: reasoning contradicts bet for '%s' (side=%s, prob=%.3f) — dropping forecast",
                result.question[:50], result.side, result.probability,
            )
            return None
        return result
    except Exception as exc:
        logger.debug("Consistency check failed (non-fatal): %s — passing through", exc)
        return result


def forecast_market(research: ResearchResult) -> ForecastResult | None:
    """
    Produce a probability forecast for a single market.
    Dispatches to ensemble or single-call based on Config.ENABLE_ENSEMBLE_FORECAST.
    Applies cross-reference consensus shrinkage if Config.ENABLE_CROSS_REFERENCE.
    Returns None on failure.
    """
    user_prompt = _build_user_prompt(research)

    if Config.ENABLE_ENSEMBLE_FORECAST:
        result = _forecast_ensemble(research, user_prompt)
    else:
        result = _forecast_single(research, user_prompt)

    if result is None:
        return None

    # Consistency gate: verify reasoning direction matches bet direction
    result = _consistency_check(result)
    if result is None:
        return None

    # Phase 2A: apply consensus shrinkage when platforms disagree
    if Config.ENABLE_CROSS_REFERENCE and research.cross_platform_prices:
        try:
            from signals.cross_reference import cross_reference_market, apply_consensus_shrinkage
            xref = cross_reference_market(result.probability, research.cross_platform_prices)
            new_prob = apply_consensus_shrinkage(result.probability, xref)
            if new_prob != result.probability:
                logger.info(
                    "Cross-reference shrinkage: %.3f -> %.3f for '%s' (max_divergence=%.3f)",
                    result.probability, new_prob, research.question[:50], xref.max_divergence,
                )
                result = _build_forecast_result(research, new_prob, result.confidence, result.rationale)
        except Exception as exc:
            logger.warning("Cross-reference shrinkage failed (non-fatal): %s", exc)

    return result


def forecast_markets(research_results: list[ResearchResult], limiter=None) -> list[ForecastResult]:
    """Forecast all researched markets."""
    results = []
    for i, research in enumerate(research_results):
        if limiter:
            limiter.wait_if_needed(next_call_estimate=3_000)
        result = forecast_market(research)
        if result:
            results.append(result)
        # Fallback delay if no limiter
        if limiter is None and i < len(research_results) - 1:
            time.sleep(max(Config.API_CALL_DELAY_SECONDS, 20))
    logger.info("Forecasting complete: %d forecasts produced.", len(results))
    return results
