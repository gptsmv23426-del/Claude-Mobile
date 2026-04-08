"""
Devil's Advocate Critic — a second Haiku call that argues against every trade
before it reaches the risk manager.

Design principles:
  - Uses claude-haiku-4-5 (~$0.001/call, ~$0.02/cycle for 20 markets). Cheap enough
    to run on every trade candidate.
  - Never blocks on its own failure: any exception returns concern_level=LOW and logs
    a warning. The critic cannot veto trades by failing.
  - max_tokens=256 keeps output tight: concern level + 2-5 counter-arguments.
  - The critic cannot approve trades, only flag or block them.

Integration point (main.py, between forecast and risk check):
  critique = challenge_forecast(forecast)
  if critique.concern_level == "HIGH":
      alert_trade_blocked(...)  # skip to next
  decision = evaluate_trade(forecast, critique=critique)  # MEDIUM halves Kelly
"""

import json
import logging
from typing import List, Literal

import anthropic
from pydantic import BaseModel

from config import Config
from forecaster import ForecastResult

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


class CritiqueResult(BaseModel):
    concern_level: Literal["LOW", "MEDIUM", "HIGH"]
    counter_arguments: List[str]  # 2-5 specific reasons the trade could be wrong
    overconfidence_flag: bool     # True if forecaster seems overconfident
    rationale: str                # 1-sentence summary of the critic's position


_FALLBACK = CritiqueResult(
    concern_level="LOW",
    counter_arguments=[],
    overconfidence_flag=False,
    rationale="Critic unavailable — defaulting to LOW concern.",
)

_CRITIC_SYSTEM = """You are a paranoid risk officer reviewing a trade recommendation.
Your job is to find every reason this trade is wrong. Argue the opposite side.
Look for: recency bias in the evidence, overconfident language, base rate neglect,
missing context, resolution ambiguity, market manipulation, or illiquid spreads.
Be terse and specific. Do not validate the trade — your role is adversarial.
Output ONLY a JSON object. No preamble. No markdown fences. No text outside the JSON.
Keep all string values on a single line — no newlines inside strings.
JSON schema: {"concern_level": "LOW|MEDIUM|HIGH", "counter_arguments": ["...", "..."], "overconfidence_flag": true|false, "rationale": "..."}
Definitions:
  LOW    — trade has minor issues but is defensible
  MEDIUM — trade has meaningful weaknesses; position size should be reduced
  HIGH   — trade should be vetoed; fundamental flaw or unacceptable risk"""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    return _client


def _repair_json(text: str) -> dict:
    """
    Parse JSON from critic response, repairing common LLM output issues:
    - Markdown fences
    - Unescaped quotes inside string values
    - Newlines inside strings
    - Truncated output (max_tokens hit mid-string)

    Returns parsed dict. Raises ValueError if repair fails.
    """
    import re

    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Try clean parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix 1: Escape literal newlines inside strings
    text_fixed = text.replace("\n", "\\n").replace("\r", "\\r")
    # Restore structural newlines (after { , } [ ] :)
    for ch in ["{", "}", "[", "]", ",", ":"]:
        text_fixed = text_fixed.replace(f"{ch}\\n", f"{ch}\n")
        text_fixed = text_fixed.replace(f"\\n{ch}", f"\n{ch}")

    try:
        return json.loads(text_fixed)
    except json.JSONDecodeError:
        pass

    # Fix 2: If truncated, try to close it
    truncated = text.rstrip()
    open_braces = truncated.count("{") - truncated.count("}")
    open_brackets = truncated.count("[") - truncated.count("]")

    if open_braces > 0 or open_brackets > 0:
        for end_char in [",", "}"]:
            idx = truncated.rfind(end_char)
            if idx > 0:
                attempt = truncated[:idx]
                attempt += "]" * open_brackets + "}" * open_braces
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue

    # Fix 3: Regex extraction — just get concern_level (minimum useful data)
    concern_match = re.search(r'"concern_level"\s*:\s*"(LOW|MEDIUM|HIGH)"', text)
    if concern_match:
        return {
            "concern_level": concern_match.group(1),
            "counter_arguments": [],
            "overconfidence_flag": False,
            "rationale": "JSON repair: extracted concern_level only",
        }

    raise ValueError(f"Could not repair JSON: {text[:200]}")


def challenge_forecast(forecast: ForecastResult) -> CritiqueResult:
    """
    Run the devil's advocate critique on a forecast.
    Returns CritiqueResult. Never raises — falls back to LOW concern on any failure.
    """
    try:
        client = _get_client()

        cross_platform_note = ""
        if forecast.cross_platform_prices:
            lines = [f"  {k}: {v:.3f}" for k, v in forecast.cross_platform_prices.items()]
            cross_platform_note = "\nCross-platform prices:\n" + "\n".join(lines)

        user_prompt = (
            f"Market: {forecast.question}\n"
            f"Category: {forecast.category}\n"
            f"Days to expiry: {forecast.days_to_expiry:.1f}\n"
            f"Forecaster probability: {forecast.probability:.3f} (YES)\n"
            f"Bet side: {forecast.side}\n"
            f"Edge claimed: {forecast.edge:.4f}\n"
            f"Confidence: {forecast.confidence}\n"
            f"Evidence quality: {forecast.evidence_quality:.2f}\n"
            f"Forecaster rationale: {forecast.rationale}"
            f"{cross_platform_note}\n\n"
            "Find every reason this trade is wrong. Output JSON only."
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_CRITIC_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()

        data = _repair_json(text)
        concern = str(data.get("concern_level", "LOW")).upper()
        if concern not in ("LOW", "MEDIUM", "HIGH"):
            concern = "LOW"

        result = CritiqueResult(
            concern_level=concern,
            counter_arguments=data.get("counter_arguments", [])[:5],
            overconfidence_flag=bool(data.get("overconfidence_flag", False)),
            rationale=str(data.get("rationale", ""))[:300],
        )

        logger.info(
            "Critic verdict for '%s': %s — %s",
            forecast.question[:50], result.concern_level, result.rationale,
        )
        return result

    except Exception as exc:
        logger.warning(
            "Critic failed for '%s': %s — defaulting to LOW",
            forecast.question[:50], exc,
        )
        return _FALLBACK
