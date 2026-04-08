"""
Cheap pre-screener — filters markets BEFORE expensive web-search research.

Uses a single Haiku call with NO web search (~300-500 input tokens per market,
batched into one call for up to 20 markets). This costs ~1/20th of a research
call per market.

Markets that fail pre-screening are skipped from research entirely,
saving ~5-10k tokens each.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import anthropic

from config import Config

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


@dataclass
class PreScreenResult:
    market_id: str
    tradeable: bool
    reason: str = ""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    return _client


_SYSTEM = (
    "You are a prediction market analyst doing a quick screen. "
    "For each market, decide if it is TRADEABLE (worth deep research) or not. "
    "A market is tradeable if: (1) the outcome is objectively verifiable, "
    "(2) there is likely public information that could give an edge over the current price, "
    "(3) the question is not too vague or too far in the future to forecast. "
    "IMPORTANT: Markets priced near 0.01-0.10 or 0.90-0.99 are OFTEN tradeable — "
    "extreme-outcome markets (e.g. 'Will X hit $200?') can have edge by betting NO "
    "when the price is mispriced even slightly. Do NOT reject a market just because "
    "the outcome seems unlikely. "
    "A market is NOT tradeable ONLY if: it is essentially random with no information edge, "
    "depends on unknowable private info, or has ambiguous resolution criteria. "
    "Pass at least 50% of markets. When in doubt, pass the market through."
)


def pre_screen_markets(markets) -> Tuple[List[PreScreenResult], Optional[object]]:
    """
    Batch pre-screen markets with a single cheap Haiku call (no web search).
    Returns (list of PreScreenResult, raw API response or None on failure).
    """
    if not markets:
        return [], None

    client = _get_client()

    lines = []
    for i, m in enumerate(markets, 1):
        lines.append(
            f"{i}. [{m.market_id}] {m.question} "
            f"(YES={m.yes_price:.2f}, expiry={m.days_to_expiry:.0f}d, vol=${m.volume_usd:.0f})"
        )

    user_prompt = (
        "Screen these markets. For each, output exactly:\n"
        "MARKET: <question>\n"
        "TRADEABLE: YES or NO\n"
        "REASON: <one sentence>\n\n"
        + "\n".join(lines)
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        market_ids = [m.market_id for m in markets]
        results = parse_pre_screen_response(text, market_ids)

        passed = sum(1 for r in results if r.tradeable)
        logger.info(
            "Pre-screen: %d/%d markets passed (saved ~%d research calls)",
            passed, len(markets), len(markets) - passed,
        )

        return results, response

    except Exception as exc:
        logger.warning("Pre-screen failed (non-fatal): %s — passing all markets through", exc)
        return [PreScreenResult(market_id=m.market_id, tradeable=True, reason="pre-screen unavailable") for m in markets], None


def parse_pre_screen_response(text: str, market_ids: List[str]) -> List[PreScreenResult]:
    """Parse the structured response into PreScreenResult list."""
    results = []
    current_tradeable = None
    current_reason = ""

    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("TRADEABLE:"):
            val = line.split(":", 1)[1].strip().upper()
            current_tradeable = val.startswith("YES")
        elif upper.startswith("REASON:"):
            current_reason = line.split(":", 1)[1].strip()
            idx = len(results)
            mid = market_ids[idx] if idx < len(market_ids) else f"unknown_{idx}"
            results.append(PreScreenResult(
                market_id=mid,
                tradeable=current_tradeable if current_tradeable is not None else True,
                reason=current_reason,
            ))
            current_tradeable = None
            current_reason = ""

    # Fill in any markets that weren't in the response (benefit of the doubt)
    while len(results) < len(market_ids):
        idx = len(results)
        results.append(PreScreenResult(
            market_id=market_ids[idx],
            tradeable=True,
            reason="not in pre-screen response",
        ))

    return results
