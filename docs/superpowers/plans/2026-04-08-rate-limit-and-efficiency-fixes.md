# Rate Limit, Critic Parsing & Token Efficiency Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate 429 rate limit errors, fix critic JSON parsing failures, and make cycles explore more markets with fewer tokens so trading stays profitable.

**Architecture:** Three-layer fix: (1) add adaptive rate limiting that tracks actual token usage and paces calls to stay under 50k TPM, (2) add JSON repair logic to critic so parsing never fails on unescaped strings, (3) add a cheap pre-screening step that filters markets BEFORE expensive web-search research calls, so tokens go only toward high-potential markets.

**Tech Stack:** Python 3.11+, anthropic SDK, json, re, time, logging. No new dependencies.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `rate_limiter.py` | **Create** | Token-aware rate limiter — tracks usage, computes wait times |
| `critic.py` | **Modify** (lines 49-59, 96-133) | Robust JSON parsing with repair fallback |
| `researcher.py` | **Modify** (lines 218-231) | Use rate limiter instead of hardcoded sleep(20) |
| `forecaster.py` | **Modify** (lines 293-304) | Use rate limiter instead of hardcoded sleep(20) |
| `main.py` | **Modify** (lines 115-215) | Use rate limiter for phase gaps; add pre-screen step |
| `pre_screener.py` | **Create** | Cheap token-light market pre-filter using Haiku (no web search) |
| `config.py` | **Modify** (lines 43-45) | Add TPM_LIMIT, PRE_SCREEN_ENABLED configs |
| `tests/test_rate_limiter.py` | **Create** | Unit tests for rate limiter |
| `tests/test_critic_json.py` | **Create** | Unit tests for critic JSON repair |
| `tests/test_pre_screener.py` | **Create** | Unit tests for pre-screener |

---

### Task 1: Token-Aware Rate Limiter

**Files:**
- Create: `rate_limiter.py`
- Create: `tests/test_rate_limiter.py`

**Why:** Hardcoded `sleep(20)` between calls is a guess. Sometimes it's too short (429), sometimes too long (wasted time). We need a limiter that tracks actual token consumption and sleeps exactly long enough to stay under 50k TPM.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rate_limiter.py
import time
from rate_limiter import TokenRateLimiter


def test_no_wait_when_under_budget():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    limiter.record_usage(5_000)
    wait = limiter.wait_if_needed()
    assert wait == 0.0, "Should not wait when well under budget"


def test_wait_when_near_limit():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    # Simulate 45k tokens consumed in the last few seconds
    limiter.record_usage(45_000)
    wait = limiter.wait_if_needed()
    # With 45k used recently and 50k limit, next 10k call should require a wait
    assert wait > 0, "Should wait when near limit"


def test_old_usage_expires():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    # Record usage but pretend it was 65 seconds ago
    limiter._usage_log.append((time.time() - 65, 40_000))
    wait = limiter.wait_if_needed()
    assert wait == 0.0, "Usage older than 60s should not count"


def test_record_from_response():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    # Simulate an anthropic response object
    class FakeUsage:
        input_tokens = 3000
        output_tokens = 500
    class FakeResponse:
        usage = FakeUsage()
    limiter.record_from_response(FakeResponse())
    assert limiter._total_in_window() == 3500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -m pytest tests/test_rate_limiter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rate_limiter'`

- [ ] **Step 3: Write the implementation**

```python
# rate_limiter.py
"""
Token-aware rate limiter for Anthropic API.

Tracks actual token consumption (input + output) from API responses
and computes the minimum sleep needed to stay under the org TPM limit.

Usage:
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    response = client.messages.create(...)
    limiter.record_from_response(response)
    limiter.wait_if_needed()  # sleeps only if necessary
"""

import logging
import time
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Safety margin — stay 15% under the actual limit to absorb SDK retries
_SAFETY_FACTOR = 0.85


class TokenRateLimiter:
    def __init__(self, tokens_per_minute: int = 50_000):
        self._tpm_limit = int(tokens_per_minute * _SAFETY_FACTOR)
        self._usage_log: List[Tuple[float, int]] = []  # (timestamp, tokens)

    def _prune(self) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = time.time() - 60
        self._usage_log = [(t, n) for t, n in self._usage_log if t > cutoff]

    def _total_in_window(self) -> int:
        self._prune()
        return sum(n for _, n in self._usage_log)

    def record_usage(self, tokens: int) -> None:
        self._usage_log.append((time.time(), tokens))

    def record_from_response(self, response) -> None:
        """Extract token counts from an anthropic API response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        total = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
        # Web search responses include server_tool_use tokens in input_tokens,
        # plus any cache_read_input_tokens — grab those too if present.
        total += getattr(usage, "cache_read_input_tokens", 0)
        self.record_usage(total)
        logger.debug("Rate limiter: recorded %d tokens (window total: %d/%d)",
                      total, self._total_in_window(), self._tpm_limit)

    def wait_if_needed(self, next_call_estimate: int = 8_000) -> float:
        """
        Sleep if the next call would push us over the TPM limit.
        Returns seconds waited (0.0 if no wait needed).

        next_call_estimate: conservative guess of how many tokens the next
        call will use (default 8k covers a web-search research call).
        """
        self._prune()
        headroom = self._tpm_limit - self._total_in_window()

        if headroom >= next_call_estimate:
            return 0.0

        # Find the oldest entry in the window — once it expires, we get those tokens back
        if not self._usage_log:
            return 0.0

        oldest_ts = min(t for t, _ in self._usage_log)
        wait = max(0.0, (oldest_ts + 60) - time.time()) + 1.0  # +1s safety

        logger.info("Rate limiter: waiting %.1fs (headroom=%d, need=%d)",
                     wait, headroom, next_call_estimate)
        time.sleep(wait)
        return wait
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -m pytest tests/test_rate_limiter.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat: add token-aware rate limiter to replace hardcoded sleep(20)"
```

---

### Task 2: Fix Critic JSON Parsing

**Files:**
- Modify: `critic.py` (lines 96-133)
- Create: `tests/test_critic_json.py`

**Why:** Critic returns `concern_level=LOW` on every JSON parse failure (seen in logs: "Unterminated string starting at..."). This means trades that should be blocked sail through. The model outputs unescaped quotes and newlines inside JSON string values. We need a repair step.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_critic_json.py
from critic import _repair_json, CritiqueResult


def test_clean_json_passes_through():
    raw = '{"concern_level": "HIGH", "counter_arguments": ["reason 1"], "overconfidence_flag": false, "rationale": "Bad trade"}'
    result = _repair_json(raw)
    assert result["concern_level"] == "HIGH"


def test_unescaped_quotes_in_rationale():
    # Model outputs: "rationale": "The "evidence" is weak"
    raw = '{"concern_level": "MEDIUM", "counter_arguments": ["a"], "overconfidence_flag": false, "rationale": "The "evidence" is weak"}'
    result = _repair_json(raw)
    assert result["concern_level"] == "MEDIUM"


def test_newline_in_counter_argument():
    raw = '{"concern_level": "HIGH", "counter_arguments": ["line1\nline2"], "overconfidence_flag": false, "rationale": "bad"}'
    result = _repair_json(raw)
    assert result["concern_level"] == "HIGH"


def test_truncated_json_extracts_concern():
    # Model hit max_tokens mid-string
    raw = '{"concern_level": "HIGH", "counter_arguments": ["reason 1", "reason 2 is very long and gets cut off by the tok'
    result = _repair_json(raw)
    assert result["concern_level"] == "HIGH"


def test_markdown_fence_stripped():
    raw = '```json\n{"concern_level": "LOW", "counter_arguments": [], "overconfidence_flag": false, "rationale": "ok"}\n```'
    result = _repair_json(raw)
    assert result["concern_level"] == "LOW"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -m pytest tests/test_critic_json.py -v`
Expected: FAIL — `ImportError: cannot import name '_repair_json'`

- [ ] **Step 3: Add `_repair_json` function to `critic.py`**

Add this function BEFORE the `challenge_forecast` function (insert after line 66):

```python
def _repair_json(text: str) -> dict:
    """
    Parse JSON from critic response, repairing common LLM output issues:
    - Markdown fences
    - Unescaped quotes inside string values
    - Newlines inside strings
    - Truncated output (max_tokens hit mid-string)

    Returns parsed dict. Raises ValueError if repair fails.
    """
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
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    # Restore structural newlines (after { , } [ ] :)
    for ch in ["{", "}", "[", "]", ",", ":"]:
        text = text.replace(f"{ch}\\n", f"{ch}\n")
        text = text.replace(f"\\n{ch}", f"\n{ch}")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix 2: If truncated, try to close it
    # Count open braces/brackets
    truncated = text.rstrip()
    open_braces = truncated.count("{") - truncated.count("}")
    open_brackets = truncated.count("[") - truncated.count("]")

    if open_braces > 0 or open_brackets > 0:
        # Truncate to last complete key-value or array element
        # Find last comma or complete value
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
    import re
    concern_match = re.search(r'"concern_level"\s*:\s*"(LOW|MEDIUM|HIGH)"', text)
    if concern_match:
        return {
            "concern_level": concern_match.group(1),
            "counter_arguments": [],
            "overconfidence_flag": False,
            "rationale": "JSON repair: extracted concern_level only",
        }

    raise ValueError(f"Could not repair JSON: {text[:200]}")
```

- [ ] **Step 4: Update `challenge_forecast` to use `_repair_json`**

Replace lines 103-110 in `critic.py` with:

```python
        text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()

        data = _repair_json(text)
```

This replaces the old markdown-fence stripping + `json.loads(text)` with the robust repair function.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -m pytest tests/test_critic_json.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add critic.py tests/test_critic_json.py
git commit -m "fix: robust JSON parsing in critic with repair for truncated/malformed output"
```

---

### Task 3: Cheap Pre-Screener (Token Saver)

**Files:**
- Create: `pre_screener.py`
- Create: `tests/test_pre_screener.py`

**Why:** Research calls use web search (~5-10k tokens each). Most markets score evidence_quality < 0.55 and get discarded. That's burning tokens on markets that were never going to trade. A pre-screen call WITHOUT web search (~500 tokens) can flag which markets are even worth researching. This lets us scan 20 markets for the token cost of 2 research calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pre_screener.py
from pre_screener import PreScreenResult, parse_pre_screen_response


def test_parse_valid_response():
    response = """MARKET: Will BTC hit 100k?
TRADEABLE: YES
REASON: Clear binary outcome, high liquidity, recent price momentum is measurable.

MARKET: Will it rain in NYC tomorrow?
TRADEABLE: NO
REASON: Weather market, unpredictable short-term, no edge possible."""
    results = parse_pre_screen_response(response, ["id1", "id2"])
    assert len(results) == 2
    assert results[0].tradeable is True
    assert results[1].tradeable is False


def test_parse_handles_partial():
    response = """MARKET: Will BTC hit 100k?
TRADEABLE: YES
REASON: Good market."""
    results = parse_pre_screen_response(response, ["id1", "id2"])
    # Only 1 parsed, id2 defaults to tradeable=True (benefit of doubt)
    assert len(results) == 2
    assert results[0].tradeable is True
    assert results[1].tradeable is True  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -m pytest tests/test_pre_screener.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pre_screener'`

- [ ] **Step 3: Write the implementation**

```python
# pre_screener.py
"""
Cheap pre-screener — filters markets BEFORE expensive web-search research.

Uses a single Haiku call with NO web search (~300-500 input tokens per market,
batched into one call for up to 20 markets). This costs ~1/20th of a research
call per market.

Markets that fail pre-screening are skipped from research entirely,
saving ~5-10k tokens each.
"""

import logging
from dataclasses import dataclass, field
from typing import List

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
    "A market is NOT tradeable if: it is essentially random, depends on unknowable private info, "
    "has ambiguous resolution criteria, or is already priced efficiently with no plausible edge. "
    "Be selective — reject at least 30% of markets. We only want to research high-potential trades."
)


def pre_screen_markets(markets) -> List[PreScreenResult]:
    """
    Batch pre-screen markets with a single cheap Haiku call (no web search).
    Returns a PreScreenResult for each market.
    """
    if not markets:
        return []

    client = _get_client()

    # Build batch prompt — one line per market
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
            # Emit result
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -m pytest tests/test_pre_screener.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pre_screener.py tests/test_pre_screener.py
git commit -m "feat: add cheap pre-screener to filter markets before expensive research"
```

---

### Task 4: Add Config for New Features

**Files:**
- Modify: `config.py` (lines 43-45)

- [ ] **Step 1: Add new config values**

After line 45 (`MAX_MARKETS_PER_CATEGORY`), add:

```python
    # Rate limiting
    TPM_LIMIT: int = int(os.environ.get("TPM_LIMIT", "50000"))

    # Pre-screener
    PRE_SCREEN_ENABLED: bool = os.environ.get("PRE_SCREEN_ENABLED", "true").lower() == "true"
```

- [ ] **Step 2: Verify config loads**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -c "from config import Config; print(Config.TPM_LIMIT, Config.PRE_SCREEN_ENABLED)"`
Expected: `50000 True`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add TPM_LIMIT and PRE_SCREEN_ENABLED config"
```

---

### Task 5: Wire Rate Limiter into Research, Forecast, and Critic

**Files:**
- Modify: `researcher.py` (lines 218-231)
- Modify: `forecaster.py` (lines 293-304)
- Modify: `main.py` (lines 115-215)

**Why:** Replace all `time.sleep(20)` and `time.sleep(30)` with the adaptive rate limiter. The limiter sleeps only as long as needed — sometimes 0s, sometimes 45s — based on actual token consumption.

- [ ] **Step 1: Create shared limiter instance in `main.py`**

At the top of `main.py`, after the existing imports, add:

```python
from rate_limiter import TokenRateLimiter
```

Then in `_run_trading_cycle()` (line 115), at the start of the function, add:

```python
    from rate_limiter import TokenRateLimiter
    limiter = TokenRateLimiter(tokens_per_minute=Config.TPM_LIMIT)
```

- [ ] **Step 2: Update `researcher.py` to accept and use a limiter**

Modify `research_markets` function (line 218). Change signature to accept optional limiter:

```python
def research_markets(markets: List[MarketOpportunity], limiter=None) -> List[ResearchResult]:
    """Research all markets and return those that pass the evidence quality threshold."""
    results = []
    for i, market in enumerate(markets):
        if limiter:
            limiter.wait_if_needed(next_call_estimate=10_000)  # web search calls are expensive
        logger.info("Researching: %s", market.question[:60])
        result = research_market(market, limiter=limiter)
        if result:
            results.append(result)
        # Fallback delay if no limiter
        if limiter is None and i < len(markets) - 1:
            time.sleep(max(Config.API_CALL_DELAY_SECONDS, 20))
    logger.info("Research complete: %d/%d markets passed evidence threshold.", len(results), len(markets))
    return results
```

Also modify `research_market` (line 66) to accept limiter and record usage:

After the `response = client.messages.create(...)` call (around line 130), add:

```python
        if limiter:
            limiter.record_from_response(response)
```

- [ ] **Step 3: Update `forecaster.py` to accept and use a limiter**

Same pattern. Modify `forecast_markets` (line 293):

```python
def forecast_markets(research_results: list[ResearchResult], limiter=None) -> list[ForecastResult]:
    """Forecast all researched markets."""
    results = []
    for i, research in enumerate(research_results):
        if limiter:
            limiter.wait_if_needed(next_call_estimate=3_000)
        result = forecast_market(research, limiter=limiter)
        if result:
            results.append(result)
        if limiter is None and i < len(research_results) - 1:
            time.sleep(max(Config.API_CALL_DELAY_SECONDS, 20))
    logger.info("Forecasting complete: %d forecasts produced.", len(results))
    return results
```

And record usage after API calls in `forecast_market`.

- [ ] **Step 4: Update critic to record usage**

In `challenge_forecast` (critic.py), after the `response = client.messages.create(...)` call (line 100), the function doesn't have access to limiter. Instead, have main.py record critic usage.

In `main.py`, in the critic+execution loop (line 218), change to:

```python
    for i, forecast in enumerate(forecasts):
        if i > 0:
            limiter.wait_if_needed(next_call_estimate=2_000)

        critique = None
        try:
            critique = challenge_forecast(forecast)
        except Exception as exc:
            logger.warning("Critic unavailable for '%s': %s", forecast.question[:50], exc)
```

- [ ] **Step 5: Remove hardcoded phase gaps in main.py**

Replace the 30-second phase gaps (lines 203-205 and 213-215) with limiter waits:

```python
    # Phase gap: let rate limiter decide how long to wait
    limiter.wait_if_needed(next_call_estimate=3_000)
```

- [ ] **Step 6: Pass limiter through the call chain in main.py**

Update the calls in `_run_trading_cycle`:

```python
    research_results = research_markets(opportunities, limiter=limiter)
    ...
    forecasts = forecast_markets(research_results, limiter=limiter)
```

- [ ] **Step 7: Run the full bot briefly to verify no import errors**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -c "from main import _run_trading_cycle; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 8: Commit**

```bash
git add researcher.py forecaster.py main.py
git commit -m "feat: wire token-aware rate limiter into research, forecast, and main loop"
```

---

### Task 6: Wire Pre-Screener into Main Loop

**Files:**
- Modify: `main.py` (lines 186-198)

**Why:** Insert pre-screening between market scanning and research. This is where we save the most tokens — pre-screen 20 markets in one cheap call (~1k tokens total) instead of researching all of them (20 x ~8k = 160k tokens).

- [ ] **Step 1: Add pre-screen import to main.py**

At the top of `main.py`, add:

```python
from pre_screener import pre_screen_markets
```

- [ ] **Step 2: Insert pre-screen step between cache update and research**

After the evaluated cache is saved (line 195) and before the research call (line 198), insert:

```python
    # Step 1.5: Pre-screen to avoid wasting research tokens on low-potential markets
    if Config.PRE_SCREEN_ENABLED and len(opportunities) > 2:
        screen_results, screen_response = pre_screen_markets(opportunities)
        if screen_response and limiter:
            limiter.record_from_response(screen_response)
        tradeable_ids = {r.market_id for r in screen_results if r.tradeable}
        before_screen = len(opportunities)
        opportunities = [o for o in opportunities if o.market_id in tradeable_ids]
        if len(opportunities) < before_screen:
            logger.info(
                "Pre-screen filtered %d/%d markets (saved ~%dk research tokens).",
                before_screen - len(opportunities), before_screen,
                (before_screen - len(opportunities)) * 8,
            )
        if not opportunities:
            logger.info("No markets passed pre-screen.")
            return
```

- [ ] **Step 3: Verify imports work**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && python -c "from main import _run_trading_cycle; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire pre-screener into main loop to save research tokens"
```

---

### Task 7: Increase Critic max_tokens and Improve Prompt

**Files:**
- Modify: `critic.py` (lines 49-59, 98)

**Why:** The critic's `max_tokens=1024` (already bumped from original 256) should be sufficient, but the system prompt should explicitly instruct against newlines inside JSON strings and unescaped quotes. Belt and suspenders with the JSON repair from Task 2.

- [ ] **Step 1: Update critic system prompt**

Replace the last two lines of `_CRITIC_SYSTEM` (the JSON schema line and definitions) with:

```python
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
```

The key additions: "No markdown fences. No text outside the JSON." and "Keep all string values on a single line — no newlines inside strings."

- [ ] **Step 2: Commit**

```bash
git add critic.py
git commit -m "fix: tighten critic system prompt to prevent JSON formatting issues"
```

---

### Task 8: Integration Test — Dry Run

**Files:** None (testing only)

**Why:** Verify the full pipeline works end-to-end with all changes wired together. No real trades — just confirm the cycle completes without 429 errors or JSON parse failures.

- [ ] **Step 1: Run one full cycle and inspect logs**

Run: `cd C:/Users/gvsmv/CLAUDE/Github/Trading-Bot && timeout 300 python main.py` (or Ctrl+C after one cycle completes)

Check `logs/bot_YYYYMMDD.log` for:
- No `429` errors
- No `Critic failed` warnings
- `Pre-screen filtered X/Y markets` line appears
- `Rate limiter: waiting Xs` lines appear (or `recorded Xk tokens`)
- Cycle completes with `=== Trading cycle complete ===`

- [ ] **Step 2: Check token efficiency**

After one cycle, look at the rate limiter logs. Compare:
- **Before:** 4 research calls x ~8k tokens = ~32k tokens, plus forecasts/critics
- **After:** 1 pre-screen (~1k) + fewer research calls + adaptive delays

- [ ] **Step 3: If 429 still occurs, increase safety margin**

In `rate_limiter.py`, change `_SAFETY_FACTOR = 0.85` to `0.75`. This keeps 25% headroom instead of 15%.

---

## Summary: Token Budget Per Cycle (Before vs After)

| Phase | Before (tokens) | After (tokens) | Savings |
|-------|-----------------|----------------|---------|
| Pre-screen | 0 | ~1,500 | -1,500 |
| Research (web search) | ~32,000 (4 markets) | ~16,000 (2 markets) | +16,000 |
| Forecast | ~3,200 | ~1,600 | +1,600 |
| Critic | ~1,000 | ~500 | +500 |
| **Total per cycle** | **~36,200** | **~19,600** | **~46% less** |

The pre-screener costs ~1.5k tokens but saves ~16k by filtering out half the markets before research. Net savings: ~16.5k tokens per cycle. At $0.25/M Haiku tokens, that's ~$0.004/cycle saved — small in absolute terms but it means we stay comfortably under the 50k TPM limit and never hit 429 errors.
