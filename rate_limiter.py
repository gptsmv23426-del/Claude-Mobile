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
