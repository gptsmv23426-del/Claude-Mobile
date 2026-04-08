import time
from rate_limiter import TokenRateLimiter


def test_no_wait_when_under_budget():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    limiter.record_usage(5_000)
    wait = limiter.wait_if_needed()
    assert wait == 0.0, "Should not wait when well under budget"


def test_wait_when_near_limit():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    limiter.record_usage(45_000)
    wait = limiter.wait_if_needed()
    assert wait > 0, "Should wait when near limit"


def test_old_usage_expires():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)
    limiter._usage_log.append((time.time() - 65, 40_000))
    wait = limiter.wait_if_needed()
    assert wait == 0.0, "Usage older than 60s should not count"


def test_record_from_response():
    limiter = TokenRateLimiter(tokens_per_minute=50_000)

    class FakeUsage:
        input_tokens = 3000
        output_tokens = 500

    class FakeResponse:
        usage = FakeUsage()

    limiter.record_from_response(FakeResponse())
    assert limiter._total_in_window() == 3500
