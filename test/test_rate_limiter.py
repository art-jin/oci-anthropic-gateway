from src.utils.rate_limiter import SlidingWindowRateLimiter


def test_sliding_window_rate_limiter_blocks_after_limit():
    limiter = SlidingWindowRateLimiter(limit=2, window_sec=60)
    assert limiter.allow("k1")[0] is True
    assert limiter.allow("k1")[0] is True
    allowed, retry_after = limiter.allow("k1")
    assert allowed is False
    assert retry_after >= 1
