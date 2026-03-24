import time
import pytest
from coreml_converter.core.registry.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_allows_burst_up_to_capacity(self):
        limiter = TokenBucketRateLimiter(rate=2.0, capacity=3)
        for _ in range(3):
            assert limiter.try_acquire() is True

    def test_blocks_after_burst(self):
        limiter = TokenBucketRateLimiter(rate=2.0, capacity=2)
        limiter.try_acquire()
        limiter.try_acquire()
        assert limiter.try_acquire() is False

    def test_refills_over_time(self):
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=1)
        limiter.try_acquire()
        assert limiter.try_acquire() is False
        time.sleep(0.15)
        assert limiter.try_acquire() is True

    def test_wait_for_token(self):
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=1)
        limiter.try_acquire()
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05
        assert elapsed < 0.5
