"""Tests for CircuitBreaker, RateLimiter, APICallManager, and resilient_api_call."""

import time
import pytest
from bakkesmod_rag.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    RateLimiter,
    FallbackChain,
    APICallManager,
    RateLimitedCaller,
    resilient_api_call,
)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitBreaker.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitBreaker.OPEN

    def test_open_breaker_raises_immediately(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitBreaker.OPEN
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "ok")

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        # Fail twice
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.failure_count == 2

        # Success resets
        cb.call(lambda: "ok")
        assert cb.failure_count == 0
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_on_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitBreaker.OPEN

        # Recovery timeout is 0, so it should transition to HALF_OPEN
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitBreaker.CLOSED

    def test_successful_call_returns_value(self):
        cb = CircuitBreaker()
        result = cb.call(lambda: 42)
        assert result == 42


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_acquire_succeeds(self):
        rl = RateLimiter(requests_per_minute=60)
        # Should not block for the first request
        start = time.time()
        rl.acquire()
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_multiple_acquires(self):
        rl = RateLimiter(requests_per_minute=120)
        for _ in range(5):
            rl.acquire()
        # All should succeed quickly


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_uses_first_successful_provider(self):
        chain = FallbackChain(["a", "b", "c"])
        result = chain.execute({
            "a": lambda: "from_a",
            "b": lambda: "from_b",
        })
        assert result == "from_a"

    def test_falls_through_on_failure(self):
        chain = FallbackChain(["a", "b"])
        call_count = {"b": 0}

        def fail_a():
            raise RuntimeError("a failed")

        def succeed_b():
            call_count["b"] += 1
            return "from_b"

        result = chain.execute({"a": fail_a, "b": succeed_b})
        assert result == "from_b"
        assert call_count["b"] == 1

    def test_all_fail_raises(self):
        chain = FallbackChain(["a", "b"])
        with pytest.raises(RuntimeError, match="All providers"):
            chain.execute({
                "a": lambda: (_ for _ in ()).throw(RuntimeError("a")),
                "b": lambda: (_ for _ in ()).throw(RuntimeError("b")),
            })


# ---------------------------------------------------------------------------
# RateLimitedCaller
# ---------------------------------------------------------------------------

class TestRateLimitedCaller:
    def test_calls_function(self):
        rl = RateLimiter(requests_per_minute=120)
        caller = RateLimitedCaller(lambda x: x * 2, rl)
        assert caller(5) == 10

    def test_passes_kwargs(self):
        rl = RateLimiter(requests_per_minute=120)
        caller = RateLimitedCaller(lambda x, y=0: x + y, rl)
        assert caller(3, y=4) == 7


# ---------------------------------------------------------------------------
# APICallManager
# ---------------------------------------------------------------------------

class TestAPICallManager:
    def test_successful_call(self):
        mgr = APICallManager()
        result = mgr.call("openai", lambda: "hello")
        assert result == "hello"

    def test_creates_breaker_per_provider(self):
        mgr = APICallManager()
        mgr.call("openai", lambda: "ok")
        mgr.call("anthropic", lambda: "ok")
        status = mgr.get_status()
        assert "openai" in status
        assert "anthropic" in status
        assert status["openai"] == CircuitBreaker.CLOSED

    def test_breaker_opens_on_failures(self):
        mgr = APICallManager()
        mgr._failure_threshold = 2

        for _ in range(2):
            with pytest.raises(ValueError):
                mgr.call("bad_provider", lambda: (_ for _ in ()).throw(ValueError("fail")))

        with pytest.raises(CircuitBreakerOpen):
            mgr.call("bad_provider", lambda: "ok")

    def test_config_overrides_defaults(self):
        class MockConfig:
            failure_threshold = 10
            recovery_timeout = 120
            requests_per_minute = 30

        mgr = APICallManager(config=MockConfig())
        assert mgr._failure_threshold == 10
        assert mgr._recovery_timeout == 120
        assert mgr._requests_per_minute == 30


# ---------------------------------------------------------------------------
# resilient_api_call decorator
# ---------------------------------------------------------------------------

class TestResilientDecorator:
    def test_successful_call_no_retry(self):
        @resilient_api_call(max_retries=3)
        def good_func():
            return 42

        assert good_func() == 42

    def test_retries_on_failure(self):
        call_count = {"n": 0}

        @resilient_api_call(max_retries=3, backoff_factor=0.01, jitter=False)
        def flaky_func():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("transient")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count["n"] == 3

    def test_exhausts_retries(self):
        @resilient_api_call(max_retries=2, backoff_factor=0.01, jitter=False)
        def always_fail():
            raise RuntimeError("permanent")

        with pytest.raises((RuntimeError, Exception)):
            # tenacity wraps in RetryError; manual fallback raises RuntimeError
            always_fail()
