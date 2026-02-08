"""
Resilience & Reliability
=========================
Circuit breakers, retry strategies, and fallback mechanisms.
"""

import time
import random
import logging
from typing import Callable, Any, List
from functools import wraps
from datetime import datetime
from threading import Lock

logger = logging.getLogger("bakkesmod_rag.resilience")


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern for API calls."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.CLOSED
        self.lock = Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == self.OPEN:
                if self._should_attempt_reset():
                    self.state = self.HALF_OPEN
                else:
                    raise CircuitBreakerOpen("Circuit breaker OPEN. Service unavailable.")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _on_success(self):
        with self.lock:
            self.failure_count = 0
            if self.state == self.HALF_OPEN:
                logger.info("Circuit breaker CLOSED (recovered)")
            self.state = self.CLOSED

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                if self.state != self.OPEN:
                    logger.error(
                        "Circuit breaker OPEN after %d failures", self.failure_count
                    )
                self.state = self.OPEN


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.last_update = time.time()
        self.lock = Lock()

    def acquire(self):
        """Acquire a token, blocking if necessary."""
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update

                self.tokens = min(
                    self.requests_per_minute,
                    self.tokens + (elapsed * self.requests_per_minute / 60.0),
                )
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                wait_time = (1 - self.tokens) * (60.0 / self.requests_per_minute)

            if wait_time > 0:
                time.sleep(wait_time)


class FallbackChain:
    """Implements fallback chain for LLM providers."""

    def __init__(self, providers: List[str]):
        """Initialize fallback chain.

        Args:
            providers: Provider names in order of preference.
        """
        self.providers = providers

    def execute(self, provider_funcs: dict) -> Any:
        """Execute with fallback chain.

        Args:
            provider_funcs: Dict mapping provider name to callable.
        """
        last_error = None

        for provider in self.providers:
            if provider not in provider_funcs:
                continue

            try:
                logger.info("Attempting provider: %s", provider)
                result = provider_funcs[provider]()
                logger.info("Success with provider: %s", provider)
                return result

            except Exception as e:
                logger.warning("Provider %s failed: %s", provider, e)
                last_error = e
                continue

        raise RuntimeError(
            f"All providers in fallback chain failed. Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# RateLimitedCaller
# ---------------------------------------------------------------------------

class RateLimitedCaller:
    """Combines rate limiting with a callable function.

    Wraps a function so that each invocation first acquires a token from a
    ``RateLimiter`` before proceeding.

    Args:
        func: The callable to protect.
        rate_limiter: A ``RateLimiter`` instance.
    """

    def __init__(self, func: Callable, rate_limiter: RateLimiter):
        self.func = func
        self.rate_limiter = rate_limiter

    def __call__(self, *args, **kwargs) -> Any:
        self.rate_limiter.acquire()
        return self.func(*args, **kwargs)


# ---------------------------------------------------------------------------
# APICallManager
# ---------------------------------------------------------------------------

class APICallManager:
    """Per-provider circuit breakers and rate limiting coordination.

    Creates and manages a ``CircuitBreaker`` and ``RateLimiter`` for each
    registered API provider.  Callers use ``call(provider, func, ...)`` to
    route through the correct breaker/limiter pair automatically.

    Args:
        config: Optional production config with rate-limit and circuit-breaker
            settings. Uses sensible defaults when *None*.
    """

    def __init__(self, config=None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._limiters: dict[str, RateLimiter] = {}

        # Defaults (overridden by config when present)
        self._failure_threshold = 5
        self._recovery_timeout = 60
        self._requests_per_minute = 60

        if config is not None:
            self._failure_threshold = getattr(config, "failure_threshold", 5)
            self._recovery_timeout = getattr(config, "recovery_timeout", 60)
            self._requests_per_minute = getattr(config, "requests_per_minute", 60)

    def _ensure_provider(self, provider: str) -> None:
        """Lazily create breaker/limiter for a provider on first use."""
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )
            self._limiters[provider] = RateLimiter(self._requests_per_minute)

    def call(self, provider: str, func: Callable, *args, **kwargs) -> Any:
        """Execute *func* with circuit-breaker and rate-limit protection.

        Args:
            provider: Provider name (e.g. ``"openai"``, ``"anthropic"``).
            func: The callable to execute.
            *args, **kwargs: Forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            CircuitBreakerOpen: If the provider's breaker is open.
        """
        self._ensure_provider(provider)
        self._limiters[provider].acquire()
        return self._breakers[provider].call(func, *args, **kwargs)

    def get_status(self) -> dict[str, str]:
        """Return the circuit-breaker state for every registered provider."""
        return {p: b.state for p, b in self._breakers.items()}


# ---------------------------------------------------------------------------
# resilient_api_call decorator
# ---------------------------------------------------------------------------

def resilient_api_call(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    expected_exception: type = Exception,
):
    """Decorator: retry with exponential backoff + optional jitter.

    Uses ``tenacity`` when available for robust retry behaviour.  Falls back
    to a simple sleep-loop implementation otherwise.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for exponential backoff.
        jitter: Whether to add random jitter to the wait time.
        expected_exception: Exception type to catch and retry on.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        # Try tenacity for production-grade retries
        try:
            from tenacity import (
                retry,
                stop_after_attempt,
                wait_exponential_jitter,
                wait_exponential,
                retry_if_exception_type,
            )

            wait_strategy = (
                wait_exponential_jitter(
                    initial=1, max=60, jitter=5
                ) if jitter else wait_exponential(
                    multiplier=backoff_factor, min=1, max=60
                )
            )

            return retry(
                stop=stop_after_attempt(max_retries),
                wait=wait_strategy,
                retry=retry_if_exception_type(expected_exception),
                before_sleep=lambda rs: logger.warning(
                    "Retry attempt %d for %s after %s",
                    rs.attempt_number, func.__name__, rs.outcome.exception(),
                ),
            )(func)

        except ImportError:
            # Fallback: manual exponential backoff
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exc = None
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except expected_exception as e:
                        last_exc = e
                        if attempt < max_retries - 1:
                            wait = backoff_factor ** attempt
                            if jitter:
                                wait += random.uniform(0, wait * 0.5)
                            logger.warning(
                                "Retry %d/%d for %s after %.1fs: %s",
                                attempt + 1, max_retries, func.__name__, wait, e,
                            )
                            time.sleep(wait)
                raise last_exc  # type: ignore[misc]

            return wrapper

    return decorator
