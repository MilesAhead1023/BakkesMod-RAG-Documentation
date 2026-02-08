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
