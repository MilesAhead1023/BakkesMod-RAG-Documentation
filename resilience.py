"""
Resilience & Reliability
=========================
Circuit breakers, retry strategies, and fallback mechanisms for production-grade RAG.
"""

import time
import random
from typing import Callable, Any, Optional, List
from functools import wraps
from datetime import datetime, timedelta
from threading import Lock
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import logging

from config import get_config
from observability import get_logger

logger = get_logger()


class CircuitBreaker:
    """Circuit breaker pattern for API calls."""
    
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
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
                    raise Exception(f"Circuit breaker OPEN. Service unavailable.")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            if self.state == self.HALF_OPEN:
                logger.logger.info(f"Circuit breaker CLOSED (recovered)")
            self.state = self.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                if self.state != self.OPEN:
                    logger.logger.error(
                        f"Circuit breaker OPEN after {self.failure_count} failures"
                    )
                self.state = self.OPEN


class APICallManager:
    """Manages API calls with retries, circuit breakers, and fallbacks."""
    
    def __init__(self):
        self.config = get_config().production
        
        # Circuit breakers for each provider
        self.circuit_breakers = {
            "openai": CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout
            ),
            "anthropic": CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout
            ),
            "gemini": CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout
            ),
        }
    
    def with_retry_and_circuit_breaker(
        self,
        provider: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic and circuit breaker."""
        
        @retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=self.config.retry_backoff_factor,
                min=1,
                max=60
            ),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger.logger, logging.WARNING),
            after=after_log(logger.logger, logging.INFO)
        )
        def execute_with_retry():
            # Add jitter if enabled
            if self.config.retry_jitter:
                jitter = random.uniform(0, 0.5)
                time.sleep(jitter)
            
            # Use circuit breaker if enabled
            if self.config.circuit_breaker_enabled and provider in self.circuit_breakers:
                return self.circuit_breakers[provider].call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return execute_with_retry()


class FallbackChain:
    """Implements fallback chain for LLM providers."""
    
    def __init__(self, providers: List[str]):
        """
        Initialize fallback chain.
        
        Args:
            providers: List of provider names in order of preference
                      e.g., ["anthropic", "openai", "gemini"]
        """
        self.providers = providers
        self.api_manager = APICallManager()
    
    def execute(self, provider_funcs: dict) -> Any:
        """
        Execute with fallback chain.
        
        Args:
            provider_funcs: Dict mapping provider name to callable
                           e.g., {"anthropic": lambda: call_anthropic(), ...}
        """
        last_error = None
        
        for provider in self.providers:
            if provider not in provider_funcs:
                continue
            
            try:
                logger.logger.info(f"Attempting provider: {provider}")
                result = self.api_manager.with_retry_and_circuit_breaker(
                    provider,
                    provider_funcs[provider]
                )
                logger.logger.info(f"Success with provider: {provider}")
                return result
                
            except Exception as e:
                logger.log_error(e, {"provider": provider, "fallback": True})
                last_error = e
                continue
        
        # All providers failed
        raise Exception(f"All providers in fallback chain failed. Last error: {last_error}")


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = Lock()
    
    def acquire(self):
        """Acquire a token, blocking if necessary."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + (elapsed * self.requests_per_minute / 60.0)
            )
            self.last_update = now
            
            # Check if we have tokens
            if self.tokens < 1:
                # Calculate wait time
                wait_time = (1 - self.tokens) * (60.0 / self.requests_per_minute)
                time.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class RateLimitedCaller:
    """Wrapper for rate-limited API calls."""
    
    def __init__(self):
        self.config = get_config().production
        
        if self.config.rate_limit_enabled:
            self.limiter = RateLimiter(self.config.requests_per_minute)
        else:
            self.limiter = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with rate limiting."""
        if self.limiter:
            self.limiter.acquire()
        
        return func(*args, **kwargs)


# Decorator for resilient API calls
def resilient_api_call(provider: str):
    """Decorator for resilient API calls with retry, circuit breaker, and rate limiting."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            api_manager = APICallManager()
            rate_limiter = RateLimitedCaller()
            
            def rate_limited_func():
                return rate_limiter.call(func, *args, **kwargs)
            
            return api_manager.with_retry_and_circuit_breaker(
                provider,
                rate_limited_func
            )
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test circuit breaker
    print("Testing Circuit Breaker...")
    
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def failing_func():
        raise Exception("Simulated failure")
    
    # Trigger failures
    for i in range(5):
        try:
            cb.call(failing_func)
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
    
    print("\n✅ Resilience module test complete!")
