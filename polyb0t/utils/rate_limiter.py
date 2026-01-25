"""Rate limiter utility for API calls.

Implements token bucket algorithm to prevent exceeding API rate limits.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterStats:
    """Statistics for rate limiter."""
    
    total_requests: int = 0
    throttled_requests: int = 0
    total_wait_time: float = 0.0
    last_request_time: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "throttled_requests": self.throttled_requests,
            "total_wait_time": self.total_wait_time,
            "throttle_rate": self.throttled_requests / self.total_requests if self.total_requests > 0 else 0,
        }


class RateLimiter:
    """Token bucket rate limiter for API calls.
    
    Ensures we don't exceed rate limits by throttling requests.
    """
    
    def __init__(
        self,
        calls_per_second: float = 40.0,
        burst_size: Optional[int] = None,
        name: str = "default",
    ):
        """Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum sustained rate
            burst_size: Maximum burst size (defaults to 2x rate)
            name: Name for logging
        """
        self.rate = calls_per_second
        self.burst_size = burst_size or int(calls_per_second * 2)
        self.name = name
        
        # Token bucket
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        
        # Stats
        self.stats = RateLimiterStats()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now
        
        # Add tokens based on time elapsed
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.rate
        )
    
    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            Time waited in seconds
        """
        async with self._lock:
            self._refill_tokens()
            
            self.stats.total_requests += 1
            self.stats.last_request_time = time.time()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            # Need to wait for tokens
            self.stats.throttled_requests += 1
            
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate
            
            self.stats.total_wait_time += wait_time
            
            logger.debug(
                f"Rate limiter '{self.name}' throttling: "
                f"waiting {wait_time:.3f}s for {tokens_needed:.1f} tokens"
            )
            
            await asyncio.sleep(wait_time)
            
            self._refill_tokens()
            self.tokens -= tokens
            
            return wait_time
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            True if tokens were acquired, False otherwise
        """
        self._refill_tokens()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.stats.total_requests += 1
            return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return self.stats.to_dict()
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self.stats = RateLimiterStats()


class CircuitBreaker:
    """Circuit breaker for API calls.
    
    Prevents cascading failures by temporarily stopping requests
    when error rate gets too high.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        name: str = "default",
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            reset_timeout: Seconds to wait before attempting reset
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._is_open = False
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if not self._is_open:
            return False
        
        # Check if we should try to reset
        if self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info(f"Circuit breaker '{self.name}' attempting reset")
                return False  # Allow a test request
        
        return True
    
    def record_success(self) -> None:
        """Record a successful request."""
        if self._is_open:
            logger.info(f"Circuit breaker '{self.name}' closed after success")
        self._failures = 0
        self._is_open = False
    
    def record_failure(self) -> None:
        """Record a failed request."""
        self._failures += 1
        self._last_failure_time = time.time()
        
        if self._failures >= self.failure_threshold:
            if not self._is_open:
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after "
                    f"{self._failures} failures"
                )
            self._is_open = True
    
    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._failures = 0
        self._is_open = False
        self._last_failure_time = None


# Global rate limiters for different APIs
_gamma_limiter: Optional[RateLimiter] = None
_clob_limiter: Optional[RateLimiter] = None


def get_gamma_limiter() -> RateLimiter:
    """Get rate limiter for Gamma API."""
    global _gamma_limiter
    if _gamma_limiter is None:
        # Gamma: ~1000 requests/hour = ~17/sec, use 15 for safety
        _gamma_limiter = RateLimiter(calls_per_second=15.0, name="gamma")
    return _gamma_limiter


def get_clob_limiter() -> RateLimiter:
    """Get rate limiter for CLOB API."""
    global _clob_limiter
    if _clob_limiter is None:
        # CLOB: ~600 requests/hour = ~10/sec, use 8 for safety
        _clob_limiter = RateLimiter(calls_per_second=8.0, name="clob")
    return _clob_limiter
