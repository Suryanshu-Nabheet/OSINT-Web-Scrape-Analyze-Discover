#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rate Limiter Module

This module provides rate limiting functionality for web scraping operations,
including domain-specific rate limiting, token bucket implementation, and
adaptive rate limiting based on server responses.
"""

import time
import random
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, List, Tuple, Union, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import math

from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger("webscraper.rate_limiter")


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    
    FIXED = "fixed"              # Fixed rate limit
    TOKEN_BUCKET = "token_bucket"  # Token bucket algorithm
    ADAPTIVE = "adaptive"        # Adaptive rate limiting based on responses
    CONCURRENT = "concurrent"    # Limit concurrent requests


class BackoffStrategy(str, Enum):
    """Backoff strategies for handling rate limit errors."""
    
    NONE = "none"                # No backoff
    LINEAR = "linear"            # Linear backoff
    EXPONENTIAL = "exponential"  # Exponential backoff
    FIBONACCI = "fibonacci"      # Fibonacci backoff


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    
    # General settings
    enabled: bool = Field(True, description="Whether rate limiting is enabled")
    strategy: RateLimitStrategy = Field(RateLimitStrategy.TOKEN_BUCKET, description="Rate limiting strategy")
    
    # Request rate settings
    requests_per_second: float = Field(1.0, description="Maximum requests per second")
    requests_per_minute: Optional[int] = Field(None, description="Maximum requests per minute")
    max_concurrent: int = Field(5, description="Maximum concurrent requests")
    
    # Domain-specific settings
    domain_specific: bool = Field(True, description="Whether to use domain-specific rate limits")
    default_domain_rps: float = Field(0.5, description="Default requests per second for domains")
    domain_limits: Dict[str, float] = Field(default_factory=dict, description="Domain-specific rate limits")
    
    # Token bucket settings
    bucket_capacity: int = Field(10, description="Maximum token bucket capacity")
    bucket_fill_rate: float = Field(1.0, description="Rate at which tokens are added to the bucket")
    
    # Adaptive settings
    min_delay: float = Field(0.1, description="Minimum delay between requests")
    max_delay: float = Field(60.0, description="Maximum delay between requests")
    increase_factor: float = Field(2.0, description="Factor to increase delay on failure")
    decrease_factor: float = Field(0.5, description="Factor to decrease delay on success")
    success_streak_threshold: int = Field(10, description="Number of successful requests before decreasing delay")
    
    # Backoff settings
    backoff_strategy: BackoffStrategy = Field(BackoffStrategy.EXPONENTIAL, description="Backoff strategy for rate limit errors")
    backoff_factor: float = Field(2.0, description="Factor for backoff calculation")
    backoff_max_retries: int = Field(5, description="Maximum number of retries with backoff")
    backoff_base_delay: float = Field(1.0, description="Base delay for backoff calculation")
    
    # Jitter settings
    jitter_enabled: bool = Field(True, description="Whether to add jitter to delays")
    jitter_factor: float = Field(0.25, description="Maximum jitter as a fraction of the delay")
    
    # Retry settings
    retry_on_429: bool = Field(True, description="Whether to retry on 429 Too Many Requests")
    retry_on_5xx: bool = Field(True, description="Whether to retry on 5xx Server Error")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "ignore"


class RateLimiter(ABC):
    """
    Abstract base class for rate limiters.
    
    This class defines the interface that all rate limiters must implement.
    """
    
    @abstractmethod
    def acquire(self, domain: Optional[str] = None) -> float:
        """
        Acquire permission to make a request, blocking until the rate limit allows it.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        pass
    
    @abstractmethod
    async def async_acquire(self, domain: Optional[str] = None) -> float:
        """
        Asynchronously acquire permission to make a request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        pass
    
    @abstractmethod
    def feedback(self, success: bool, status_code: Optional[int] = None, domain: Optional[str] = None) -> None:
        """
        Provide feedback about a request to adjust rate limiting.
        
        Args:
            success: Whether the request was successful
            status_code: HTTP status code of the response
            domain: Optional domain for domain-specific rate limiting
        """
        pass
    
    @abstractmethod
    def get_delay(self, domain: Optional[str] = None) -> float:
        """
        Get the current delay between requests.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds
        """
        pass
    
    @abstractmethod
    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update the rate limiter configuration.
        
        Args:
            config: New configuration
        """
        pass


class FixedRateLimiter(RateLimiter):
    """
    Fixed rate limiter that maintains a constant delay between requests.
    
    This implementation uses a simple time-based approach to ensure that
    requests are made at a fixed rate.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the fixed rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.last_request_time = 0.0
        self.lock = threading.Lock()
    
    def acquire(self, domain: Optional[str] = None) -> float:
        """
        Acquire permission to make a request, blocking until the rate limit allows it.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        with self.lock:
            # Calculate time since last request
            now = time.time()
            elapsed = now - self.last_request_time
            
            # Calculate required delay
            delay = self._get_delay_with_jitter(domain)
            
            # If not enough time has passed, wait
            if elapsed < delay:
                wait_time = delay - elapsed
                time.sleep(wait_time)
                waiting_time = wait_time
            else:
                waiting_time = 0.0
            
            # Update last request time
            self.last_request_time = time.time()
            
            return waiting_time
    
    async def async_acquire(self, domain: Optional[str] = None) -> float:
        """
        Asynchronously acquire permission to make a request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        # Use a separate lock for async operations
        async with asyncio.Lock():
            # Calculate time since last request
            now = time.time()
            elapsed = now - self.last_request_time
            
            # Calculate required delay
            delay = self._get_delay_with_jitter(domain)
            
            # If not enough time has passed, wait
            if elapsed < delay:
                wait_time = delay - elapsed
                await asyncio.sleep(wait_time)
                waiting_time = wait_time
            else:
                waiting_time = 0.0
            
            # Update last request time
            self.last_request_time = time.time()
            
            return waiting_time
    
    def feedback(self, success: bool, status_code: Optional[int] = None, domain: Optional[str] = None) -> None:
        """
        Provide feedback about a request to adjust rate limiting.
        
        For a fixed rate limiter, this doesn't do anything.
        
        Args:
            success: Whether the request was successful
            status_code: HTTP status code of the response
            domain: Optional domain for domain-specific rate limiting
        """
        # Fixed rate limiter doesn't adjust based on feedback
        pass
    
    def get_delay(self, domain: Optional[str] = None) -> float:
        """
        Get the current delay between requests.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds
        """
        # Calculate the base delay
        if self.config.requests_per_minute is not None:
            delay = 60.0 / self.config.requests_per_minute
        else:
            delay = 1.0 / self.config.requests_per_second
        
        # Apply domain-specific rate limiting if enabled
        if self.config.domain_specific and domain:
            domain_rps = self.config.domain_limits.get(domain, self.config.default_domain_rps)
            domain_delay = 1.0 / domain_rps
            delay = max(delay, domain_delay)
        
        return delay
    
    def _get_delay_with_jitter(self, domain: Optional[str] = None) -> float:
        """
        Get the delay with jitter applied.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds with jitter
        """
        delay = self.get_delay(domain)
        
        # Apply jitter if enabled
        if self.config.jitter_enabled:
            jitter = random.uniform(0, self.config.jitter_factor * delay)
            delay += jitter
        
        return delay
    
    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update the rate limiter configuration.
        
        Args:
            config: New configuration
        """
        with self.lock:
            self.config = config


class TokenBucketRateLimiter(RateLimiter):
    """
    Token bucket rate limiter.
    
    This implementation uses the token bucket algorithm to enforce rate limits.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.tokens = config.bucket_capacity
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        
        # Domain-specific token buckets
        self.domain_tokens: Dict[str, float] = {}
        self.domain_last_refill: Dict[str, float] = {}
    
    def acquire(self, domain: Optional[str] = None) -> float:
        """
        Acquire permission to make a request, blocking until the rate limit allows it.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        start_time = time.time()
        waiting_time = 0.0
        
        with self.lock:
            # Refill the token bucket
            self._refill_tokens(domain)
            
            # Get the appropriate token bucket
            if self.config.domain_specific and domain:
                # Initialize domain token bucket if needed
                if domain not in self.domain_tokens:
                    self.domain_tokens[domain] = self.config.bucket_capacity
                    self.domain_last_refill[domain] = time.time()
                
                tokens = self.domain_tokens[domain]
            else:
                tokens = self.tokens
            
            # If no tokens are available, wait
            while tokens < 1.0:
                # Calculate wait time until next token
                if self.config.domain_specific and domain:
                    fill_rate = 1.0 / (1.0 / self.config.domain_limits.get(domain, self.config.default_domain_rps))
                else:
                    fill_rate = self.config.bucket_fill_rate
                
                wait_time = (1.0 - tokens) / fill_rate
                
                # Apply jitter if enabled
                if self.config.jitter_enabled:
                    jitter = random.uniform(0, self.config.jitter_factor * wait_time)
                    wait_time += jitter
                
                # Release the lock while waiting
                self.lock.release()
                time.sleep(wait_time)
                waiting_time += wait_time
                self.lock.acquire()
                
                # Refill tokens after waiting
                self._refill_tokens(domain)
                
                # Get updated token count
                if self.config.domain_specific and domain:
                    tokens = self.domain_tokens[domain]
                else:
                    tokens = self.tokens
            
            # Consume a token
            if self.config.domain_specific and domain:
                self.domain_tokens[domain] -= 1.0
            else:
                self.tokens -= 1.0
        
        return waiting_time
    
    async def async_acquire(self, domain: Optional[str] = None) -> float:
        """
        Asynchronously acquire permission to make a request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        start_time = time.time()
        waiting_time = 0.0
        
        # Use asyncio.Lock() for asynchronous locking
        async with asyncio.Lock():
            # Refill the token bucket
            self._refill_tokens(domain)
            
            # Get the appropriate token bucket
            if self.config.domain_specific and domain:
                # Initialize domain token bucket if needed
                if domain not in self.domain_tokens:
                    self.domain_tokens[domain] = self.config.bucket_capacity
                    self.domain_last_refill[domain] = time.time()
                
                tokens = self.domain_tokens[domain]
            else:
                tokens = self.tokens
            
            # If no tokens are available, wait
            while tokens < 1.0:
                # Calculate wait time until next token
                if self.config.domain_specific and domain:
                    fill_rate = 1.0 / (1.0 / self.config.domain_limits.get(domain, self.config.default_domain_rps))
                else:
                    fill_rate = self.config.bucket_fill_rate
                
                wait_time = (1.0 - tokens) / fill_rate
                
                # Apply jitter if enabled
                if self.config.jitter_enabled:
                    jitter = random.uniform(0, self.config.jitter_factor * wait_time)
                    wait_time += jitter
                
                # Wait asynchronously
                await asyncio.sleep(wait_time)
                waiting_time += wait_time
                
                # Refill tokens after waiting
                self._refill_tokens(domain)
                
                # Get updated token count
                if self.config.domain_specific and domain:
                    tokens = self.domain_tokens[domain]
                else:
                    tokens = self.tokens
            
            # Consume a token
            if self.config.domain_specific and domain:
                self.domain_tokens[domain] -= 1.0
            else:
                self.tokens -= 1.0
        
        return waiting_time
    
    def _refill_tokens(self, domain: Optional[str] = None) -> None:
        """
        Refill tokens in the bucket based on elapsed time.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
        """
        now = time.time()
        
        if self.config.domain_specific and domain:
            # Initialize domain token bucket if needed
            if domain not in self.domain_tokens:
                self.domain_tokens[domain] = self.config.bucket_capacity
                self.domain_last_refill[domain] = now
                return
            
            # Calculate elapsed time since last refill
            elapsed = now - self.domain_last_refill[domain]
            
            # Calculate tokens to add
            domain_rps = self.config.domain_limits.get(domain, self.config.default_domain_rps)
            fill_rate = domain_rps
            tokens_to_add = elapsed * fill_rate
            
            # Add tokens and update last refill time
            self.domain_tokens[domain] = min(
                self.config.bucket_capacity,
                self.domain_tokens[domain] + tokens_to_add
            )
            self.domain_last_refill[domain] = now
        else:
            # Calculate elapsed time since last refill
            elapsed = now - self.last_refill_time
            
            # Calculate tokens to add
            tokens_to_add = elapsed * self.config.bucket_fill_rate
            
            # Add tokens and update last refill time
            self.tokens = min(
                self.config.bucket_capacity,
                self.tokens + tokens_to_add
            )
            self.last_refill_time = now
    
    def feedback(self, success: bool, status_code: Optional[int] = None, domain: Optional[str] = None) -> None:
        """
        Provide feedback about a request to adjust rate limiting.
        
        For a token bucket limiter, this doesn't do anything.
        
        Args:
            success: Whether the request was successful
            status_code: HTTP status code of the response
            domain: Optional domain for domain-specific rate limiting
        """
        # Token bucket doesn't adjust based on feedback
        pass
    
    def get_delay(self, domain: Optional[str] = None) -> float:
        """
        Get the current delay between requests.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds
        """
        with self.lock:
            # Refill tokens
            self._refill_tokens(domain)
            
            # Get current token count
            if self.config.domain_specific and domain:
                if domain not in self.domain_tokens:
                    return 0.0
                tokens = self.domain_tokens[domain]
                domain_rps = self.config.domain_limits.get(domain, self.config.default_domain_rps)
                fill_rate = domain_rps
            else:
                tokens = self.tokens
                fill_rate = self.config.bucket_fill_rate
            
            # If tokens are available, no delay
            if tokens >= 1.0:
                return 0.0
            
            # Calculate delay until next token
            return (1.0 - tokens) / fill_rate
    
    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update the rate limiter configuration.
        
        Args:
            config: New configuration
        """
        with self.lock:
            old_config = self.config
            self.config = config
            
            # If bucket capacity changed, adjust tokens
            if config.bucket_capacity != old_config.bucket_capacity:
                self.tokens = min(self.tokens, config.bucket_capacity)
                
                # Adjust domain tokens
                for domain in self.domain_tokens:
                    self.domain_tokens[domain] = min(
                        self.domain_tokens[domain],
                        config.bucket_capacity
                    )


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on server responses.
    
    This implementation dynamically adjusts the delay between requests
    based on the success or failure of previous requests.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.delay = 1.0 / config.requests_per_second
        self.last_request_time = 0.0
        self.success_streak = 0
        self.lock = threading.Lock()
        
        # Domain-specific state
        self.domain_delay: Dict[str, float] = {}
        self.domain_last_request: Dict[str, float] = {}
        self.domain_success_streak: Dict[str, int] = {}
    
    def acquire(self, domain: Optional[str] = None) -> float:
        """
        Acquire permission to make a request, blocking until the rate limit allows it.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        with self.lock:
            # Calculate time since last request
            now = time.time()
            
            if self.config.domain_specific and domain:
                # Initialize domain state if needed
                if domain not in self.domain_delay:
                    self.domain_delay[domain] = 1.0 / self.config.domain_limits.get(
                        domain, self.config.default_domain_rps
                    )
                    self.domain_last_request[domain] = 0.0
                    self.domain_success_streak[domain] = 0
                
                elapsed = now - self.domain_last_request.get(domain, 0.0)
                delay = self._get_delay_with_jitter(domain)
            else:
                elapsed = now - self.last_request_time
                delay = self._get_delay_with_jitter()
            
            # If not enough time has passed, wait
            if elapsed < delay:
                wait_time = delay - elapsed
                time.sleep(wait_time)
                waiting_time = wait_time
            else:
                waiting_time = 0.0
            
            # Update last request time
            if self.config.domain_specific and domain:
                self.domain_last_request[domain] = time.time()
            else:
                self.last_request_time = time.time()
            
            return waiting_time
    
    async def async_acquire(self, domain: Optional[str] = None) -> float:
        """
        Asynchronously acquire permission to make a request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        # Use asyncio.Lock() for asynchronous locking
        async with asyncio.Lock():
            # Calculate time since last request
            now = time.time()
            
            if self.config.domain_specific and domain:
                # Initialize domain state if needed
                if domain not in self.domain_delay:
                    self.domain_delay[domain] = 1.0 / self.config.domain_limits.get(
                        domain, self.config.default_domain_rps
                    )
                    self.domain_last_request[domain] = 0.0
                    self.domain_success_streak[domain] = 0
                
                elapsed = now - self.domain_last_request.get(domain, 0.0)
                delay = self._get_delay_with_jitter(domain)
            else:
                elapsed = now - self.last_request_time
                delay = self._get_delay_with_jitter()
            
            # If not enough time has passed, wait
            if elapsed < delay:
                wait_time = delay - elapsed
                await asyncio.sleep(wait_time)
                waiting_time = wait_time
            else:
                waiting_time = 0.0
            
            # Update last request time
            if self.config.domain_specific and domain:
                self.domain_last_request[domain] = time.time()
            else:
                self.last_request_time = time.time()
            
            return waiting_time
    
    def feedback(self, success: bool, status_code: Optional[int] = None, domain: Optional[str] = None) -> None:
        """
        Provide feedback about a request to adjust rate limiting.
        
        Args:
            success: Whether the request was successful
            status_code: HTTP status code of the response
            domain: Optional domain for domain-specific rate limiting
        """
        if not self.config.enabled:
            return
        
        with self.lock:
            # Handle rate limit responses
            if status_code == 429:  # Too Many Requests
                self._handle_rate_limit_exceeded(domain)
                return
            
            # Handle server errors
            if status_code and status_code >= 500 and status_code < 600:
                self._handle_server_error(domain)
                return
            
            # Adjust delay based on success or failure
            if success:
                self._handle_success(domain)
            else:
                self._handle_failure(domain)
    
    def _handle_rate_limit_exceeded(self, domain: Optional[str] = None) -> None:
        """
        Handle a rate limit exceeded response.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
        """
        # Increase the delay significantly
        if self.config.domain_specific and domain:
            self.domain_delay[domain] = min(
                self.domain_delay[domain] * self.config.increase_factor * 2,
                self.config.max_delay
            )
            self.domain_success_streak[domain] = 0
            logger.warning("Rate limit exceeded for domain %s, increasing delay to %.2f seconds",
                         domain, self.domain_delay[domain])
        else:
            self.delay = min(
                self.delay * self.config.increase_factor * 2,
                self.config.max_delay
            )
            self.success_streak = 0
            logger.warning("Rate limit exceeded, increasing delay to %.2f seconds", self.delay)
    
    def _handle_server_error(self, domain: Optional[str] = None) -> None:
        """
        Handle a server error response.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
        """
        # Increase the delay moderately
        if self.config.domain_specific and domain:
            self.domain_delay[domain] = min(
                self.domain_delay[domain] * self.config.increase_factor,
                self.config.max_delay
            )
            self.domain_success_streak[domain] = 0
            logger.warning("Server error for domain %s, increasing delay to %.2f seconds",
                         domain, self.domain_delay[domain])
        else:
            self.delay = min(
                self.delay * self.config.increase_factor,
                self.config.max_delay
            )
            self.success_streak = 0
            logger.warning("Server error, increasing delay to %.2f seconds", self.delay)
    
    def _handle_success(self, domain: Optional[str] = None) -> None:
        """
        Handle a successful request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
        """
        if self.config.domain_specific and domain:
            # Increase success streak
            self.domain_success_streak[domain] = self.domain_success_streak.get(domain, 0) + 1
            
            # If we've had enough successful requests, decrease the delay
            if self.domain_success_streak[domain] >= self.config.success_streak_threshold:
                self.domain_delay[domain] = max(
                    self.domain_delay[domain] * self.config.decrease_factor,
                    1.0 / self.config.domain_limits.get(domain, self.config.default_domain_rps),
                    self.config.min_delay
                )
                self.domain_success_streak[domain] = 0
                logger.debug("Success streak reached for domain %s, decreasing delay to %.2f seconds",
                           domain, self.domain_delay[domain])
        else:
            # Increase success streak
            self.success_streak += 1
            
            # If we've had enough successful requests, decrease the delay
            if self.success_streak >= self.config.success_streak_threshold:
                self.delay = max(
                    self.delay * self.config.decrease_factor,
                    1.0 / self.config.requests_per_second,
                    self.config.min_delay
                )
                self.success_streak = 0
                logger.debug("Success streak reached, decreasing delay to %.2f seconds", self.delay)
    
    def _handle_failure(self, domain: Optional[str] = None) -> None:
        """
        Handle a failed request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
        """
        if self.config.domain_specific and domain:
            # Reset success streak
            self.domain_success_streak[domain] = 0
            
            # Increase the delay
            self.domain_delay[domain] = min(
                self.domain_delay[domain] * self.config.increase_factor,
                self.config.max_delay
            )
            logger.debug("Request failed for domain %s, increasing delay to %.2f seconds",
                       domain, self.domain_delay[domain])
        else:
            # Reset success streak
            self.success_streak = 0
            
            # Increase the delay
            self.delay = min(
                self.delay * self.config.increase_factor,
                self.config.max_delay
            )
            logger.debug("Request failed, increasing delay to %.2f seconds", self.delay)
    
    def get_delay(self, domain: Optional[str] = None) -> float:
        """
        Get the current delay between requests.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds
        """
        if self.config.domain_specific and domain:
            return self.domain_delay.get(
                domain,
                1.0 / self.config.domain_limits.get(domain, self.config.default_domain_rps)
            )
        else:
            return self.delay
    
    def _get_delay_with_jitter(self, domain: Optional[str] = None) -> float:
        """
        Get the delay with jitter applied.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds with jitter
        """
        delay = self.get_delay(domain)
        
        # Apply jitter if enabled
        if self.config.jitter_enabled:
            jitter = random.uniform(0, self.config.jitter_factor * delay)
            delay += jitter
        
        return delay
    
    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update the rate limiter configuration.
        
        Args:
            config: New configuration
        """
        with self.lock:
            self.config = config
            
            # Reset delay if new rate is higher than current rate
            new_base_delay = 1.0 / config.requests_per_second
            if new_base_delay < self.delay:
                self.delay = new_base_delay
                
                # Reset domain delays if needed
                for domain in self.domain_delay:
                    domain_rps = config.domain_limits.get(domain, config.default_domain_rps)
                    domain_delay = 1.0 / domain_rps
                    if domain_delay < self.domain_delay[domain]:
                        self.domain_delay[domain] = domain_delay


class ConcurrentRateLimiter(RateLimiter):
    """
    Concurrent rate limiter that limits the number of concurrent requests.
    
    This implementation uses a semaphore to limit the number of concurrent
    requests, ensuring that only a certain number of requests are in flight
    at any given time.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the concurrent rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.semaphore = threading.Semaphore(config.max_concurrent)
        self.async_semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # Domain-specific semaphores
        self.domain_semaphores: Dict[str, threading.Semaphore] = {}
        self.domain_async_semaphores: Dict[str, asyncio.Semaphore] = {}
    
    def acquire(self, domain: Optional[str] = None) -> float:
        """
        Acquire permission to make a request, blocking until a concurrent slot is available.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds (always 0.0 for this implementation)
        """
        if not self.config.enabled:
            return 0.0
        
        start_time = time.time()
        
        # Get the appropriate semaphore
        if self.config.domain_specific and domain:
            # Initialize domain semaphore if needed
            if domain not in self.domain_semaphores:
                # Get domain-specific concurrent limit
                domain_limit = min(
                    self.config.max_concurrent,
                    3  # Default domain limit
                )
                self.domain_semaphores[domain] = threading.Semaphore(domain_limit)
            
            # Acquire domain semaphore
            self.domain_semaphores[domain].acquire()
        else:
            # Acquire global semaphore
            self.semaphore.acquire()
        
        # Return time spent waiting
        return time.time() - start_time
    
    async def async_acquire(self, domain: Optional[str] = None) -> float:
        """
        Asynchronously acquire permission to make a request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds (always 0.0 for this implementation)
        """
        if not self.config.enabled:
            return 0.0
        
        start_time = time.time()
        
        # Get the appropriate semaphore
        if self.config.domain_specific and domain:
            # Initialize domain semaphore if needed
            if domain not in self.domain_async_semaphores:
                # Get domain-specific concurrent limit
                domain_limit = min(
                    self.config.max_concurrent,
                    3  # Default domain limit
                )
                self.domain_async_semaphores[domain] = asyncio.Semaphore(domain_limit)
            
            # Acquire domain semaphore
            await self.domain_async_semaphores[domain].acquire()
        else:
            # Acquire global semaphore
            await self.async_semaphore.acquire()
        
        # Return time spent waiting
        return time.time() - start_time
    
    def release(self, domain: Optional[str] = None) -> None:
        """
        Release a concurrent slot.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
        """
        if not self.config.enabled:
            return
        
        # Release the appropriate semaphore
        if self.config.domain_specific and domain:
            if domain in self.domain_semaphores:
                self.domain_semaphores[domain].release()
        else:
            self.semaphore.release()
    
    def async_release(self, domain: Optional[str] = None) -> None:
        """
        Release a concurrent slot (async version).
        
        Args:
            domain: Optional domain for domain-specific rate limiting
        """
        if not self.config.enabled:
            return
        
        # Release the appropriate semaphore
        if self.config.domain_specific and domain:
            if domain in self.domain_async_semaphores:
                self.domain_async_semaphores[domain].release()
        else:
            self.async_semaphore.release()
    
    def feedback(self, success: bool, status_code: Optional[int] = None, domain: Optional[str] = None) -> None:
        """
        Provide feedback about a request to adjust rate limiting.
        
        For a concurrent limiter, this doesn't do anything.
        
        Args:
            success: Whether the request was successful
            status_code: HTTP status code of the response
            domain: Optional domain for domain-specific rate limiting
        """
        # Concurrent rate limiter doesn't adjust based on feedback
        pass
    
    def get_delay(self, domain: Optional[str] = None) -> float:
        """
        Get the current delay between requests.
        
        For a concurrent limiter, this always returns 0.0 as there
        is no fixed delay between requests.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds (always 0.0 for this implementation)
        """
        return 0.0
    
    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update the rate limiter configuration.
        
        Args:
            config: New configuration
        """
        self.config = config
        
        # Create a new semaphore with the updated concurrent limit
        self.semaphore = threading.Semaphore(config.max_concurrent)
        self.async_semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # Reset domain semaphores
        self.domain_semaphores = {}
        self.domain_async_semaphores = {}


class DomainRateLimiter(RateLimiter):
    """
    Domain-specific rate limiter that maintains separate rate limits for different domains.
    
    This implementation delegates to other rate limiters for each domain.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the domain rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.default_limiter = self._create_limiter(config)
        self.domain_limiters: Dict[str, RateLimiter] = {}
        self.lock = threading.Lock()
    
    def _create_limiter(self, config: RateLimitConfig) -> RateLimiter:
        """
        Create a rate limiter based on the strategy.
        
        Args:
            config: Rate limiting configuration
            
        Returns:
            Rate limiter instance
        """
        strategy = config.strategy
        
        if strategy == RateLimitStrategy.FIXED:
            return FixedRateLimiter(config)
        elif strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        elif strategy == RateLimitStrategy.ADAPTIVE:
            return AdaptiveRateLimiter(config)
        elif strategy == RateLimitStrategy.CONCURRENT:
            return ConcurrentRateLimiter(config)
        else:
            logger.warning("Unknown rate limiting strategy: %s, using token bucket", strategy)
            return TokenBucketRateLimiter(config)
    
    def acquire(self, domain: Optional[str] = None) -> float:
        """
        Acquire permission to make a request, blocking until the rate limit allows it.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        if domain:
            # Get or create domain-specific limiter
            limiter = self._get_domain_limiter(domain)
            return limiter.acquire()
        else:
            # Use default limiter
            return self.default_limiter.acquire()
    
    async def async_acquire(self, domain: Optional[str] = None) -> float:
        """
        Asynchronously acquire permission to make a request.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Time spent waiting in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        if domain:
            # Get or create domain-specific limiter
            limiter = self._get_domain_limiter(domain)
            return await limiter.async_acquire()
        else:
            # Use default limiter
            return await self.default_limiter.async_acquire()
    
    def _get_domain_limiter(self, domain: str) -> RateLimiter:
        """
        Get or create a rate limiter for the specified domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Rate limiter for the domain
        """
        with self.lock:
            if domain not in self.domain_limiters:
                # Create domain-specific configuration
                domain_config = RateLimitConfig.model_validate(self.config.model_dump())
                
                # Set domain-specific rate limit
                if domain in self.config.domain_limits:
                    domain_rps = self.config.domain_limits[domain]
                    domain_config.requests_per_second = domain_rps
                
                # Create limiter
                self.domain_limiters[domain] = self._create_limiter(domain_config)
            
            return self.domain_limiters[domain]
    
    def feedback(self, success: bool, status_code: Optional[int] = None, domain: Optional[str] = None) -> None:
        """
        Provide feedback about a request to adjust rate limiting.
        
        Args:
            success: Whether the request was successful
            status_code: HTTP status code of the response
            domain: Optional domain for domain-specific rate limiting
        """
        if not self.config.enabled:
            return
        
        if domain:
            # Get or create domain-specific limiter
            limiter = self._get_domain_limiter(domain)
            limiter.feedback(success, status_code)
        else:
            # Use default limiter
            self.default_limiter.feedback(success, status_code)
    
    def get_delay(self, domain: Optional[str] = None) -> float:
        """
        Get the current delay between requests.
        
        Args:
            domain: Optional domain for domain-specific rate limiting
            
        Returns:
            Delay in seconds
        """
        if not self.config.enabled:
            return 0.0
        
        if domain:
            # Get or create domain-specific limiter
            limiter = self._get_domain_limiter(domain)
            return limiter.get_delay()
        else:
            # Use default limiter
            return self.default_limiter.get_delay()
    
    def update_config(self, config: RateLimitConfig) -> None:
        """
        Update the rate limiter configuration.
        
        Args:
            config: New configuration
        """
        with self.lock:
            old_config = self.config
            self.config = config
            
            # Update default limiter
            if config.strategy == old_config.strategy:
                # If strategy is the same, update the existing limiter
                self.default_limiter.update_config(config)
            else:
                # If strategy changed, create a new limiter
                self.default_limiter = self._create_limiter(config)
            
            # Clear domain limiters to recreate them with the new configuration
            self.domain_limiters = {}


def create_rate_limiter(config: RateLimitConfig) -> RateLimiter:
    """
    Create a rate limiter based on the configuration.
    
    Args:
        config: Rate limiting configuration
        
    Returns:
        Rate limiter instance
    """
    if config.domain_specific:
        return DomainRateLimiter(config)
    else:
        strategy = config.strategy
        
        if strategy == RateLimitStrategy.FIXED:
            return FixedRateLimiter(config)
        elif strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        elif strategy == RateLimitStrategy.ADAPTIVE:
            return AdaptiveRateLimiter(config)
        elif strategy == RateLimitStrategy.CONCURRENT:
            return ConcurrentRateLimiter(config)
        else:
            logger.warning("Unknown rate limiting strategy: %s, using token bucket", strategy)
            return TokenBucketRateLimiter(config)


class BackoffHandler:
    """
    Handler for implementing backoff strategies for retries.
    
    This class provides methods for calculating delays for retries
    based on different backoff strategies.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize the backoff handler.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
    
    def get_backoff_delay(self, retry_count: int) -> float:
        """
        Get the delay for a retry based on the backoff strategy.
        
        Args:
            retry_count: Current retry count (0-based)
            
        Returns:
            Delay in seconds
        """
        strategy = self.config.backoff_strategy
        base_delay = self.config.backoff_base_delay
        factor = self.config.backoff_factor
        
        if strategy == BackoffStrategy.NONE:
            return 0.0
        elif strategy == BackoffStrategy.LINEAR:
            return self._linear_backoff(retry_count, base_delay)
        elif strategy == BackoffStrategy.EXPONENTIAL:
            return self._exponential_backoff(retry_count, base_delay, factor)
        elif strategy == BackoffStrategy.FIBONACCI:
            return self._fibonacci_backoff(retry_count, base_delay)
        else:
            logger.warning("Unknown backoff strategy: %s, using exponential", strategy)
            return self._exponential_backoff(retry_count, base_delay, factor)
    
    def _linear_backoff(self, retry_count: int, base_delay: float) -> float:
        """
        Calculate delay using linear backoff.
        
        Args:
            retry_count: Current retry count (0-based)
            base_delay: Base delay in seconds
            
        Returns:
            Delay in seconds
        """
        delay = base_delay * (retry_count + 1)
        
        # Apply jitter if enabled
        if self.config.jitter_enabled:
            jitter = random.uniform(0, self.config.jitter_factor * delay)
            delay += jitter
        
        return delay
    
    def _exponential_backoff(self, retry_count: int, base_delay: float, factor: float) -> float:
        """
        Calculate delay using exponential backoff.
        
        Args:
            retry_count: Current retry count (0-based)
            base_delay: Base delay in seconds
            factor: Exponential factor
            
        Returns:
            Delay in seconds
        """
        delay = base_delay * (factor ** retry_count)
        
        # Apply jitter if enabled
        if self.config.jitter_enabled:
            jitter = random.uniform(0, self.config.jitter_factor * delay)
            delay += jitter
        
        return delay
    
    def _fibonacci_backoff(self, retry_count: int, base_delay: float) -> float:
        """
        Calculate delay using Fibonacci backoff.
        
        Args:
            retry_count: Current retry count (0-based)
            base_delay: Base delay in seconds
            
        Returns:
            Delay in seconds
        """
        # Calculate Fibonacci number
        a, b = 1, 1
        for _ in range(retry_count):
            a, b = b, a + b
        
        delay = base_delay * a
        
        # Apply jitter if enabled
        if self.config.jitter_enabled:
            jitter = random.uniform(0, self.config.jitter_factor * delay)
            delay += jitter
        
        return delay

