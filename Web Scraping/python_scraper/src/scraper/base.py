#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Scraper Module

This module provides the core scraping functionality with rate limiting,
authentication handling, request management, and error handling capabilities.
It serves as the foundation for all specialized scrapers in the application.
"""

import time
import random
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from urllib.parse import urljoin, urlparse
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

import requests
from requests.exceptions import RequestException, Timeout, TooManyRedirects
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger("webscraper.base")


class RequestConfig(BaseModel):
    """Configuration for HTTP requests."""
    
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    follow_redirects: bool = Field(True, description="Whether to follow redirects")
    verify_ssl: bool = Field(True, description="Whether to verify SSL certificates")
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        },
        description="HTTP headers to include in requests"
    )


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    
    enabled: bool = Field(True, description="Whether rate limiting is enabled")
    requests_per_second: float = Field(1.0, description="Maximum requests per second")
    max_concurrent: int = Field(5, description="Maximum concurrent requests")
    jitter: float = Field(0.1, description="Random jitter factor to add to delays")


class ProxyConfig(BaseModel):
    """Configuration for proxy support."""
    
    enabled: bool = Field(False, description="Whether to use proxies")
    rotation_enabled: bool = Field(False, description="Whether to rotate proxies")
    rotation_interval: int = Field(10, description="Number of requests before rotating proxies")
    providers: List[str] = Field(default_factory=list, description="List of proxy providers or URLs")
    proxies: List[Dict[str, str]] = Field(default_factory=list, description="List of proxy configurations")


class AuthConfig(BaseModel):
    """Configuration for authentication."""
    
    method: Optional[str] = Field(None, description="Authentication method (basic, token, oauth)")
    credentials: Dict[str, Any] = Field(default_factory=dict, description="Authentication credentials")
    token_refresh_url: Optional[str] = Field(None, description="URL for refreshing OAuth tokens")
    token_refresh_interval: int = Field(3600, description="Seconds before token refresh")


class ScraperConfig(BaseModel):
    """Master configuration for the scraper."""
    
    request: RequestConfig = Field(default_factory=RequestConfig, description="HTTP request configuration")
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig, description="Rate limiting configuration")
    proxy: ProxyConfig = Field(default_factory=ProxyConfig, description="Proxy configuration")
    auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication configuration")
    max_depth: int = Field(3, description="Maximum depth for crawling")
    respect_robots_txt: bool = Field(True, description="Whether to respect robots.txt rules")
    storage_path: Path = Field(default_factory=lambda: Path("./scraper_output/data"), description="Path to store scraped data")


class ScraperResult(BaseModel):
    """Model for storing scraper results."""
    
    url: str = Field(..., description="URL that was scraped")
    status_code: int = Field(..., description="HTTP status code of the response")
    content_type: str = Field(..., description="Content type of the response")
    data: Any = Field(..., description="Extracted data from the response")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the data was scraped")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the scrape")


class BaseScraper(ABC):
    """
    Base class for all scrapers in the application.
    
    This class provides core functionality for making HTTP requests,
    handling authentication, rate limiting, and error handling.
    Specialized scrapers should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], ScraperConfig]] = None):
        """
        Initialize the scraper with the given configuration.
        
        Args:
            config: A configuration dictionary or ScraperConfig object.
                   If None, default configuration is used.
        """
        if config is None:
            self.config = ScraperConfig()
        elif isinstance(config, dict):
            self.config = ScraperConfig.model_validate(config)
        else:
            self.config = config
        
        # Set up session for requests
        self.session = requests.Session()
        self.session.headers.update(self.config.request.headers)
        
        # Set up rate limiting
        self._last_request_time = time.time()
        self._request_count = 0
        self._current_proxy_index = 0
        
        # Authentication state
        self._auth_token = None
        self._token_expiry = None
        
        # Setup complete
        logger.info("Scraper initialized with configuration: %s", self.config.model_dump())
    
    def _update_auth_header(self):
        """Update the authentication header based on the current auth configuration."""
        if not self.config.auth.method:
            return
        
        # Check if token needs refresh
        if (self._token_expiry is not None and 
            datetime.now() > self._token_expiry and 
            self.config.auth.token_refresh_url):
            self._refresh_token()
        
        if self.config.auth.method.lower() == "basic":
            if "username" in self.config.auth.credentials and "password" in self.config.auth.credentials:
                username = self.config.auth.credentials["username"]
                password = self.config.auth.credentials["password"]
                auth = requests.auth.HTTPBasicAuth(username, password)
                self.session.auth = auth
        
        elif self.config.auth.method.lower() == "token":
            if "token" in self.config.auth.credentials:
                token = self.config.auth.credentials["token"]
                self.session.headers.update({"Authorization": f"Bearer {token}"})
        
        elif self.config.auth.method.lower() == "oauth":
            if self._auth_token:
                self.session.headers.update({"Authorization": f"Bearer {self._auth_token}"})
    
    def _refresh_token(self):
        """Refresh the OAuth token if needed and possible."""
        if not self.config.auth.token_refresh_url:
            logger.warning("Token refresh requested but no refresh URL configured")
            return
        
        try:
            # This is a simplified implementation - in practice, this would depend on the OAuth provider
            refresh_token = self.config.auth.credentials.get("refresh_token")
            client_id = self.config.auth.credentials.get("client_id")
            client_secret = self.config.auth.credentials.get("client_secret")
            
            if not refresh_token or not client_id or not client_secret:
                logger.error("Missing required OAuth credentials for token refresh")
                return
            
            response = requests.post(
                self.config.auth.token_refresh_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=self.config.request.timeout
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self._auth_token = token_data.get("access_token")
                
                # Calculate token expiry
                expires_in = token_data.get("expires_in", self.config.auth.token_refresh_interval)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
                
                # Update session headers
                self.session.headers.update({"Authorization": f"Bearer {self._auth_token}"})
                
                logger.info("OAuth token refreshed successfully, expires at %s", self._token_expiry)
            else:
                logger.error("Failed to refresh OAuth token: HTTP %d", response.status_code)
        
        except Exception as e:
            logger.exception("Error refreshing OAuth token: %s", str(e))
    
    def _get_proxy(self):
        """Get the current proxy configuration, rotating if needed."""
        if not self.config.proxy.enabled or not self.config.proxy.proxies:
            return None
        
        # Rotate proxy if needed
        if (self.config.proxy.rotation_enabled and 
            self._request_count % self.config.proxy.rotation_interval == 0):
            self._current_proxy_index = (self._current_proxy_index + 1) % len(self.config.proxy.proxies)
            logger.debug("Rotating to proxy %d", self._current_proxy_index)
        
        return self.config.proxy.proxies[self._current_proxy_index]
    
    def _apply_rate_limiting(self):
        """Apply rate limiting based on configuration."""
        if not self.config.rate_limit.enabled:
            return
        
        # Calculate time since last request
        now = time.time()
        elapsed = now - self._last_request_time
        
        # Calculate required delay for rate limiting
        min_interval = 1.0 / self.config.rate_limit.requests_per_second
        
        if elapsed < min_interval:
            # Add jitter to avoid synchronized requests
            jitter = random.uniform(0, self.config.rate_limit.jitter * min_interval)
            delay = min_interval - elapsed + jitter
            logger.debug("Rate limiting: sleeping for %.2f seconds", delay)
            time.sleep(delay)
        
        self._last_request_time = time.time()
    
    def make_request(
        self, 
        url: str, 
        method: str = "GET", 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        allow_redirects: Optional[bool] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Make an HTTP request with rate limiting, authentication, and error handling.
        
        Args:
            url: The URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Form data or raw data to send
            json_data: JSON data to send
            headers: Additional headers to include
            cookies: Cookies to include
            allow_redirects: Whether to follow redirects (overrides config)
            timeout: Request timeout in seconds (overrides config)
            
        Returns:
            Tuple of (status_code, content_type, json_data, text_content)
        """
        # Apply rate limiting
        self._apply_rate_limiting()
        
        # Update authentication if needed
        self._update_auth_header()
        
        # Get proxy if configured
        proxy = self._get_proxy()
        
        # Prepare request parameters
        request_kwargs = {
            "method": method.upper(),
            "url": url,
            "params": params,
            "data": data,
            "json": json_data,
            "timeout": timeout if timeout is not None else self.config.request.timeout,
            "allow_redirects": allow_redirects if allow_redirects is not None else self.config.request.follow_redirects,
            "verify": self.config.request.verify_ssl,
        }
        
        # Add optional parameters if provided
        if headers:
            merged_headers = self.session.headers.copy()
            merged_headers.update(headers)
            request_kwargs["headers"] = merged_headers
            
        if cookies:
            request_kwargs["cookies"] = cookies
            
        if proxy:
            request_kwargs["proxies"] = proxy
        
        # Track request count for proxy rotation
        self._request_count += 1
        
        # Make the request with retries
        retries = 0
        max_retries = self.config.request.max_retries
        retry_delay = self.config.request.retry_delay
        
        while retries <= max_retries:
            try:
                logger.debug("Making %s request to %s (retry %d/%d)", 
                           method.upper(), url, retries, max_retries)
                
                response = self.session.request(**request_kwargs)
                
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                
                # Try to parse JSON if content type is JSON
                json_data = None
                if "application/json" in content_type:
                    try:
                        json_data = response.json()
                    except ValueError:
                        logger.warning("Failed to parse JSON response from %s", url)
                
                return response.status_code, content_type, json_data, response.text
                
            except (RequestException, Timeout, TooManyRedirects) as e:
                retries += 1
                logger.warning("Request to %s failed (%s), retry %d/%d",
                              url, str(e), retries, max_retries)
                
                if retries <= max_retries:
                    # Add jitter to retry delay
                    jitter = random.uniform(0, 0.25 * retry_delay)
                    time.sleep(retry_delay + jitter)
                    
                    # Exponential backoff for retry delay
                    retry_delay *= 2
                else:
                    logger.error("Max retries exceeded for request to %s", url)
                    return 0, None, None, None
    
    def fetch_html(self, url: str, **kwargs) -> Tuple[int, Optional[BeautifulSoup]]:
        """
        Fetch an HTML page and return a BeautifulSoup object.
        
        Args:
            url: The URL to fetch
            **kwargs: Additional arguments to pass to make_request
            
        Returns:
            Tuple of (status_code, BeautifulSoup object or None if failed)
        """
        status, content_type, _, html = self.make_request(url, **kwargs)
        
        if status < 200 or status >= 300:
            logger.warning("Failed to fetch HTML from %s: HTTP %d", url, status)
            return status, None
            
        if html and ("text/html" in content_type or "application/xhtml+xml" in content_type):
            try:
                soup = BeautifulSoup(html, "lxml")
                return status, soup
            except Exception as e:
                logger.error("Failed to parse HTML from %s: %s", url, str(e))
                return status, None
        else:
            logger.warning("Response from %s is not HTML (Content-Type: %s)", url, content_type)
            return status, None
    
    def fetch_json(self, url: str, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Fetch JSON data from a URL.
        
        Args:
            url: The URL to fetch
            **kwargs: Additional arguments to pass to make_request
            
        Returns:
            Tuple of (status_code, parsed JSON or None if failed)
        """
        status, content_type, json_data, text = self.make_request(url, **kwargs)
        
        if status < 200 or status >= 300:
            logger.warning("Failed to fetch JSON from %s: HTTP %d", url, status)
            return status, None
            
        if json_data is not None:
            return status, json_data
        
        # Try to parse JSON even if content type doesn't match
        if text:
            try:
                json_data = json.loads(text)
                logger.debug("Successfully parsed JSON from %s despite content type %s", url, content_type)
                return status, json_data
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from %s (Content-Type: %s)", url, content_type)
                return status, None
        
        return status, None
    
    async def make_async_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        allow_redirects: Optional[bool] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Make an asynchronous HTTP request.
        
        This is similar to make_request but uses aiohttp for asynchronous operation.
        
        Args:
            url: The URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Form data or raw data to send
            json_data: JSON data to send
            headers: Additional headers to include
            cookies: Cookies to include
            allow_redirects: Whether to follow redirects (overrides config)
            timeout: Request timeout in seconds (overrides config)
            
        Returns:
            Tuple of (status_code, content_type, json_data, text_content)
        """
        # Prepare headers
        request_headers = dict(self.config.request.headers)
        if headers:
            request_headers.update(headers)
        
        # Update auth headers
        self._update_auth_header()
        if "Authorization" in self.session.headers:
            request_headers["Authorization"] = self.session.headers["Authorization"]
        
        # Apply rate limiting (simplified for async context)
        await asyncio.sleep(1.0 / self.config.rate_limit.requests_per_second)
        
        # Get proxy
        proxy = self._get_proxy()
        proxy_url = None
        if proxy and "http" in proxy:
            proxy_url = proxy["http"]
        
        # Set timeout
        if timeout is None:
            timeout = self.config.request.timeout
        
        # Make async request
        try:
            async with aiohttp.ClientSession(headers=request_headers, cookies=cookies) as session:
                async with session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    allow_redirects=(
                        allow_redirects 
                        if allow_redirects is not None 
                        else self.config.request.follow_redirects
                    ),
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    proxy=proxy_url,
                    ssl=None if not self.config.request.verify_ssl else None,
                ) as response:
                    content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                    text = await response.text()
                    
                    json_data = None
                    if "application/json" in content_type:
                        try:
                            json_data = await response.json()
                        except ValueError:
                            logger.warning("Failed to parse JSON response from %s", url)
                    
                    return response.status, content_type, json_data, text
                    
        except Exception as e:
            logger.error("Async request to %s failed: %s", url, str(e))
            return 0, None, None, None
    
    async def fetch_multiple(
        self,
        urls: List[str],
        callback: Optional[Callable[[str, int, Optional[str]], None]] = None,
        max_concurrent: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Tuple[int, Optional[str]]]:
        """
        Fetch multiple URLs concurrently with rate limiting.
        
        Args:
            urls: List of URLs to fetch
            callback: Optional callback function to call with results as they arrive
            max_concurrent: Maximum number of concurrent requests (overrides config)
            **kwargs: Additional arguments to pass to make_async_request
            
        Returns:
            Dictionary mapping URLs to (status_code, content) tuples
        """
        if max_concurrent is None:
            max_concurrent = self.config.rate_limit.max_concurrent
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def fetch_with_semaphore(url):
            async with semaphore:
                status, content_type, _, content = await self.make_async_request(url, **kwargs)
                results[url] = (status, content)
                if callback:
                    callback(url, status, content)
        
        # Create tasks for all URLs
        tasks = [fetch_with_semaphore(url) for url in urls]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        return results
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract all links from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#"):
                continue  # Skip anchor links
                
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
        
        return links
    
    def extract_forms(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract form information from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of dictionaries containing form information
        """
        forms = []
        for form in soup.find_all("form"):
            form_data = {
                "action": urljoin(base_url, form.get("action", "")),
                "method": form.get("method", "get").upper(),
                "fields": []
            }
            
            for input_field in form.find_all("input"):
                field = {
                    "name": input_field.get("name", ""),
                    "type": input_field.get("type", "text"),
                    "required": input_field.has_attr("required"),
                    "value": input_field.get("value", "")
                }
                if field["name"]:  # Only include fields with names
                    form_data["fields"].append(field)
            
            forms.append(form_data)
        
        return forms
    
    def save_result(self, result: ScraperResult, filename: Optional[str] = None) -> Path:
        """
        Save a scraper result to disk.
        
        Args:
            result: The ScraperResult to save
            filename: Optional filename to use
            
        Returns:
            Path to the saved file
        """
        # Create storage directory if it doesn't exist
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            parsed_url = urlparse(result.url)
            domain = parsed_url.netloc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{domain}_{timestamp}.json"
        
        # Save result to file
        file_path = self.config.storage_path / filename
        with open(file_path, "w") as f:
            f.write(result.model_dump_json(indent=2))
        
        logger.info("Saved scraper result to %s", file_path)
        return file_path
    
    @abstractmethod
    def scrape(self, url: str, **kwargs) -> ScraperResult:
        """
        Scrape data from the given URL.
        
        This is an abstract method that must be implemented by subclasses.
        
        Args:
            url: The URL to scrape
            **kwargs: Additional scraper-specific parameters
            
        Returns:
            A ScraperResult object containing the scraped data
        """
        pass
    
    def close(self):
        """Clean up resources used by the scraper."""
        self.session.close()
        logger.debug("Scraper resources released")
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context."""
        self.close()

