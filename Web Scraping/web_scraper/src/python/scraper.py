#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Python scraping implementation module.
Provides base scraper classes and functionality for web scraping operations.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException, Timeout

from ..config.config import ScraperConfig
from ..utils.helpers import validate_url, get_user_agent, setup_logger


class ScraperMethod(Enum):
    """Enum representing different scraping methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"


class ScraperError(Exception):
    """Base exception class for scraper errors."""
    pass


class RequestError(ScraperError):
    """Exception raised for errors in the HTTP request."""
    pass


class ParseError(ScraperError):
    """Exception raised for errors during content parsing."""
    pass


class AuthenticationError(ScraperError):
    """Exception raised for authentication errors."""
    pass


class RateLimitError(ScraperError):
    """Exception raised when rate limits are exceeded."""
    pass


@dataclass
class ScraperResponse:
    """Container for scraper response data."""
    status_code: int
    content: str
    headers: Dict[str, str]
    url: str
    elapsed: float
    success: bool
    error: Optional[str] = None
    parsed_content: Optional[Any] = None


class BaseRateLimiter:
    """Base class for rate limiting implementations."""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0.0
    
    def wait(self) -> None:
        """Wait if necessary to comply with rate limits."""
        if self.interval <= 0:
            return
            
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.interval:
            sleep_time = self.interval - time_since_last_request
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()


class BaseScraper(ABC):
    """
    Base scraper class providing core functionality for web scraping.
    
    Attributes:
        config: Scraper configuration
        session: Requests session for HTTP communication
        logger: Logger instance
        rate_limiter: Rate limiter instance
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the base scraper.
        
        Args:
            config: Scraper configuration (optional)
        """
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.logger = setup_logger(self.__class__.__name__, self.config.log_level)
        self.rate_limiter = BaseRateLimiter(self.config.requests_per_minute)
        
        # Setup session with default headers
        self.session.headers.update({
            'User-Agent': get_user_agent(self.config.user_agent),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        })
        
        # Configure proxies if specified
        if self.config.proxy:
            self.session.proxies.update({
                'http': self.config.proxy,
                'https': self.config.proxy
            })
    
    def __enter__(self):
        """Context manager enter method."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method, ensures resources are cleaned up."""
        self.close()
    
    def close(self):
        """Close session and clean up resources."""
        if hasattr(self, 'session') and self.session:
            self.session.close()
    
    def request(
        self, 
        url: str, 
        method: ScraperMethod = ScraperMethod.GET, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify: bool = True,
        allow_redirects: bool = True,
        parse_as: Optional[str] = None
    ) -> ScraperResponse:
        """
        Make an HTTP request to the specified URL.
        
        Args:
            url: URL to request
            method: HTTP method to use
            params: URL parameters
            data: Request body data
            headers: Additional headers
            cookies: Additional cookies
            timeout: Request timeout in seconds
            verify: Whether to verify SSL certificates
            allow_redirects: Whether to follow redirects
            parse_as: Content parsing method ('html', 'json', None)
            
        Returns:
            ScraperResponse object containing response data
            
        Raises:
            RequestError: If the request fails
        """
        if not validate_url(url):
            raise RequestError(f"Invalid URL: {url}")
        
        start_time = time.time()
        
        # Apply rate limiting
        self.rate_limiter.wait()
        
        # Merge custom headers with session headers
        merged_headers = dict(self.session.headers)
        if headers:
            merged_headers.update(headers)
            
        # Log request details
        self.logger.debug(f"Making {method.value} request to {url}")
        
        try:
            response = self.session.request(
                method=method.value,
                url=url,
                params=params,
                data=data,
                headers=merged_headers,
                cookies=cookies,
                timeout=timeout,
                verify=verify,
                allow_redirects=allow_redirects
            )
            
            elapsed = time.time() - start_time
            self.logger.debug(f"Received response: {response.status_code} in {elapsed:.2f}s")
            
            # Create response object
            scraper_response = ScraperResponse(
                status_code=response.status_code,
                content=response.text,
                headers=dict(response.headers),
                url=response.url,
                elapsed=elapsed,
                success=response.ok
            )
            
            # Parse content if requested
            if parse_as and response.ok:
                try:
                    if parse_as.lower() == 'html':
                        scraper_response.parsed_content = BeautifulSoup(response.text, 'html.parser')
                    elif parse_as.lower() == 'json':
                        scraper_response.parsed_content = response.json()
                except Exception as e:
                    self.logger.warning(f"Failed to parse response as {parse_as}: {e}")
                    scraper_response.error = f"Parse error: {str(e)}"
            
            return scraper_response
            
        except Timeout as e:
            self.logger.error(f"Request timed out: {e}")
            raise RequestError(f"Request timed out: {str(e)}")
        except RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise RequestError(f"Request failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise RequestError(f"Unexpected error: {str(e)}")
    
    def get(self, url: str, **kwargs) -> ScraperResponse:
        """Shorthand for making GET requests."""
        return self.request(url, ScraperMethod.GET, **kwargs)
    
    def post(self, url: str, **kwargs) -> ScraperResponse:
        """Shorthand for making POST requests."""
        return self.request(url, ScraperMethod.POST, **kwargs)
    
    async def async_request(
        self,
        url: str,
        method: ScraperMethod = ScraperMethod.GET,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify: bool = True,
        allow_redirects: bool = True,
        parse_as: Optional[str] = None
    ) -> ScraperResponse:
        """
        Make an asynchronous HTTP request to the specified URL.
        
        Args:
            url: URL to request
            method: HTTP method to use
            params: URL parameters
            data: Request body data
            headers: Additional headers
            cookies: Additional cookies
            timeout: Request timeout in seconds
            verify: Whether to verify SSL certificates
            allow_redirects: Whether to follow redirects
            parse_as: Content parsing method ('html', 'json', None)
            
        Returns:
            ScraperResponse object containing response data
            
        Raises:
            RequestError: If the request fails
        """
        if not validate_url(url):
            raise RequestError(f"Invalid URL: {url}")
        
        # Apply rate limiting (in async context)
        await asyncio.sleep(self.rate_limiter.interval)
        
        # Merge custom headers with default headers
        merged_headers = dict(self.session.headers)
        if headers:
            merged_headers.update(headers)
            
        start_time = time.time()
        self.logger.debug(f"Making async {method.value} request to {url}")
        
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            
            async with aiohttp.ClientSession(headers=merged_headers, cookies=cookies) as session:
                async with session.request(
                    method=method.value,
                    url=url,
                    params=params,
                    data=data,
                    timeout=timeout_obj,
                    ssl=verify,
                    allow_redirects=allow_redirects
                ) as response:
                    content = await response.text()
                    elapsed = time.time() - start_time
                    
                    self.logger.debug(f"Received async response: {response.status} in {elapsed:.2f}s")
                    
                    # Create response object
                    scraper_response = ScraperResponse(
                        status_code=response.status,
                        content=content,
                        headers=dict(response.headers),
                        url=str(response.url),
                        elapsed=elapsed,
                        success=response.status < 400
                    )
                    
                    # Parse content if requested
                    if parse_as and response.status < 400:
                        try:
                            if parse_as.lower() == 'html':
                                scraper_response.parsed_content = BeautifulSoup(content, 'html.parser')
                            elif parse_as.lower() == 'json':
                                scraper_response.parsed_content = await response.json()
                        except Exception as e:
                            self.logger.warning(f"Failed to parse response as {parse_as}: {e}")
                            scraper_response.error = f"Parse error: {str(e)}"
                    
                    return scraper_response
                    
        except asyncio.TimeoutError as e:
            self.logger.error(f"Async request timed out: {e}")
            raise RequestError(f"Async request timed out: {str(e)}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Async request failed: {e}")
            raise RequestError(f"Async request failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in async request: {e}")
            raise RequestError(f"Unexpected error in async request: {str(e)}")
    
    @abstractmethod
    def parse(self, response: ScraperResponse) -> Any:
        """
        Parse the response content.
        To be implemented by subclasses.
        
        Args:
            response: ScraperResponse object
            
        Returns:
            Parsed data in the appropriate format
        """
        pass
    
    @abstractmethod
    def run(self) -> Any:
        """
        Execute the scraping operation.
        To be implemented by subclasses.
        
        Returns:
            Scraped data in the appropriate format
        """
        pass


class WebScraper(BaseScraper):
    """
    Web scraper implementation for HTML content.
    
    This scraper specifically handles HTML web pages using BeautifulSoup.
    """
    
    def __init__(self, base_url: str, config: Optional[ScraperConfig] = None):
        """
        Initialize the web scraper.
        
        Args:
            base_url: Base URL for the website to scrape
            config: Scraper configuration
        """
        super().__init__(config)
        self.base_url = base_url
        self.visited_urls = set()
    
    def parse(self, response: ScraperResponse) -> BeautifulSoup:
        """
        Parse HTML content from the response.
        
        Args:
            response: ScraperResponse object
            
        Returns:
            BeautifulSoup object
            
        Raises:
            ParseError: If parsing fails
        """
        try:
            if not response.success:
                raise ParseError(f"Cannot parse unsuccessful response (status {response.status_code})")
                
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            self.logger.error(f"Failed to parse HTML: {e}")
            raise ParseError(f"Failed to parse HTML: {str(e)}")
    
    def extract_links(self, soup: BeautifulSoup, same_domain: bool = True) -> List[str]:
        """
        Extract links from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            same_domain: Whether to only return links from the same domain
            
        Returns:
            List of URL strings
        """
        links = []
        base_domain = urlparse(self.base_url).netloc
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            full_url = urljoin(self.base_url, href)
            
            if same_domain and urlparse(full_url).netloc != base_domain:
                continue
                
            if full_url not in self.visited_urls:
                links.append(full_url)
                
        return links
    
    def run(self, max_pages: int = 10, same_domain: bool = True) -> Dict[str, Any]:
        """
        Run the web scraper to collect data from multiple pages.
        
        Args:
            max_pages: Maximum number of pages to scrape
            same_domain: Whether to only follow links on the same domain
            
        Returns:
            Dictionary containing scraped data
        """
        results = {
            'base_url': self.base_url,
            'pages': [],
            'stats': {
                'pages_scraped': 0,
                'success_count': 0,
                'error_count': 0,
                'total_time': 0
            }
        }
        
        urls_to_visit = [self.base_url]
        start_time = time.time()
        
        while urls_to_visit and len(self.visited_urls) < max_pages:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            self.visited_urls.add(url)
            self.logger.info(f"Scraping page {len(self.visited_urls)}/{max_pages}: {url}")
            
            try:
                response = self.get(url, parse_as='html')
                results['stats']['pages_scraped'] += 1
                
                if response.success:
                    results['stats']['success_count'] += 1
                    page_data = {
                        'url': url,
                        'title': response.parsed_content.title.text.strip() if response.parsed_content.title else None,
                        'status_code': response.status_code,
                        'elapsed': response.elapsed
                    }
                    results['pages'].append(page_data)
                    
                    # Extract new links to visit
                    if len(self.visited_urls) < max_pages:
                        new_links = self.extract_links(response.parsed_content, same_domain)
                        urls_to_visit.extend([link for link in new_links if link not in self.visited_urls])
                else:
                    results['stats']['error_count'] += 1
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {e}")
                results['stats']['error_count'] += 1
        
        results['stats']['total_time'] = time.time() - start_time
        self.logger.info(f"Scraping completed: {results['stats']['pages_scraped']} pages in {results['stats']['total_time']:.2f}s")
        
        return results


class ApiScraper(BaseScraper):
    """
    API scraper implementation for JSON APIs.
    
    This scraper specifically handles JSON API endpoints.
    """
    
    def __init__(self, base_url: str, config: Optional[ScraperConfig] = None):
        """
        Initialize the API scraper.
        
        Args:
            base_url: Base URL for the API
            config: Scraper configuration
        """
        super().__init__(config)
        self.base_url = base_url
        self.endpoints = {}
    
    def add_endpoint(self, name: str, path: str, method: ScraperMethod = ScraperMethod.GET,
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None):
        """
        Add an API endpoint to the scraper.
        
        Args:
            name: Endpoint name for reference
            path: Endpoint path (will be appended to base_url)
            method: HTTP method to use
            params: Default URL parameters
            data: Default request body data
            headers: Default headers for this endpoint
        """
        self.endpoints[name] = {
            'path': path,
            'method': method,
            'params': params or {},
            'data': data or {},
            'headers': headers or {}
        }
        self.logger.debug(f"Added endpoint '{name}': {method.value} {path}")
    
    def parse(self, response: ScraperResponse) -> Dict[str, Any]:
        """
        Parse JSON content from the response.
        
        Args:
            response: ScraperResponse object
            
        Returns:
            Parsed JSON data as a dictionary
            
        Raises:
            ParseError: If parsing fails
        """
        try:
            if not response.success:
                raise ParseError(f"Cannot parse unsuccessful response (status {response.status_code})")
                
            if response.parsed_content:
                return response.parsed_content
                
            return requests.Response.json(response.content)
        except Exception as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise ParseError(f"Failed to parse JSON: {str(e)}")
    
    def call_endpoint(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a specific API endpoint by name.
        
        Args:
            name: Name of the endpoint to call
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Parsed JSON response
            
        Raises:
            RequestError: If the endpoint doesn't exist or the request fails
        """
        if name not in self.endpoints:
            raise RequestError(f"Endpoint '{name}' not defined")
            
        endpoint = self.endpoints[name]
        url = urljoin(self.base_url, endpoint['path'])
        
        # Merge default parameters with provided ones
        params = dict(endpoint['params'])
        if 'params' in kwargs:
            params.update(kwargs.pop('params'))
            
        data = dict(endpoint['data'])
        if 'data' in kwargs:
            data.update(kwargs.pop('data'))
            
        headers = dict(endpoint['headers'])
        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))
        
        response = self.request(
            url=url,
            method=endpoint['method'],
            params=params,
            data=data,
            headers=headers,
            parse_as='json',
            **kwargs
        )
        
        return self.parse(response)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the API scraper to collect data from all endpoints.
        
        Returns:
            Dictionary containing scraped data from all endpoints
        """
        results = {
            'base_url': self.base_url,
            'endpoints': {},
            'stats': {
                'total_endpoints': len(self.endpoints),
                'success_count': 0,
                'error_count': 0,
                'total_time': 0
            }
        }
        
        start_time = time.time()
        
        for name, endpoint in self.endpoints.items():
            self.logger.info(f"Calling endpoint '{name}': {endpoint['method'].value} {endpoint['path']}")
            
            try:
                data = self.call_endpoint(name)
                results['endpoints'][name] = data
                results['stats']['success_count'] += 1
            except Exception as e:
                self.logger.error(f"Error calling endpoint '{name}': {e}")
                results['endpoints'][name] = {'error': str(e)}
                results['stats']['error_count'] += 1
        
        results['stats']['total_time'] = time.time() - start_time
        self.logger.info(f"API scraping completed: {results['stats']['success_count']}/{results['stats']['total_endpoints']} endpoints in {results['stats']['total_time']:.2f}s")
        
        return results

