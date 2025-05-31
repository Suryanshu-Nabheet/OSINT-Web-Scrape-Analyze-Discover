#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Scraper Module

This module provides functionality for discovering and interacting with
API endpoints on websites, including API schema detection and data extraction.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Pattern, Match
from urllib.parse import urljoin, urlparse
import asyncio
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import jsonschema
from pydantic import BaseModel, Field

from .base import BaseScraper, ScraperResult, ScraperConfig

# Set up logging
logger = logging.getLogger("webscraper.api_scraper")


class ApiEndpoint(BaseModel):
    """Model for storing API endpoint information."""
    
    url: str = Field(..., description="The URL of the API endpoint")
    method: str = Field("GET", description="HTTP method (GET, POST, etc.)")
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Parameters accepted by the endpoint")
    response_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema of the response")
    description: Optional[str] = Field(None, description="Description of the endpoint")
    requires_auth: bool = Field(False, description="Whether the endpoint requires authentication")
    content_type: Optional[str] = Field(None, description="Expected content type of the response")
    sample_response: Optional[Any] = Field(None, description="Sample response from the endpoint")
    discovered_from: Optional[str] = Field(None, description="Source where this endpoint was discovered")


class ApiScraperConfig(BaseModel):
    """Configuration specific to the API scraper."""
    
    js_analysis_enabled: bool = Field(True, description="Whether to analyze JavaScript files for API endpoints")
    pattern_matching_enabled: bool = Field(True, description="Whether to use pattern matching for API discovery")
    known_api_paths: List[str] = Field(
        default_factory=lambda: [
            "/api/", "/v1/", "/v2/", "/graphql", "/rest/", "/service/", "/services/", 
            "/webservice/", "/wp-json/", "/ajax/"
        ],
        description="Common API path patterns to look for"
    )
    common_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH"],
        description="HTTP methods to test when discovering endpoints"
    )
    test_endpoints: bool = Field(False, description="Whether to send test requests to discovered endpoints")
    max_test_requests: int = Field(5, description="Maximum number of test requests to send")
    extract_schema: bool = Field(True, description="Whether to attempt to extract JSON schema from responses")
    graphql_enabled: bool = Field(True, description="Whether to look for GraphQL endpoints")


class ApiScraper(BaseScraper):
    """
    API Scraper for discovering and interacting with API endpoints.
    
    This class extends the BaseScraper to provide specialized functionality
    for discovering API endpoints, detecting schemas, and extracting data
    from APIs found on websites.
    """
    
    def __init__(
        self, 
        config: Optional[Union[Dict[str, Any], ScraperConfig]] = None,
        api_config: Optional[Union[Dict[str, Any], ApiScraperConfig]] = None
    ):
        """
        Initialize the API scraper with the given configuration.
        
        Args:
            config: General scraper configuration
            api_config: API-specific scraper configuration
        """
        super().__init__(config)
        
        # Set up API-specific configuration
        if api_config is None:
            self.api_config = ApiScraperConfig()
        elif isinstance(api_config, dict):
            self.api_config = ApiScraperConfig.model_validate(api_config)
        else:
            self.api_config = api_config
        
        # Set up regex patterns for API endpoint discovery
        self._setup_patterns()
        
        # Track discovered endpoints to avoid duplicates
        self.discovered_endpoints: Set[str] = set()
        self.endpoints: List[ApiEndpoint] = []
        
        logger.info("API Scraper initialized with configuration: %s", self.api_config.model_dump())
    
    def _setup_patterns(self):
        """Set up regex patterns for API endpoint discovery."""
        # Pattern to match API endpoints in JavaScript
        self.api_url_pattern: Pattern = re.compile(
            r'(?:fetch|axios\.(?:get|post|put|delete|patch)|api|ajax|XMLHttpRequest)'
            r'[\s\S]{0,50}?["\']([^"\']*?/(?:api|v1|v2|graphql|rest|service|services|wp-json)[^"\']*?)["\']',
            re.IGNORECASE
        )
        
        # Pattern to match API endpoints in HTML
        self.html_api_pattern: Pattern = re.compile(
            r'(?:data-url|api-url|endpoint|action)=["\']([^"\']*?/(?:api|v1|v2|graphql|rest|service|services|wp-json)[^"\']*?)["\']',
            re.IGNORECASE
        )
        
        # Pattern to match GraphQL endpoints
        self.graphql_pattern: Pattern = re.compile(
            r'(?:graphql|gql|query)[\s\S]{0,100}?["\']([^"\']*?(?:graphql|gql)[^"\']*?)["\']',
            re.IGNORECASE
        )
    
    def scrape(self, url: str, **kwargs) -> ScraperResult:
        """
        Scrape API endpoints from the given URL.
        
        This method discovers API endpoints from HTML, JavaScript, and other
        sources on the target website.
        
        Args:
            url: The URL to scrape
            **kwargs: Additional scraper-specific parameters
            
        Returns:
            A ScraperResult object containing the discovered API endpoints
        """
        logger.info("Scraping API endpoints from %s", url)
        
        depth = kwargs.get("depth", self.config.max_depth)
        visited_urls = set()
        
        # Discover API endpoints
        self._discover_api_endpoints(url, depth, visited_urls)
        
        # Convert endpoints to a structured format
        endpoints_data = [endpoint.model_dump() for endpoint in self.endpoints]
        
        return ScraperResult(
            url=url,
            status_code=200,  # Placeholder status code
            content_type="application/json",
            data={
                "endpoints": endpoints_data,
                "count": len(endpoints_data),
                "discovered_from": list(visited_urls)
            },
            metadata={
                "depth": depth,
                "js_files_analyzed": len(visited_urls),
                "api_config": self.api_config.model_dump()
            }
        )
    
    def _discover_api_endpoints(self, url: str, depth: int, visited_urls: Set[str]):
        """
        Recursively discover API endpoints from the given URL.
        
        Args:
            url: The URL to start discovery from
            depth: Maximum depth for crawling
            visited_urls: Set of already visited URLs
        """
        if depth <= 0 or url in visited_urls:
            return
        
        visited_urls.add(url)
        logger.debug("Discovering API endpoints from %s (depth: %d)", url, depth)
        
        # Fetch the HTML page
        status, soup = self.fetch_html(url)
        
        if status < 200 or status >= 300 or soup is None:
            logger.warning("Failed to fetch HTML from %s: HTTP %d", url, status)
            return
        
        # Look for API endpoints in HTML attributes
        self._extract_api_from_html(soup, url)
        
        # Find and analyze JavaScript files
        if self.api_config.js_analysis_enabled:
            self._extract_api_from_javascript(soup, url, visited_urls, depth)
        
        # Look for common API paths
        if self.api_config.pattern_matching_enabled:
            self._discover_common_api_paths(url)
        
        # If GraphQL is enabled, look for GraphQL endpoints
        if self.api_config.graphql_enabled:
            self._discover_graphql_endpoints(soup, url)
        
        # Follow links for deeper crawling
        if depth > 1:
            links = self.extract_links(soup, url)
            base_domain = urlparse(url).netloc
            
            for link in links:
                link_domain = urlparse(link).netloc
                # Only follow links within the same domain
                if link_domain == base_domain and link not in visited_urls:
                    self._discover_api_endpoints(link, depth - 1, visited_urls)
    
    def _extract_api_from_html(self, soup: BeautifulSoup, base_url: str):
        """
        Extract API endpoints from HTML attributes.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
        """
        # Look for API endpoints in HTML attributes
        for attr in ["data-url", "api-url", "endpoint", "action", "src"]:
            for element in soup.find_all(attrs={attr: True}):
                attr_value = element.get(attr, "")
                if any(path in attr_value for path in self.api_config.known_api_paths):
                    absolute_url = urljoin(base_url, attr_value)
                    self._add_endpoint(
                        absolute_url, 
                        "GET", 
                        discovered_from=f"HTML attribute '{attr}'"
                    )
        
        # Use regex pattern to find more API endpoints
        html_str = str(soup)
        for match in self.html_api_pattern.finditer(html_str):
            if match:
                endpoint_url = match.group(1)
                absolute_url = urljoin(base_url, endpoint_url)
                self._add_endpoint(
                    absolute_url, 
                    "GET", 
                    discovered_from="HTML pattern match"
                )
    
    def _extract_api_from_javascript(
        self, 
        soup: BeautifulSoup, 
        base_url: str, 
        visited_urls: Set[str],
        depth: int
    ):
        """
        Extract API endpoints from JavaScript files.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            visited_urls: Set of already visited URLs
            depth: Current crawling depth
        """
        # Find all script tags with src attribute
        for script in soup.find_all("script", src=True):
            script_url = urljoin(base_url, script["src"])
            
            # Skip already visited scripts and external domains
            if script_url in visited_urls or urlparse(script_url).netloc != urlparse(base_url).netloc:
                continue
            
            # Fetch JavaScript file
            status, content_type, _, js_content = self.make_request(script_url)
            
            if status < 200 or status >= 300 or not js_content:
                logger.warning("Failed to fetch JavaScript from %s: HTTP %d", script_url, status)
                continue
            
            visited_urls.add(script_url)
            
            # Look for API endpoints in JavaScript
            self._analyze_javascript(js_content, base_url, script_url)
        
        # Also analyze inline scripts
        for script in soup.find_all("script", src=False):
            js_content = script.string
            if js_content:
                self._analyze_javascript(js_content, base_url, f"{base_url}#inline-script")
    
    def _analyze_javascript(self, js_content: str, base_url: str, source_url: str):
        """
        Analyze JavaScript content for API endpoints.
        
        Args:
            js_content: JavaScript content to analyze
            base_url: Base URL for resolving relative links
            source_url: Source URL of the JavaScript content
        """
        if not js_content:
            return
        
        # Look for API endpoints using regex
        for match in self.api_url_pattern.finditer(js_content):
            if match:
                endpoint_url = match.group(1)
                absolute_url = urljoin(base_url, endpoint_url)
                
                # Try to determine HTTP method from context
                method = self._determine_method_from_context(js_content, match)
                
                self._add_endpoint(
                    absolute_url, 
                    method, 
                    discovered_from=f"JavaScript file: {source_url}"
                )
    
    def _determine_method_from_context(self, js_content: str, match: Match) -> str:
        """
        Try to determine the HTTP method from JavaScript context.
        
        Args:
            js_content: JavaScript content
            match: Regex match object
            
        Returns:
            HTTP method string (defaults to "GET")
        """
        # Get some context around the match
        start = max(0, match.start() - 50)
        end = min(len(js_content), match.end() + 50)
        context = js_content[start:end]
        
        # Look for method indicators
        if re.search(r'\.post\(', context, re.IGNORECASE):
            return "POST"
        elif re.search(r'\.put\(', context, re.IGNORECASE):
            return "PUT"
        elif re.search(r'\.delete\(', context, re.IGNORECASE):
            return "DELETE"
        elif re.search(r'\.patch\(', context, re.IGNORECASE):
            return "PATCH"
        elif re.search(r'\.get\(', context, re.IGNORECASE):
            return "GET"
        
        # Default to GET if method cannot be determined
        return "GET"
    
    def _discover_common_api_paths(self, base_url: str):
        """
        Try common API paths to discover endpoints.
        
        Args:
            base_url: Base URL for the website
        """
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        for path in self.api_config.known_api_paths:
            api_url = f"{base_domain}{path}"
            
            # Skip testing if not enabled
            if not self.api_config.test_endpoints:
                self._add_endpoint(api_url, "GET", discovered_from="Common API path pattern")
                continue
            
            # Try to access the API endpoint
            status, content_type, json_data, _ = self.make_request(api_url)
            
            if status >= 200 and status < 500:  # Include 4xx responses as they might be auth errors
                is_api = False
                
                # Check if it's a potential API endpoint
                if content_type and "application/json" in content_type:
                    is_api = True
                elif status == 401 or status == 403:  # Unauthorized or Forbidden
                    is_api = True
                elif json_data is not None:
                    is_api = True
                
                if is_api:
                    self._add_endpoint(
                        api_url, 
                        "GET", 
                        content_type=content_type,
                        requires_auth=(status == 401 or status == 403),
                        sample_response=json_data,
                        discovered_from="Common API path pattern (verified)"
                    )
    
    def _discover_graphql_endpoints(self, soup: BeautifulSoup, base_url: str):
        """
        Discover GraphQL endpoints.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
        """
        # Look for GraphQL endpoints in HTML
        html_str = str(soup)
        for match in self.graphql_pattern.finditer(html_str):
            if match:
                endpoint_url = match.group(1)
                absolute_url = urljoin(base_url, endpoint_url)
                self._add_endpoint(
                    absolute_url, 
                    "POST",  # GraphQL typically uses POST
                    content_type="application/json",
                    discovered_from="GraphQL pattern match"
                )
        
        # Try common GraphQL paths
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        for path in ["/graphql", "/gql", "/api/graphql", "/v1/graphql"]:
            graphql_url = f"{base_domain}{path}"
            
            # Try a simple introspection query if testing is enabled
            if self.api_config.test_endpoints:
                introspection_query = {
                    "query": "{__schema{queryType{name}}}"
                }
                
                status, content_type, json_data, _ = self.make_request(
                    graphql_url, 
                    method="POST",
                    json_data=introspection_query,
                    headers={"Content-Type": "application/json"}
                )
                
                is_graphql = False
                if status >= 200 and status < 300 and json_data is not None:
                    # Check if response looks like GraphQL
                    if "data" in json_data or "errors" in json_data:
                        is_graphql = True
                
                if is_graphql or status == 401 or status == 403:
                    self._add_endpoint(
                        graphql_url, 
                        "POST",
                        content_type="application/json",
                        requires_auth=(status == 401 or status == 403),
                        sample_response=json_data,
                        discovered_from="GraphQL endpoint (verified)"
                    )
            else:
                # Just add as a potential endpoint without testing
                self._add_endpoint(
                    graphql_url, 
                    "POST",
                    content_type="application/json",
                    discovered_from="Common GraphQL path pattern"
                )
    
    def _add_endpoint(
        self, 
        url: str, 
        method: str, 
        parameters: List[Dict[str, Any]] = None,
        response_schema: Dict[str, Any] = None,
        description: str = None,
        requires_auth: bool = False,
        content_type: str = None,
        sample_response: Any = None,
        discovered_from: str = None
    ):
        """
        Add a discovered API endpoint.
        
        Args:
            url: The URL of the API endpoint
            method: HTTP method
            parameters: Parameters accepted by the endpoint
            response_schema: JSON schema of the response
            description: Description of the endpoint
            requires_auth: Whether the endpoint requires authentication
            content_type: Expected content type of the response
            sample_response: Sample response from the endpoint
            discovered_from: Source where this endpoint was discovered
        """
        # Skip if already discovered
        if url in self.discovered_endpoints:
            return
        
        self.discovered_endpoints.add(url)
        
        # Extract schema from sample response if available
        if self.api_config.extract_schema and sample_response and response_schema is None:
            response_schema = self._extract_schema_from_sample(sample_response)
        
        # Create endpoint object
        endpoint = ApiEndpoint(
            url=url,
            method=method,
            parameters=parameters or [],
            response_schema=response_schema,
            description=description,
            requires_auth=requires_auth,
            content_type=content_type,
            sample_response=sample_response,
            discovered_from=discovered_from
        )
        
        self.endpoints.append(endpoint)
        logger.info("Discovered API endpoint: %s %s (from: %s)", method, url, discovered_from)
    
    def _extract_schema_from_sample(self, sample: Any) -> Optional[Dict[str, Any]]:
        """
        Extract a JSON schema from a sample response.
        
        This is a simplified implementation. In a real implementation,
        this would use a more robust schema generation approach.
        
        Args:
            sample: Sample data to extract schema from
            
        Returns:
            JSON schema or None if schema cannot be extracted
        """
        if not sample:
            return None
        
        try:
            if isinstance(sample, (str, bytes)):
                try:
                    sample = json.loads(sample)
                except json.JSONDecodeError:
                    return None
            
            # Simple schema generation based on types
            def generate_schema(data):
                if data is None:
                    return {"type": "null"}
                elif isinstance(data, bool):
                    return {"type": "boolean"}
                elif isinstance(data, int):
                    return {"type": "integer"}
                elif isinstance(data, float):
                    return {"type": "number"}
                elif isinstance(data, str):
                    return {"type": "string"}
                elif isinstance(data, list):
                    if not data:
                        return {"type": "array", "items": {}}
                    # Use the first item as representative
                    return {
                        "type": "array",
                        "items": generate_schema(data[0])
                    }
                elif isinstance(data, dict):
                    properties = {}
                    for key, value in data.items():
                        properties[key] = generate_schema(value)
                    return {
                        "type": "object",
                        "properties": properties
                    }
                else:
                    return {"type": "string"}  # Fallback
            
            return generate_schema(sample)
            
        except Exception as e:
            logger.warning("Failed to extract schema from sample: %s", str(e))
            return None
    
    def test_endpoint(
        self, 
        endpoint: Union[str, ApiEndpoint],
        method: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Tuple[int, Optional[Any]]:
        """
        Test an API endpoint with a request.
        
        Args:
            endpoint: The endpoint URL or ApiEndpoint object
            method: HTTP method (defaults to GET or the endpoint's method)
            data: Request data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Tuple of (status_code, response_data)
        """
        if isinstance(endpoint, ApiEndpoint):
            url = endpoint.url
            method = method or endpoint.method
        else:
            url = endpoint
            method = method or "GET"
        
        logger.info("Testing API endpoint: %s %s", method, url)
        
        # Make the request
        status, content_type, json_data, text = self.make_request(
            url=url,
            method=method,
            params=params,
            data=data,
            headers=headers
        )
        
        response_data = json_data if json_data is not None else text
        
        # Log the result
        if status >= 200 and status < 300:
            logger.info("API endpoint test successful: %s %s (HTTP %d)", method, url, status)
        else:
            logger.warning("API endpoint test failed: %s %s (HTTP %d)", method, url, status)
        
        return status, response_data
    
    async def test_all_endpoints(
        self,
        concurrent: int = 5,
        include_auth: bool = False,
        timeout: int = 10
    ) -> Dict[str, Tuple[int, Optional[Any]]]:
        """
        Test all discovered API endpoints.
        
        Args:
            concurrent: Maximum number of concurrent requests
            include_auth: Whether to include endpoints that require authentication
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary mapping endpoint URLs to (status_code, response_data) tuples
        """
        endpoints_to_test = [
            endpoint for endpoint in self.endpoints 
            if not endpoint.requires_auth or include_auth
        ]
        
        results = {}
        semaphore = asyncio.Semaphore(concurrent)
        
        async def test_endpoint_async(endpoint):
            async with semaphore:
                try:
                    status, content_type, json_data, text = await self.make_async_request(
                        url=endpoint.url,
                        method=endpoint.method,
                        timeout=timeout
                    )
                    
                    response_data = json_data if json_data is not None else text
                    results[endpoint.url] = (status, response_data)
                    
                    logger.info("Tested endpoint: %s %s (HTTP %d)", 
                                endpoint.method, endpoint.url, status)
                except Exception as e:
                    logger.error("Error testing endpoint %s: %s", endpoint.url, str(e))
                    results[endpoint.url] = (0, None)
        
        tasks = [test_endpoint_async(endpoint) for endpoint in endpoints_to_test]
        await asyncio.gather(*tasks)
        
        return results
    
    def export_endpoints(self, format: str = "json") -> str:
        """
        Export discovered endpoints to the specified format.
        
        Args:
            format: Output format ("json", "yaml", or "markdown")
            
        Returns:
            String representation of the endpoints in the specified format
        """
        endpoints_data = [endpoint.model_dump() for endpoint in self.endpoints]
        
        if format.lower() == "json":
            return json.dumps(endpoints_data, indent=2)
        elif format.lower() == "yaml":
            try:
                import yaml
                return yaml.dump(endpoints_data)
            except ImportError:
                logger.warning("YAML export requested but PyYAML not installed. Falling back to JSON.")
                return json.dumps(endpoints_data, indent=2)
        elif format.lower() == "markdown":
            # Generate Markdown documentation
            md = "# API Endpoints\n\n"
            
            for endpoint in self.endpoints:
                md += f"## `{endpoint.method} {endpoint.url}`\n\n"
                
                if endpoint.description:
                    md += f"{endpoint.description}\n\n"
                
                md += f"- **Content Type**: {endpoint.content_type or 'Unknown'}\n"
                md += f"- **Requires Auth**: {'Yes' if endpoint.requires_auth else 'No'}\n"
                md += f"- **Source**: {endpoint.discovered_from or 'Unknown'}\n\n"
                
                if endpoint.parameters:
                    md += "### Parameters\n\n"
                    md += "| Name | Type | Required | Description |\n"
                    md += "|------|------|----------|-------------|\n"
                    
                    for param in endpoint.parameters:
                        name = param.get("name", "")
                        param_type = param.get("type", "")
                        required = "Yes" if param.get("required", False) else "No"
                        description = param.get("description", "")
                        md += f"| {name} | {param_type} | {required} | {description} |\n"
                    
                    md += "\n"
                
                if endpoint.response_schema:
                    md += "### Response Schema\n\n"
                    md += "```json\n"
                    md += json.dumps(endpoint.response_schema, indent=2)
                    md += "\n```\n\n"
                
                if endpoint.sample_response:
                    md += "### Sample Response\n\n"
                    md += "```json\n"
                    md += json.dumps(endpoint.sample_response, indent=2)
                    md += "\n```\n\n"
                
                md += "---\n\n"
            
            return md
        else:
            logger.warning("Unsupported export format: %s. Falling back to JSON.", format)
            return json.dumps(endpoints_data, indent=2)

