#!/usr/bin/env python3
"""
Integration Example: Combining Python and TypeScript Scrapers

This example demonstrates how to use both Python and TypeScript scrapers together in a
unified workflow. It covers:

1. Parallel execution of multiple scrapers
2. Data validation and normalization
3. Result merging from different sources
4. Error handling and retry logic
5. A practical e-commerce scraping scenario

Requirements:
- web_scraper package installed
- TypeScript scrapers compiled and available
- Node.js runtime for TypeScript scrapers
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import web_scraper modules
from web_scraper.python_scraper import PythonScraper
from web_scraper.ts_bridge import TypeScriptScraperBridge
from web_scraper.validators import validate_product_data
from web_scraper.utils.logging import setup_logger
from web_scraper.utils.rate_limiter import RateLimiter
from web_scraper.exceptions import ScraperError, ValidationError

# Set up logging
logger = setup_logger("integration_example", level=logging.INFO)

# Configuration
TS_SCRAPER_PATH = Path("../../dist/ts_scrapers")
MAX_RETRIES = 3
TIMEOUT = 30  # seconds
MAX_CONCURRENT = 5


class IntegratedScraper:
    """
    A class that combines Python and TypeScript scrapers into a unified workflow.
    """
    
    def __init__(self, python_scraper_config: Dict[str, Any], ts_scraper_paths: List[str]):
        """
        Initialize the integrated scraper with configurations for both Python and TypeScript scrapers.
        
        Args:
            python_scraper_config: Configuration for Python scrapers
            ts_scraper_paths: Paths to TypeScript scraper scripts
        """
        self.python_scraper = PythonScraper(**python_scraper_config)
        self.ts_bridges = [
            TypeScriptScraperBridge(path=TS_SCRAPER_PATH / path) 
            for path in ts_scraper_paths
        ]
        self.rate_limiter = RateLimiter(requests_per_minute=60)
        
    async def scrape_with_python(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Run Python scraper on a list of URLs with rate limiting."""
        results = []
        
        for url in urls:
            await self.rate_limiter.acquire()
            try:
                data = await self.python_scraper.scrape(url, timeout=TIMEOUT)
                if data:
                    try:
                        validate_product_data(data)
                        results.append(data)
                    except ValidationError as e:
                        logger.warning(f"Validation error for {url}: {e}")
            except ScraperError as e:
                logger.error(f"Python scraper error for {url}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error in Python scraper for {url}")
                
        return results
    
    async def scrape_with_typescript(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Run TypeScript scrapers on a list of URLs with load balancing."""
        results = []
        tasks = []
        
        # Distribute URLs across available TypeScript scrapers
        for i, url in enumerate(urls):
            bridge = self.ts_bridges[i % len(self.ts_bridges)]
            tasks.append(self._scrape_ts_with_retry(bridge, url))
            
        # Gather results from all tasks
        ts_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, filtering out exceptions
        for result in ts_results:
            if isinstance(result, Exception):
                logger.error(f"TypeScript scraper error: {result}")
            elif result:
                try:
                    validate_product_data(result)
                    results.append(result)
                except ValidationError as e:
                    logger.warning(f"Validation error in TypeScript result: {e}")
                    
        return results
    
    async def _scrape_ts_with_retry(
        self, bridge: TypeScriptScraperBridge, url: str
    ) -> Optional[Dict[str, Any]]:
        """Execute TypeScript scraper with retry logic."""
        for attempt in range(MAX_RETRIES):
            await self.rate_limiter.acquire()
            try:
                return await bridge.execute({"url": url, "timeout": TIMEOUT})
            except Exception as e:
                logger.warning(f"TypeScript scraper attempt {attempt+1} failed for {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        return None
    
    async def scrape_parallel(
        self, python_urls: List[str], typescript_urls: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute Python and TypeScript scrapers in parallel and return combined results.
        
        Args:
            python_urls: URLs to scrape with Python
            typescript_urls: URLs to scrape with TypeScript
            
        Returns:
            Dict with results from both types of scrapers
        """
        # Run both scrapers in parallel
        python_task = asyncio.create_task(self.scrape_with_python(python_urls))
        typescript_task = asyncio.create_task(self.scrape_with_typescript(typescript_urls))
        
        # Wait for both to complete
        python_results, typescript_results = await asyncio.gather(
            python_task, typescript_task, return_exceptions=True
        )
        
        # Handle any exceptions
        if isinstance(python_results, Exception):
            logger.error(f"Python scraping failed: {python_results}")
            python_results = []
            
        if isinstance(typescript_results, Exception):
            logger.error(f"TypeScript scraping failed: {typescript_results}")
            typescript_results = []
            
        return {
            "python_results": python_results,
            "typescript_results": typescript_results,
            "timestamp": datetime.now().isoformat(),
            "total_results": len(python_results) + len(typescript_results)
        }
    
    def merge_results(self, results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Merge and normalize results from different scrapers.
        
        Args:
            results: Dict containing results from different scraper types
            
        Returns:
            A unified list of normalized results
        """
        merged = []
        
        # Process Python results
        for item in results.get("python_results", []):
            normalized = self._normalize_item(item, source="python")
            merged.append(normalized)
            
        # Process TypeScript results
        for item in results.get("typescript_results", []):
            normalized = self._normalize_item(item, source="typescript")
            merged.append(normalized)
            
        # Sort by price (as an example)
        merged.sort(key=lambda x: float(x.get("price", 0)))
        
        return merged
    
    def _normalize_item(self, item: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Normalize data format from different scrapers into a consistent structure."""
        normalized = {
            "source": source,
            "scraped_at": datetime.now().isoformat()
        }
        
        # Map common fields
        field_mapping = {
            "title": ["title", "name", "product_name"],
            "price": ["price", "product_price", "cost"],
            "currency": ["currency", "currency_code"],
            "description": ["description", "product_description", "desc"],
            "image_url": ["image_url", "image", "img", "product_image"],
            "url": ["url", "product_url", "link"],
            "sku": ["sku", "product_id", "id"],
            "availability": ["availability", "in_stock", "available"]
        }
        
        # Apply field mapping
        for target_field, source_fields in field_mapping.items():
            for field in source_fields:
                if field in item:
                    normalized[target_field] = item[field]
                    break
            
        # Include any additional fields
        for key, value in item.items():
            if key not in normalized:
                normalized[key] = value
                
        return normalized


async def e_commerce_example():
    """
    A practical example demonstrating e-commerce product scraping across multiple sites.
    """
    # Example configuration
    python_scraper_config = {
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
        "use_proxy": False,
        "parse_javascript": True
    }
    
    ts_scraper_paths = [
        "amazon_scraper.js",
        "walmart_scraper.js",
        "bestbuy_scraper.js"
    ]
    
    # Example URLs (these would be real e-commerce URLs in practice)
    python_urls = [
        "https://example.com/product/1",
        "https://example.com/product/2",
        "https://example.com/product/3",
    ]
    
    typescript_urls = [
        "https://example-ts.com/product/1",
        "https://example-ts.com/product/2",
    ]
    
    # Initialize the integrated scraper
    integrated_scraper = IntegratedScraper(
        python_scraper_config=python_scraper_config,
        ts_scraper_paths=ts_scraper_paths
    )
    
    # Execute scrapers in parallel
    logger.info("Starting parallel scraping...")
    results = await integrated_scraper.scrape_parallel(
        python_urls=python_urls,
        typescript_urls=typescript_urls
    )
    
    # Merge and normalize results
    logger.info(f"Scraped {results['total_results']} products. Merging results...")
    merged_results = integrated_scraper.merge_results(results)
    
    # Output the results
    logger.info(f"Successfully processed {len(merged_results)} products")
    print(json.dumps(merged_results, indent=2))
    
    # Save results to file
    output_path = Path("./scraping_results.json")
    with open(output_path, "w") as f:
        json.dump(merged_results, indent=2, fp=f)
    logger.info(f"Results saved to {output_path.absolute()}")


def main():
    """Entry point for the integration example."""
    logger.info("Starting integration example")
    
    try:
        # Run the async example
        asyncio.run(e_commerce_example())
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.exception(f"Error in integration example: {e}")
    
    logger.info("Integration example completed")


if __name__ == "__main__":
    main()

