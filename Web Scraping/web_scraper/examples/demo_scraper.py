#!/usr/bin/env python3
"""
Demo Scraper - Example usage of the web_scraper library

This script demonstrates various capabilities of the web_scraper library including:
1. Basic website scraping using Python
2. Configuration management
3. Data extraction and storage
4. Report generation
5. Error handling and rate limiting
6. Practical examples with common use cases

It also shows how to use the TypeScript implementation via comments and examples.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# This would be your actual library import
# from web_scraper import Scraper, Config, RateLimiter, Storage, Reporter
# For the demo, we'll mock these classes

class Config:
    """Configuration management for web scraper"""
    
    def __init__(self, config_file: Optional[str] = None, **kwargs):
        """
        Initialize scraper configuration
        
        Args:
            config_file: Path to JSON configuration file
            **kwargs: Override configuration options
        """
        self.config = {
            "user_agent": "WebScraper Demo/1.0",
            "timeout": 30,
            "retry_count": 3,
            "retry_delay": 2,
            "rate_limit": {
                "requests_per_minute": 10,
                "concurrent_requests": 2
            },
            "storage": {
                "type": "file",
                "path": "./data"
            },
            "proxies": [],
            "headers": {},
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                # Handle nested configs
                for section in self.config:
                    if isinstance(self.config[section], dict) and key in self.config[section]:
                        self.config[section][key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()


class RateLimiter:
    """Rate limiting for web scraper"""
    
    def __init__(self, requests_per_minute: int = 10):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        if len(self.request_timestamps) >= self.requests_per_minute:
            # Wait until we can make another request
            oldest = self.request_timestamps[0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logging.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            
            # Clean up the list again
            self.request_timestamps = self.request_timestamps[1:]
        
        # Record this request
        self.request_timestamps.append(time.time())


class Storage:
    """Data storage for web scraper"""
    
    def __init__(self, storage_type: str = "file", path: str = "./data"):
        """
        Initialize storage system
        
        Args:
            storage_type: Type of storage ('file', 'sqlite', 'mongodb')
            path: Path to storage location
        """
        self.storage_type = storage_type
        self.path = path
        
        # Create directory if it doesn't exist
        if storage_type == "file" and not os.path.exists(path):
            os.makedirs(path)
    
    def save(self, data: Dict[str, Any], filename: str) -> str:
        """
        Save data to storage
        
        Args:
            data: Data to save
            filename: Name to save under
            
        Returns:
            Path where data was saved
        """
        if self.storage_type == "file":
            filepath = os.path.join(self.path, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return filepath
        elif self.storage_type == "sqlite":
            # Mock implementation
            logging.info(f"Saving to SQLite database: {data}")
            return f"sqlite://{self.path}/{filename}"
        elif self.storage_type == "mongodb":
            # Mock implementation
            logging.info(f"Saving to MongoDB: {data}")
            return f"mongodb://{self.path}/{filename}"
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def load(self, filename: str) -> Dict[str, Any]:
        """
        Load data from storage
        
        Args:
            filename: Name of data to load
            
        Returns:
            Loaded data
        """
        if self.storage_type == "file":
            filepath = os.path.join(self.path, filename)
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            # Mock implementation for other storage types
            logging.info(f"Loading from {self.storage_type}: {filename}")
            return {"mock_data": "This is mock data"}


class Reporter:
    """Report generation for web scraper"""
    
    def __init__(self, report_format: str = "json"):
        """
        Initialize reporter
        
        Args:
            report_format: Format for reports ('json', 'csv', 'html')
        """
        self.report_format = report_format
    
    def generate(self, data: List[Dict[str, Any]], output_file: str) -> str:
        """
        Generate a report from scraped data
        
        Args:
            data: List of scraped items
            output_file: File to save report to
            
        Returns:
            Path to generated report
        """
        if self.report_format == "json":
            with open(output_file, 'w') as f:
                json.dump({
                    "report_date": datetime.now().isoformat(),
                    "item_count": len(data),
                    "items": data
                }, f, indent=2)
            return output_file
        elif self.report_format == "csv":
            # Mock CSV generation
            logging.info(f"Generating CSV report with {len(data)} items")
            with open(output_file, 'w') as f:
                f.write("id,title,url,price\n")
                for item in data:
                    f.write(f"{item.get('id', '')},{item.get('title', '')},{item.get('url', '')},{item.get('price', '')}\n")
            return output_file
        elif self.report_format == "html":
            # Mock HTML generation
            logging.info(f"Generating HTML report with {len(data)} items")
            with open(output_file, 'w') as f:
                f.write("<html><body><h1>Scraping Report</h1><table>")
                f.write("<tr><th>ID</th><th>Title</th><th>URL</th><th>Price</th></tr>")
                for item in data:
                    f.write(f"<tr><td>{item.get('id', '')}</td><td>{item.get('title', '')}</td><td>{item.get('url', '')}</td><td>{item.get('price', '')}</td></tr>")
                f.write("</table></body></html>")
            return output_file
        else:
            raise ValueError(f"Unsupported report format: {self.report_format}")


class Scraper:
    """Main web scraper class"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize web scraper
        
        Args:
            config: Scraper configuration
        """
        self.config = config or Config()
        self.rate_limiter = RateLimiter(
            self.config.get("rate_limit", {}).get("requests_per_minute", 10)
        )
        self.storage = Storage(
            self.config.get("storage", {}).get("type", "file"),
            self.config.get("storage", {}).get("path", "./data")
        )
        self.reporter = Reporter()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("web_scraper")
    
    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch content from a URL
        
        Args:
            url: URL to fetch
            
        Returns:
            Dict containing response data
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # In a real implementation, this would use requests/httpx/aiohttp
        # For the demo, we'll mock the response
        self.logger.info(f"Fetching URL: {url}")
        
        # Simulate network delay
        time.sleep(0.5)
        
        # Mock different responses based on URL
        if "product" in url:
            return {
                "status": 200,
                "url": url,
                "content": f"""
                <html>
                    <body>
                        <div class="product">
                            <h1 class="product-title">Example Product</h1>
                            <div class="product-price">$99.99</div>
                            <div class="product-description">This is an example product description.</div>
                        </div>
                    </body>
                </html>
                """
            }
        elif "category" in url:
            return {
                "status": 200,
                "url": url,
                "content": f"""
                <html>
                    <body>
                        <div class="category">
                            <h1>Category Page</h1>
                            <div class="products">
                                <div class="product">
                                    <a href="/product/1">Product 1</a>
                                    <div class="price">$19.99</div>
                                </div>
                                <div class="product">
                                    <a href="/product/2">Product 2</a>
                                    <div class="price">$29.99</div>
                                </div>
                                <div class="product">
                                    <a href="/product/3">Product 3</a>
                                    <div class="price">$39.99</div>
                                </div>
                            </div>
                        </div>
                    </body>
                </html>
                """
            }
        else:
            return {
                "status": 200,
                "url": url,
                "content": f"""
                <html>
                    <body>
                        <h1>Example Page</h1>
                        <p>This is an example page content for {url}</p>
                    </body>
                </html>
                """
            }
    
    def parse(self, response: Dict[str, Any], selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse content using provided selectors
        
        Args:
            response: Response from fetch
            selectors: Dict of element names and CSS selectors
            
        Returns:
            Dict of parsed data
        """
        # In a real implementation, this would use BeautifulSoup/lxml/parsel
        # For the demo, we'll mock the parsing based on the URL
        
        self.logger.info(f"Parsing content from: {response['url']}")
        
        # Mock parsed data based on URL
        if "product" in response["url"]:
            return {
                "id": response["url"].split("/")[-1] if "/" in response["url"] else "1",
                "title": "Example Product",
                "price": "$99.99",
                "description": "This is an example product description.",
                "url": response["url"]
            }
        elif "category" in response["url"]:
            return {
                "title": "Category Page",
                "products": [
                    {"id": "1", "title": "Product 1", "price": "$19.99", "url": "/product/1"},
                    {"id": "2", "title": "Product 2", "price": "$29.99", "url": "/product/2"},
                    {"id": "3", "title": "Product 3", "price": "$39.99", "url": "/product/3"}
                ]
            }
        else:
            return {
                "title": "Example Page",
                "content": f"This is an example page content for {response['url']}"
            }
    
    def run(self, urls: List[str], selectors: Dict[str, str], output_file: str = "results.json") -> str:
        """
        Run the scraper on a list of URLs
        
        Args:
            urls: List of URLs to scrape
            selectors: Dict of element names and CSS selectors
            output_file: File to save results to
            
        Returns:
            Path to saved results
        """
        results = []
        errors = []
        
        for url in urls:
            try:
                response = self.fetch(url)
                if response["status"] == 200:
                    data = self.parse(response, selectors)
                    results.append(data)
                else:
                    self.logger.warning(f"Failed to fetch {url}: HTTP {response['status']}")
                    errors.append({"url": url, "error": f"HTTP {response['status']}"})
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {str(e)}")
                errors.append({"url": url, "error": str(e)})
        
        # Save results
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "success_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors
        }
        
        return self.storage.save(output_data, output_file)


def example_product_scraper():
    """Example of scraping product data"""
    print("\n=== Running Product Scraper Example ===")
    
    # Initialize with custom configuration
    config = Config(
        user_agent="ProductScraper/1.0",
        timeout=20,
        rate_limit={"requests_per_minute": 5}
    )
    
    scraper = Scraper(config)
    
    # URLs to scrape
    urls = [
        "https://example.com/product/1234",
        "https://example.com/product/5678",
        "https://example.com/product/9012"
    ]
    
    # Selectors for parsing
    selectors = {
        "title": ".product-title",
        "price": ".product-price",
        "description": ".product-description"
    }
    
    # Run the scraper
    output_file = scraper.run(urls, selectors, "product_results.json")
    print(f"Results saved to: {output_file}")
    
    # Generate a report
    data = scraper.storage.load("product_results.json")
    report_file = scraper.reporter.generate(data["results"], "product_report.html")
    print(f"Report generated: {report_file}")


def example_category_scraper():
    """Example of scraping category pages"""
    print("\n=== Running Category Scraper Example ===")
    
    # Initialize with configuration from file
    # Note: In a real scenario, you would create this file
    # config = Config("./config/category_scraper.json")
    config = Config(
        user_agent="CategoryScraper/1.0",
        storage={"type": "file", "path": "./data/categories"}
    )
    
    scraper = Scraper(config)
    
    # URLs to scrape
    urls = [
        "https://example.com/category/electronics",
        "https://example.com/category/clothing"
    ]
    
    # Selectors for parsing
    selectors = {
        "products": ".product",
        "product_title": "a",
        "product_price": ".price"
    }
    
    # Run the scraper
    output_file = scraper.run(urls, selectors, "category_results.json")
    print(f"Results saved to: {output_file}")


def example_error_handling():
    """Example of error handling and retries"""
    print("\n=== Running Error Handling Example ===")
    
    config = Config(
        retry_count=3,
        retry_delay=1
    )
    
    scraper = Scraper(config)
    
    # Include some invalid URLs
    urls = [
        "https://example.com/valid-page",
        "https://example.com/not-found-page",  # This would 404 in real implementation
        "https://invalid-domain-123456789.com"  # This would fail to connect
    ]
    
    # Run the scraper
    output_file = scraper.run(urls, {}, "error_handling_results.json")
    print(f"Results with errors saved to: {output_file}")


def example_typescript_usage():
    """Example of TypeScript implementation (shown as comments)"""
    print("\n=== TypeScript Implementation Example ===")
    print("Note: This is a Python script demonstrating how the TypeScript version would be used.")
    
    typescript_example = """
    // TypeScript implementation example
    import { Scraper, Config, RateLimiter } from 'web-scraper';
    
    async function runTypescriptExample() {
      // Create configuration
      const config = new Config({
        userAgent: 'TypeScriptScraper/1.0',
        timeout: 30000,
        rateLimit: {
          requestsPerMinute: 10
        }
      });
      
      // Initialize scraper
      const scraper = new Scraper(config);
      
      // URLs to scrape
      const urls = [
        'https://example.com/product/1234',
        'https://example.com/product/5678'
      ];
      
      // Selectors for parsing
      const selectors = {
        title: '.product-title',
        price: '.product-price',
        description: '.product-description'
      };
      
      // Run the scraper
      const outputFile = await scraper.run(urls, selectors, 'ts_results.json');
      console.log(`Results saved to: ${outputFile}`);
      
      // Generate a report
      const data = await scraper.storage.load('ts_results.json');
      const reportFile = await scraper.reporter.generate(
        data.results, 
        'ts_report.html'
      );
      console.log(`Report generated: ${reportFile}`);
    }
    
    runTypescriptExample().catch(console.error);
    """
    
    print(typescript_example)


def main():
    """Main function to run examples"""
    parser = argparse.ArgumentParser(description="Web Scraper Demo")
    parser.add_argument("--example", choices=["product", "category", "error", "typescript", "all"],
                        default="all", help="Example to run")
    args = parser.parse_args()
    
    print("Web Scraper Demo")
    print("================")
    
    # Create data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
    
    if args.example == "product" or args.example == "all":
        example_product_scraper()
    
    if args.example == "category" or args.example == "all":
        example_category_scraper()
    
    if args.example == "error" or args.example == "all":
        example_error_handling()
    
    if args.example == "typescript" or args.example == "all":
        example_typescript_usage()
    
    print("\nDemo completed. Check the ./data directory for output files.")


if __name__ == "__main__":
    main()

