#!/usr/bin/env python3
"""
Unified Web Scraper Framework
-----------------------------
Main entry point for the web scraper framework that combines both Python and TypeScript implementations.
This script coordinates running the appropriate scrapers, manages output directories, and handles logging.
"""

import os
import sys
import argparse
import json
import logging
import datetime
import shutil
import subprocess
from typing import Dict, List, Optional, Union, Any
import uuid

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import local modules
from src.python.scraper import PythonScraper
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.models.scraper_config import ScraperConfig
from src.utils.storage import StorageManager


class WebScraperFramework:
    """Main controller class for the unified web scraper framework."""

    def __init__(self):
        """Initialize the web scraper framework."""
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        self.config = None
        self.output_dir = None
        self.logger = None
        self.ts_output = None

    def setup(self, config_path: str = None, target_website: str = None, output_dir: str = None) -> None:
        """
        Set up the scraper framework with configuration and output directories.
        
        Args:
            config_path: Path to the configuration file
            target_website: Target website to scrape
            output_dir: Base directory for output
        """
        # Load configuration
        config_loader = ConfigLoader()
        self.config = config_loader.load(config_path)
        
        # Set the target website if provided
        if target_website:
            self.config.target_website = target_website
            
        # Setup output directory
        self.output_dir = self._setup_output_directory(output_dir or 'scraper_output')
        
        # Setup logger
        log_file = os.path.join(self.output_dir, 'logs', f'scraper_{self.run_id}.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = setup_logger('web_scraper', log_file)
        
        self.logger.info(f"Starting web scraper run with ID: {self.run_id}")
        self.logger.info(f"Target website: {self.config.target_website}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _setup_output_directory(self, base_dir: str) -> str:
        """
        Set up the output directory structure for this run.
        
        Args:
            base_dir: Base directory for output
            
        Returns:
            Path to the run-specific output directory
        """
        # Create the main run directory
        website_name = self.config.target_website.replace('https://', '').replace('http://', '').split('/')[0]
        run_dir = os.path.join(base_dir, website_name, self.run_id)
        
        # Create all required subdirectories
        for subdir in ['data', 'exports', 'logs', 'reports', 'schemas']:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
            
        return run_dir

    def run_python_scraper(self) -> Dict[str, Any]:
        """
        Run the Python implementation of the scraper.
        
        Returns:
            Dict containing scraping results
        """
        self.logger.info("Starting Python scraper...")
        
        scraper_config = ScraperConfig(
            url=self.config.target_website,
            output_dir=self.output_dir,
            user_agent=self.config.user_agent,
            rate_limit=self.config.rate_limit,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            auth_config=self.config.auth_config
        )
        
        python_scraper = PythonScraper(scraper_config)
        results = python_scraper.scrape()
        
        self.logger.info(f"Python scraper completed. Scraped {len(results.get('items', []))} items.")
        return results

    def run_typescript_scraper(self) -> Dict[str, Any]:
        """
        Run the TypeScript implementation of the scraper.
        
        Returns:
            Dict containing scraping results
        """
        self.logger.info("Starting TypeScript scraper...")
        
        # Create a config file for the TS scraper
        ts_config_path = os.path.join(self.output_dir, 'ts_config.json')
        with open(ts_config_path, 'w') as f:
            json.dump({
                'url': self.config.target_website,
                'outputDir': self.output_dir,
                'userAgent': self.config.user_agent,
                'rateLimit': self.config.rate_limit,
                'maxRetries': self.config.max_retries,
                'timeout': self.config.timeout,
                'authConfig': self.config.auth_config
            }, f)
        
        # Run the TypeScript scraper via npm
        ts_output_path = os.path.join(self.output_dir, 'logs', 'ts_output.json')
        try:
            result = subprocess.run(
                ['npm', 'run', 'scrape', '--', '--config', ts_config_path, '--output', ts_output_path],
                cwd=os.path.dirname(__file__),
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(f"TypeScript scraper process completed with exit code {result.returncode}")
            
            # Load the results from the output file
            if os.path.exists(ts_output_path):
                with open(ts_output_path, 'r') as f:
                    ts_results = json.load(f)
                self.logger.info(f"TypeScript scraper completed. Scraped {len(ts_results.get('items', []))} items.")
                return ts_results
            else:
                self.logger.error("TypeScript scraper did not produce output file")
                return {'items': [], 'error': 'No output file produced'}
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"TypeScript scraper failed: {str(e)}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            return {'items': [], 'error': str(e)}

    def combine_results(self, py_results: Dict[str, Any], ts_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from Python and TypeScript scrapers.
        
        Args:
            py_results: Results from Python scraper
            ts_results: Results from TypeScript scraper
            
        Returns:
            Combined results
        """
        self.logger.info("Combining results from Python and TypeScript scrapers...")
        
        combined_items = []
        
        # Add Python items with source
        for item in py_results.get('items', []):
            item['source'] = 'python'
            combined_items.append(item)
            
        # Add TypeScript items with source
        for item in ts_results.get('items', []):
            item['source'] = 'typescript'
            combined_items.append(item)
            
        # Deduplicate items based on some identifier (assuming items have an 'id' field)
        unique_items = {}
        for item in combined_items:
            item_id = item.get('id', item.get('url', str(item)))
            if item_id not in unique_items:
                unique_items[item_id] = item
        
        combined_results = {
            'items': list(unique_items.values()),
            'metadata': {
                'run_id': self.run_id,
                'target_website': self.config.target_website,
                'timestamp': datetime.datetime.now().isoformat(),
                'python_count': len(py_results.get('items', [])),
                'typescript_count': len(ts_results.get('items', [])),
                'combined_count': len(unique_items),
            }
        }
        
        self.logger.info(f"Combined {len(combined_results['items'])} unique items from both scrapers.")
        return combined_results

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save the combined results to the output directory.
        
        Args:
            results: Combined scraping results
        """
        self.logger.info("Saving combined results...")
        
        # Save the raw data
        data_path = os.path.join(self.output_dir, 'data', 'combined_data.json')
        with open(data_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Generate a CSV export
        storage_manager = StorageManager(self.output_dir)
        csv_path = os.path.join(self.output_dir, 'exports', 'combined_data.csv')
        storage_manager.export_to_csv(results['items'], csv_path)
        
        # Generate a simple HTML report
        self._generate_report(results)
        
        self.logger.info(f"Results saved to {self.output_dir}")

    def _generate_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a simple HTML report from the results.
        
        Args:
            results: Scraping results
        """
        report_path = os.path.join(self.output_dir, 'reports', 'scraping_report.html')
        
        metadata = results['metadata']
        items = results['items']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Web Scraper Report - {metadata['target_website']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Web Scraper Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Target Website:</strong> {metadata['target_website']}</p>
                <p><strong>Run ID:</strong> {metadata['run_id']}</p>
                <p><strong>Timestamp:</strong> {metadata['timestamp']}</p>
                <p><strong>Items Scraped:</strong> {metadata['combined_count']}</p>
                <p><strong>Python Scraper Items:</strong> {metadata['python_count']}</p>
                <p><strong>TypeScript Scraper Items:</strong> {metadata['typescript_count']}</p>
            </div>
            
            <h2>Scraped Items</h2>
            <table>
                <tr>
                    <th>ID/URL</th>
                    <th>Title</th>
                    <th>Source</th>
                </tr>
        """
        
        for item in items[:50]:  # Limit to first 50 items for readability
            item_id = item.get('id', item.get('url', 'N/A'))
            title = item.get('title', 'N/A')
            source = item.get('source', 'N/A')
            
            html_content += f"""
                <tr>
                    <td>{item_id}</td>
                    <td>{title}</td>
                    <td>{source}</td>
                </tr>
            """
            
        if len(items) > 50:
            html_content += f"""
                <tr>
                    <td colspan="3">... and {len(items) - 50} more items</td>
                </tr>
            """
            
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Report generated at {report_path}")

    def run(self) -> Dict[str, Any]:
        """
        Run the entire scraper framework.
        
        Returns:
            Combined results from both scrapers
        """
        self.logger.info("Starting scraper framework run...")
        
        # Run Python scraper
        py_results = self.run_python_scraper()
        
        # Run TypeScript scraper
        ts_results = self.run_typescript_scraper()
        
        # Combine results
        combined_results = self.combine_results(py_results, ts_results)
        
        # Save results
        self.save_results(combined_results)
        
        self.logger.info("Scraper framework run completed successfully.")
        return combined_results


def main():
    """Main entry point for the scraper framework."""
    parser = argparse.ArgumentParser(description='Unified Web Scraper Framework')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--url', type=str, help='Target website URL to scrape')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--python-only', action='store_true', help='Run only the Python scraper')
    parser.add_argument('--ts-only', action='store_true', help='Run only the TypeScript scraper')
    
    args = parser.parse_args()
    
    # Initialize the framework
    framework = WebScraperFramework()
    framework.setup(config_path=args.config, target_website=args.url, output_dir=args.output)
    
    # Run the framework based on arguments
    if args.python_only:
        results = framework.run_python_scraper()
        framework.save_results({'items': results.get('items', []), 'metadata': {'run_id': framework.run_id, 'target_website': framework.config.target_website, 'timestamp': datetime.datetime.now().isoformat()}})
    elif args.ts_only:
        results = framework.run_typescript_scraper()
        framework.save_results({'items': results.get('items', []), 'metadata': {'run_id': framework.run_id, 'target_website': framework.config.target_website, 'timestamp': datetime.datetime.now().isoformat()}})
    else:
        framework.run()
    
    print(f"Scraping completed. Results saved to {framework.output_dir}")
    

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('web_scraper')

# Define project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "scraper_output"

def setup_output_directory(target_url):
    """Create output directory structure for the scraping job."""
    # Extract domain from URL for folder naming
    from urllib.parse import urlparse
    domain = urlparse(target_url).netloc
    
    # Use domain and timestamp for unique folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = OUTPUT_DIR / f"{domain}_{timestamp}"
    
    # Create all necessary subdirectories
    subdirs = ["data", "exports", "logs", "reports", "schemas"]
    for subdir in subdirs:
        os.makedirs(output_folder / subdir, exist_ok=True)
    
    logger.info(f"Created output directory: {output_folder}")
    return output_folder

def run_python_scraper(target_url, output_dir, options):
    """Run the Python scraper module."""
    logger.info(f"Starting Python scraper for {target_url}")
    
    # Import Python scraper module
    sys.path.append(str(SRC_DIR))
    try:
        from python.scraper import main as py_scraper
        result = py_scraper.run(target_url, output_dir, **options)
        logger.info("Python scraper completed successfully")
        return result
    except Exception as e:
        logger.error(f"Python scraper failed: {e}")
        return None

def run_typescript_scraper(target_url, output_dir, options):
    """Run the TypeScript scraper module."""
    logger.info(f"Starting TypeScript scraper for {target_url}")
    
    # Run TypeScript scraper via Node.js
    ts_script_path = SRC_DIR / "typescript" / "dist" / "cli.js"
    
    # Ensure TypeScript is compiled
    try:
        subprocess.run(
            ["npm", "run", "build"], 
            cwd=PROJECT_ROOT,
            check=True
        )
        
        # Run the TypeScript scraper
        cmd = [
            "node", 
            str(ts_script_path), 
            "--url", target_url, 
            "--output", str(output_dir)
        ]
        
        # Add any additional options
        for key, value in options.items():
            cmd.extend([f"--{key}", str(value)])
            
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("TypeScript scraper completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"TypeScript scraper failed: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error running TypeScript scraper: {e}")
        return None

def main():
    """Main entry point for the web scraper."""
    parser = argparse.ArgumentParser(description="Unified Web Scraper")
    parser.add_argument("url", help="Target URL to scrape")
    parser.add_argument("--engine", choices=["python", "typescript", "both"], 
                        default="both", help="Scraping engine to use")
    parser.add_argument("--depth", type=int, default=1, 
                        help="Crawling depth")
    parser.add_argument("--delay", type=float, default=1.0, 
                        help="Delay between requests in seconds")
    parser.add_argument("--user-agent", 
                        help="Custom user agent string")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = setup_output_directory(args.url)
    
    # Setup options
    options = {
        "depth": args.depth,
        "delay": args.delay
    }
    if args.user_agent:
        options["user_agent"] = args.user_agent
    
    # Configure logging to file
    file_handler = logging.FileHandler(output_dir / "logs" / "scraper.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Run the selected scraper(s)
    py_result = None
    ts_result = None
    
    if args.engine in ["python", "both"]:
        py_result = run_python_scraper(args.url, output_dir, options)
    
    if args.engine in ["typescript", "both"]:
        ts_result = run_typescript_scraper(args.url, output_dir, options)
    
    # Generate final report
    with open(output_dir / "reports" / "summary.txt", "w") as f:
        f.write(f"Web Scraping Report\n")
        f.write(f"==================\n\n")
        f.write(f"Target URL: {args.url}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Engines used: {args.engine}\n\n")
        
        if py_result:
            f.write(f"Python scraper completed successfully\n")
        
        if ts_result:
            f.write(f"TypeScript scraper completed successfully\n")
    
    logger.info(f"Scraping job completed. Results stored in {output_dir}")
    print(f"Scraping job completed. Results stored in {output_dir}")

if __name__ == "__main__":
    # Create initial output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()

