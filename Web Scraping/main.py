#!/usr/bin/env python3
"""
Unified Web Scraping Framework - Main Entry Point

This script serves as the main entry point for the unified web scraping framework,
coordinating both Python and TypeScript scrapers with a consistent output structure.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('output', 'logs', f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Ensure all required directories exist."""
    directories = [
        'output',
        'output/data',
        'output/exports',
        'output/logs',
        'output/reports',
        'output/schemas',
        'src',
        'src/python',
        'src/typescript',
        'src/shared',
        'src/config',
        'src/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def load_config(config_path: str = 'src/config/config.json') -> Dict:
    """Load configuration from the specified path."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return {
            "default_scraper": "python",
            "rate_limit": {
                "requests_per_minute": 60,
                "pause_on_429": True
            },
            "output": {
                "format": "json",
                "compression": False
            },
            "authentication": {
                "enabled": False,
                "type": "none"
            }
        }
    except json.JSONDecodeError:
        logger.error(f"Error parsing config file at {config_path}")
        sys.exit(1)

def run_python_scraper(target: str, options: Dict):
    """Run the Python scraper with the given target and options."""
    logger.info(f"Starting Python scraper for target: {target}")
    
    try:
        from src.python.scraper import run_scraper
        result = run_scraper(target, options)
        logger.info(f"Python scraper completed for {target}")
        return result
    except ImportError:
        logger.error("Failed to import Python scraper module. Ensure it's properly installed.")
        return None
    except Exception as e:
        logger.error(f"Error running Python scraper: {str(e)}")
        return None

def run_typescript_scraper(target: str, options: Dict):
    """Run the TypeScript scraper with the given target and options."""
    logger.info(f"Starting TypeScript scraper for target: {target}")
    
    try:
        # Convert options to a JSON string to pass to the TypeScript scraper
        options_json = json.dumps(options)
        
        # Run the TypeScript scraper as a subprocess
        process = subprocess.run(
            ["node", "dist/src/typescript/scraper.js", target, options_json],
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            logger.error(f"TypeScript scraper failed: {process.stderr}")
            return None
        
        # Parse the JSON output from the TypeScript scraper
        try:
            result = json.loads(process.stdout)
            logger.info(f"TypeScript scraper completed for {target}")
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse TypeScript scraper output as JSON")
            return None
    except Exception as e:
        logger.error(f"Error running TypeScript scraper: {str(e)}")
        return None

def run_scraper(target: str, scraper_type: str = None, options: Dict = None):
    """
    Run the specified scraper type for the given target.
    
    Args:
        target: URL or API endpoint to scrape
        scraper_type: 'python', 'typescript', or None (use default from config)
        options: Additional options for the scraper
    
    Returns:
        The result from the scraper or None if failed
    """
    if options is None:
        options = {}
    
    config = load_config()
    
    if scraper_type is None:
        scraper_type = config.get('default_scraper', 'python')
    
    # Merge config with options, with options taking precedence
    merged_options = {**config, **options}
    
    if scraper_type.lower() == 'python':
        return run_python_scraper(target, merged_options)
    elif scraper_type.lower() == 'typescript':
        return run_typescript_scraper(target, merged_options)
    else:
        logger.error(f"Unknown scraper type: {scraper_type}")
        return None

def main():
    """Main entry point for the unified web scraping framework."""
    parser = argparse.ArgumentParser(description='Unified Web Scraping Framework')
    parser.add_argument('target', help='URL or API endpoint to scrape')
    parser.add_argument('--scraper', choices=['python', 'typescript'], 
                        help='Scraper type to use (default from config if not specified)')
    parser.add_argument('--config', default='src/config/config.json',
                        help='Path to configuration file')
    parser.add_argument('--output-format', choices=['json', 'csv', 'sqlite'],
                        help='Output format for scraped data')
    parser.add_argument('--rate-limit', type=int,
                        help='Maximum requests per minute')
    parser.add_argument('--auth-type', choices=['none', 'basic', 'token', 'oauth'],
                        help='Authentication type to use')
    
    args = parser.parse_args()
    
    # Ensure all required directories exist
    ensure_directories_exist()
    
    # Load config and override with command line options
    config = load_config(args.config)
    options = {}
    
    if args.output_format:
        options['output'] = {'format': args.output_format}
    
    if args.rate_limit:
        options['rate_limit'] = {'requests_per_minute': args.rate_limit}
    
    if args.auth_type:
        options['authentication'] = {'type': args.auth_type, 'enabled': args.auth_type != 'none'}
    
    # Run the selected scraper
    result = run_scraper(args.target, args.scraper, options)
    
    if result:
        logger.info("Scraping completed successfully")
    else:
        logger.error("Scraping failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

