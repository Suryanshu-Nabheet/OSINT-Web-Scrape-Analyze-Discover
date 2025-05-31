# Advanced Web Scraping Tool

A comprehensive web scraping framework with support for both Python and TypeScript implementations. This tool provides capabilities for extracting data from websites, including schema information, API endpoints, forms, user data, and more.

## Project Overview

This web scraping tool is designed to provide a robust and flexible framework for extracting data from websites. It offers multiple implementation options (Python and TypeScript) with identical functionality, allowing you to choose the language that best fits your workflow.

### Key Features

- **Comprehensive Data Extraction**: Extract data from websites including HTML content, API endpoints, JSON schemas, and forms
- **Authentication Support**: Handle various authentication methods (Basic, Token, OAuth) for accessing protected resources
- **Rate Limiting**: Built-in configurable rate limiting to respect website constraints
- **Proxy Support**: Rotate through proxies to avoid IP-based rate limiting or blocking
- **Schema Discovery**: Automatically detect and document website structure and API schemas
- **Modular Architecture**: Extensible design allowing for custom scraping strategies
- **CLI Interface**: Easy-to-use command-line interface for both Python and TypeScript implementations
- **Data Export**: Export scraped data in various formats (JSON, CSV, SQLite)
- **Documentation Generation**: Generate comprehensive documentation from discovered schemas

## Getting Started

This section provides a quick guide to help you get up and running with the web scraping tool.

### Prerequisites

- **Python Implementation**: Python 3.9 or higher
- **TypeScript Implementation**: Node.js 16 or higher, npm 7 or higher

## Installation

The project provides two implementation options: Python and TypeScript. You can choose either one or both based on your preferences and requirements.

### Python Implementation

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/yourusername/web-scraper.git
   cd web-scraper
   ```

2. Navigate to the Python scraper directory:
   ```bash
   cd python_scraper
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

This will install the package in development mode along with all required dependencies specified in `requirements.txt`.

### TypeScript Implementation

1. Navigate to the TypeScript scraper directory:
   ```bash
   cd typescript_scraper
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Build the project:
   ```bash
   npm run build
   ```

4. Link the CLI tool globally (optional):
   ```bash
   npm link
   ```

## Basic Usage

Both implementations provide identical functionality through a consistent CLI interface. The examples below work for both Python and TypeScript implementations, with minor syntax differences.

### Initialize Your Project

First, initialize the scraper to create the necessary configuration and directory structure:

```bash
# Python
webscrape init --output-dir ./my_project

# TypeScript
tsscrape init --output-dir ./my_project
```

This creates a directory structure and a configuration file that you can customize.

### Scanning a Website

To discover endpoints, API schemas, and other information from a website:

```bash
# Python
webscrape scan https://example.com --depth 2

# TypeScript
tsscrape scan https://example.com --depth 2
```

This will scan the website up to a depth of 2 links and save the discovered information.

### Extracting Data

To extract data from a website using a schema:

```bash
# Python
webscrape extract https://example.com/products --schema ./my_project/schemas/example_schema.json

# TypeScript
tsscrape extract https://example.com/products --schema ./my_project/schemas/example_schema.json
```

## Common Scraping Scenarios

### Scenario 1: Scraping Product Information

Extract product details from an e-commerce website:

```bash
# 1. Scan the website to discover structure
webscrape scan https://example-store.com --extract-schema

# 2. Extract product data
webscrape extract https://example-store.com/products --format json --output ./products.json
```

### Scenario 2: Extracting API Data

Discover and extract data from a website's API:

```bash
# 1. Scan for API endpoints
webscrape scan https://api-example.com --extract-apis

# 2. Set up authentication if needed
webscrape auth https://api-example.com --method token

# 3. Extract data from discovered API
webscrape extract https://api-example.com/api/data --format json
```

### Scenario 3: Scraping with Authentication

Access protected content by setting up authentication:

```bash
# 1. Configure authentication
webscrape auth https://members-only.com --method basic --username user --password pass

# 2. Extract data from authenticated pages
webscrape extract https://members-only.com/protected-content
```

### Scenario 4: Generating API Documentation

Create documentation from API schemas:

```bash
# 1. Scan API to extract schema
webscrape scan https://api.example.com

# 2. Generate documentation
webscrape docs ./scraper_output/schemas/api_example_com_schema.json --format markdown --output ./api_docs.md
```

## Using the Scraper Programmatically

### Python Example

```python
from scraper import ApiScraper, ScraperConfig

# Configure the scraper
config = ScraperConfig(
    max_depth=2,
    request_delay=1.0,
    respect_robots_txt=True
)

# Create scraper instance
scraper = ApiScraper(config=config)

# Scrape API endpoints
result = scraper.scrape("https://api.example.com")

# Process results
for endpoint in result.data["endpoints"]:
    print(f"Discovered API: {endpoint['method']} {endpoint['url']}")
```

### TypeScript Example

```typescript
import { ApiScraper, ScraperConfig } from '../src/scraper';

// Configure the scraper
const config = {
  maxDepth: 2,
  requestDelay: 1.0,
  respectRobotsTxt: true
};

// Create scraper instance
const scraper = new ApiScraper(config);

// Scrape API endpoints
async function scrapeApi() {
  const result = await scraper.scrape("https://api.example.com");
  
  // Process results
  result.data.endpoints.forEach(endpoint => {
    console.log(`Discovered API: ${endpoint.method} ${endpoint.url}`);
  });
}

scrapeApi();
```

## Configuration Examples

The scraper can be configured through a JSON configuration file, which is created during initialization. You can customize various aspects of the scraper's behavior:

### Basic Configuration

```json
{
  "version": "0.1.0",
  "settings": {
    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "requestTimeout": 30,
    "requestDelay": 1.0,
    "maxRetries": 3,
    "followRedirects": true,
    "respectRobotsTxt": true,
    "maxDepth": 3
  },
  "storage": {
    "type": "file",
    "format": "json",
    "path": "./scraper_output/data"
  }
}
```

### Configuration with Authentication

```json
{
  "version": "0.1.0",
  "settings": {
    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "requestTimeout": 30,
    "requestDelay": 1.0
  },
  "authentication": {
    "method": "basic",
    "credentials": {
      "example.com": {
        "username": "user",
        "password": "pass"
      }
    }
  },
  "storage": {
    "type": "file",
    "format": "json",
    "path": "./scraper_output/data"
  }
}
```

### Configuration with Proxy

```json
{
  "version": "0.1.0",
  "settings": {
    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "requestTimeout": 30
  },
  "proxy": {
    "enabled": true,
    "rotationEnabled": true,
    "rotationInterval": 10,
    "proxies": [
      {"http": "http://proxy1.example.com:8080"},
      {"http": "http://proxy2.example.com:8080"}
    ]
  }
}
```

### Configuration with Database Storage

```json
{
  "version": "0.1.0",
  "settings": {
    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  },
  "storage": {
    "type": "database",
    "format": "sqlite",
    "path": "./scraper_output/data/scraper.db",
    "connection": {
      "url": "sqlite:///scraper_output/data/scraper.db"
    }
  }
}
```

## Security and Ethical Considerations

When using this web scraping tool, please consider the following ethical and legal guidelines:

### Legal Compliance

1. **Terms of Service**: Always review and comply with the website's Terms of Service before scraping.
2. **Copyright Laws**: Respect copyright protections and intellectual property rights.
3. **Data Protection Laws**: Comply with GDPR, CCPA, and other data protection regulations when collecting personal data.

### Technical Best Practices

1. **Rate Limiting**: Use the built-in rate limiting features to avoid overwhelming websites with requests:
   ```json
   {
     "rate_limit": {
       "enabled": true,
       "requestsPerSecond": 1.0,
       "maxConcurrent": 3
     }
   }
   ```

2. **User-Agent Identification**: Properly identify your scraper with an informative user agent:
   ```json
   {
     "settings": {
       "userAgent": "YourCompany Bot/1.0 (https://example.com/bot; bot@example.com)"
     }
   }
   ```

3. **Respect robots.txt**: Keep the `respectRobotsTxt` setting enabled:
   ```json
   {
     "settings": {
       "respectRobotsTxt": true
     }
   }
   ```

### Ethical Guidelines

1. **Minimize Impact**: Configure appropriate delays between requests to reduce server load
2. **Data Privacy**: Avoid scraping personal or sensitive information
3. **Transparency**: Identify your scraper through a proper user agent string
4. **Public Data Only**: Focus on publicly available data rather than attempting to access restricted areas

## Advanced Usage

Both implementations provide identical functionality through a consistent CLI interface. The examples below work for both Python and TypeScript implementations, with minor syntax differences.

### Python CLI

```bash
# Python command format
webscrape [command] [options]
```

### TypeScript CLI

```bash
# TypeScript command format
tsscrape [command] [options]
```

### Common Commands

#### Initialize Scraper

Set up the necessary configuration and directory structure:

```bash
# Python
webscrape init --output-dir ./my_scraper

# TypeScript
tsscrape init --output-dir ./my_scraper
```

#### Scan Website

Discover endpoints, APIs, and schema information:

```bash
# Python
webscrape scan https://example.com --depth 3 --output ./output/example_schema.json

# TypeScript
tsscrape scan https://example.com --depth 3 --output ./output/example_schema.json
```

#### Extract Data

Extract structured data from a website based on a schema:

```bash
# Python
webscrape extract https://example.com --schema ./schemas/example_schema.json --format json

# TypeScript
tsscrape extract https://example.com --schema ./schemas/example_schema.json --format json
```

#### Configure Authentication

Set up authentication credentials for accessing protected resources:

```bash
# Python
webscrape auth https://example.com --method basic --username user --password pass

# TypeScript
tsscrape auth https://example.com --method basic --username user --password pass
```

#### Export Data

Convert scraped data between different formats:

```bash
# Python
webscrape export ./data/example_data.json --format csv --output ./exports/example_data.csv

# TypeScript
tsscrape export ./data/example_data.json --format csv --output ./exports/example_data.csv
```

#### Generate Documentation

Create human-readable documentation from schema information:

```bash
# Python
webscrape docs ./schemas/example_schema.json --format markdown --output ./docs/api_docs.md

# TypeScript
tsscrape docs ./schemas/example_schema.json --format markdown --output ./docs/api_docs.md
```

## Configuration Details

### Configuration Options

- **settings**: General scraper settings
  - **userAgent**: User agent string to use for requests
  - **requestTimeout**: Timeout for HTTP requests in seconds
  - **requestDelay**: Delay between requests in seconds
  - **maxRetries**: Maximum number of retry attempts for failed requests
  - **followRedirects**: Whether to follow HTTP redirects
  - **respectRobotsTxt**: Whether to respect robots.txt rules
  - **maxDepth**: Maximum depth for crawling links

- **authentication**: Authentication configuration
  - **method**: Authentication method (basic, token, oauth)
  - **credentials**: Domain-specific credentials

- **proxy**: Proxy configuration
  - **enabled**: Whether to use proxies
  - **rotationEnabled**: Whether to rotate proxies
  - **rotationInterval**: Number of requests before rotating proxies
  - **providers**: List of proxy providers or URLs
  - **proxies**: List of proxy configurations

- **storage**: Data storage configuration
  - **type**: Storage type (file, database)
  - **format**: Storage format (json, csv, sqlite)
  - **path**: Path to store scraped data

## Directory Structure

```
Web Scraping/
├── python_scraper/           # Python implementation
│   ├── src/                  # Source code
│   │   ├── cli.py            # CLI entry point
│   │   ├── scraper/          # Scraper modules
│   │   │   ├── base.py       # Base scraper class
│   │   │   ├── api_scraper.py # API scraping functionality
│   │   │   ├── schema_extractor.py # Schema extraction
│   │   │   └── auth_handler.py # Authentication handling
│   │   └── utils/            # Utility modules
│   │       ├── rate_limiter.py # Rate limiting
│   │       └── data_storage.py # Data storage
│   ├── requirements.txt      # Python dependencies
│   └── setup.py              # Package configuration
├── typescript_scraper/       # TypeScript implementation
│   ├── src/                  # Source code
│   │   ├── cli.ts            # CLI entry point
│   │   ├── scraper/          # Scraper modules
│   │   │   ├── base.ts       # Base scraper class
│   │   │   ├── apiScraper.ts # API scraping functionality
│   │   │   ├── schemaExtractor.ts # Schema extraction
│   │   │   └── authHandler.ts # Authentication handling
│   │   └── utils/            # Utility modules
│   │       ├── rateLimiter.ts # Rate limiting
│   │       └── dataStorage.ts # Data storage
│   ├── package.json          # Node.js dependencies
│   └── tsconfig.json         # TypeScript configuration
└── README.md                 # This file
```

## Development Setup

### Prerequisites

- Python 3.9+ (for Python implementation)
- Node.js 16+ (for TypeScript implementation)
- Git

### Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/web-scraper.git
   cd web-scraper
   ```

2. Set up Python development environment:
   ```bash
   cd python_scraper
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Set up TypeScript development environment:
   ```bash
   cd typescript_scraper
   npm install
   npm run watch  # Watches for file changes and rebuilds
   ```

### Running Tests

#### Python Tests
```bash
cd python_scraper
python -m pytest
```

#### TypeScript Tests
```bash
cd typescript_scraper
npm test
```

## Extending the Scraper

The scraper is designed to be extensible. You can create custom scrapers by extending the base classes:

### Python Extension

```python
from scraper.base import BaseScraper, ScraperResult

class CustomScraper(BaseScraper):
    def scrape(self, url, **kwargs):
        # Implement custom scraping logic
        status, soup = self.fetch_html(url)
        
        # Extract data
        data = {
            "title": soup.title.text if soup.title else "",
            "headings": [h.text for h in soup.find_all("h1")]
        }
        
        # Return result
        return ScraperResult(
            url=url,
            status_code=status,
            content_type="text/html",
            data=data,
            metadata={"custom": "value"}
        )
```

### TypeScript Extension

```typescript
import { BaseScraper, ScraperResult } from '../scraper/base';

class CustomScraper extends BaseScraper {
  async scrape(url: string, options?: any): Promise<ScraperResult> {
    // Implement custom scraping logic
    const { status, $ } = await this.fetchHtml(url);
    
    // Extract data
    const data = {
      title: $('title').text(),
      headings: $('h1').map((i, el) => $(el).text()).get()
    };
    
    // Return result
    return {
      url,
      statusCode: status,
      contentType: 'text/html',
      data,
      timestamp: new Date(),
      metadata: { custom: 'value' }
    };
  }
}
```

## Legal and Ethical Considerations

This tool is provided for educational and legitimate research purposes only. When using this tool, please:

1. Always respect website terms of service
2. Consider the load your scraping places on websites
3. Respect robots.txt directives when configured to do so
4. Obtain proper authorization before scraping non-public data
5. Be aware of and comply with relevant laws and regulations
6. Handle personal data in accordance with privacy laws

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

