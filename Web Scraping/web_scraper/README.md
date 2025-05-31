# Unified Web Scraper Framework

A powerful, unified web scraping framework that integrates both Python and TypeScript implementations to provide maximum flexibility and performance for diverse scraping requirements.

## Features

- **Dual Language Support**: Use Python, TypeScript, or both for your scraping needs
- **Unified Configuration**: Shared configuration system across both implementations
- **Robust Data Models**: Structured data schemas for consistent output
- **Comprehensive Logging**: Detailed logging for debugging and audit trails
- **Flexible Output Formats**: Export data to JSON, CSV, Excel, and databases
- **Rate Limiting & Politeness**: Built-in mechanisms to respect website resources
- **Proxy & Authentication Support**: Tools for handling access restrictions
- **Error Handling & Retries**: Resilient operation with automatic retry mechanisms
- **Report Generation**: Automatic generation of scraping session reports

## Directory Structure

```
web_scraper/
├── main.py                  # Main entry point for scraping operations
├── setup.py                 # Python package setup
├── package.json             # Node.js package configuration
├── .env                     # Environment configuration
├── scraper_output/          # Output directory
│   ├── data/                # Scraped raw data
│   ├── exports/             # Processed exports (CSV, JSON, etc.)
│   ├── logs/                # Scraping logs
│   ├── reports/             # Scraping reports
│   └── schemas/             # Generated data schemas
└── src/                     # Source code
    ├── config/              # Configuration management
    │   └── config.py        # Shared configuration
    ├── models/              # Data models
    │   └── schema.py        # Structured data schemas
    ├── python/              # Python implementation
    │   └── scraper.py       # Python scraper classes
    ├── typescript/          # TypeScript implementation
    │   ├── src/             # TypeScript source files
    │   └── dist/            # Compiled JavaScript
    └── utils/               # Shared utilities
        └── helpers.py       # Helper functions
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher
- npm 6.x or higher

### Python Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/web_scraper.git
cd web_scraper
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### TypeScript Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Build TypeScript code:
```bash
npm run build
```

## Configuration

The framework uses a combination of configuration files and environment variables:

1. `.env` file for environment-specific settings
2. Configuration objects in code for default settings

Example `.env` file:
```
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
REQUEST_TIMEOUT=30
MAX_RETRIES=3
RATE_LIMIT=1
PROXY_ENABLED=false
PROXY_URL=http://proxy-server:8080
LOG_LEVEL=INFO
```

## Usage

### Basic Usage

Run a scraping task using the main script:

```bash
python main.py --url https://example.com --scraper python
```

### Command Line Options

- `--url`: Target URL to scrape (required)
- `--scraper`: Scraper engine to use: "python", "typescript", or "both" (default: "python")
- `--output-dir`: Custom output directory (default: "scraper_output")
- `--config`: Path to custom configuration file
- `--verbose`: Enable verbose logging
- `--headless`: Run browser in headless mode (when applicable)
- `--delay`: Delay between requests in seconds (overrides config)

### Using Python API

```python
from web_scraper import WebScraper

scraper = WebScraper(
    target_url="https://example.com",
    config_path="custom_config.yaml",
    output_dir="custom_output"
)

results = scraper.run()
scraper.export_data(format="json")
report = scraper.generate_report()
```

### Using TypeScript API

```typescript
import { WebScraper } from '../dist/scraper';

const scraper = new WebScraper({
  targetUrl: "https://example.com",
  configPath: "custom_config.yaml",
  outputDir: "custom_output"
});

const results = await scraper.run();
await scraper.exportData("json");
const report = scraper.generateReport();
```

## Extending the Framework

### Creating a Custom Scraper

1. Create a new class that inherits from the base scraper:

```python
# In src/python/custom_scraper.py
from web_scraper.src.python.scraper import BaseScraper

class CustomScraper(BaseScraper):
    def __init__(self, config):
        super().__init__(config)
        
    def parse_content(self, content):
        # Custom parsing logic
        return parsed_data
```

2. Register your scraper in the main module:

```python
# In main.py
from src.python.custom_scraper import CustomScraper

# Register custom scraper
scrapers["custom"] = CustomScraper
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

