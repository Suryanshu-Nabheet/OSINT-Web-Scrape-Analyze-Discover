from setuptools import setup, find_packages

setup(
    name="web_scraper",
    version="0.1.0",
    description="A web scraper that extracts data from websites",
    author="Suryansh Unabheet",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.3",
        "selenium>=4.0.0",
        "python-dotenv>=0.19.0",
        "pandas>=1.3.0",
        "lxml>=4.6.3",
        "tqdm>=4.62.0",
        "aiohttp>=3.8.0",
        "urllib3>=1.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "scrape=web_scraper.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

