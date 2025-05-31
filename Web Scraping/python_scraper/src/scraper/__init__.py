"""
Scraper module.

This module provides various scraper implementations for extracting
data from websites, APIs, and other sources.
"""

from .base import BaseScraper, ScraperResult, ScraperConfig
from .api_scraper import ApiScraper
from .schema_extractor import SchemaExtractor
from .auth_handler import AuthHandler

__all__ = [
    'BaseScraper',
    'ScraperResult',
    'ScraperConfig',
    'ApiScraper',
    'SchemaExtractor',
    'AuthHandler',
]

