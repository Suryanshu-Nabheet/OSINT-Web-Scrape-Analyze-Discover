#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared configuration for both Python and TypeScript scrapers.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class LogLevel(Enum):
    """Enum representing different logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StorageFormat(Enum):
    """Enum representing different storage formats."""
    JSON = "json"
    CSV = "csv"
    SQLITE = "sqlite"
    PARQUET = "parquet"


class UserAgentType(Enum):
    """Enum representing different user agent types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    RANDOM = "random"
    CUSTOM = "custom"


@dataclass
class ScraperConfig:
    """Configuration class for the web scraper."""
    
    # Basic settings
    name: str = "web_scraper"
    version: str = "1.0.0"
    log_level: LogLevel = LogLevel.INFO
    
    # Request settings
    requests_per_minute: int = 30
    timeout: int = 30
    retries: int = 3
    backoff_factor: float = 1.5
    user_agent: UserAgentType = UserAgentType.RANDOM
    custom_user_agent: Optional[str] = None
    
    # Proxy settings
    proxy: Optional[str] = None
    proxy_rotation: bool = False
    proxy_list: List[str] = field(default_factory=list)
    
    # Authentication settings
    auth_enabled: bool = False
    auth_type: Optional[str] = None
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    auth_token: Optional[str] = None
    
    # Storage settings
    storage_format: StorageFormat = StorageFormat.JSON
    output_dir: str = "output"
    
    # Behavior settings
    follow_redirects: bool = True
    respect_robots_txt: bool = True
    verify_ssl: bool = True
    extract_metadata: bool = True
    javascript_enabled: bool = False
    
    # Advanced settings
    concurrent_requests: int = 1
    depth_limit: int = 3
    same_domain_only: bool = True
    url_patterns: List[str] = field(default_factory=list)
    url_exclusions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ScraperConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            ScraperConfig instance
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the configuration file contains invalid JSON
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            config_data = json.load(f)
            
        # Convert enum string values to enum objects
        if 'log_level' in config_data:
            config_data['log_level'] = LogLevel[config_data['log_level'].upper()]
            
        if 'storage_format' in config_data:
            config_data['storage_format'] = StorageFormat[config_data['storage_format'].upper()]
            
        if 'user_agent' in config_data:
            config_data['user_agent'] = UserAgentType[config_data['user_agent'].upper()]
            
        return cls(**config_data)
    
    def to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration file
            
        Raises:
            PermissionError: If the file cannot be written due to permissions
        """
        file_path = Path(file_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert enum objects to string values
        config_data = {
            "name": self.name,
            "version": self.version,
            "log_level": self.log_level.name,
            "requests_per_minute": self.requests_per_minute,
            "timeout": self.timeout,
            "retries": self.retries,
            "backoff_factor": self.backoff_factor,
            "user_agent": self.user_agent.name,
            "custom_user_agent": self.custom_user_agent,
            "proxy": self.proxy,
            "proxy_rotation": self.proxy_rotation,
            "proxy_list": self.proxy_list,
            "auth_enabled": self.auth_enabled,
            "auth_type": self.auth_type,
            "auth_username": self.auth_username,
            "auth_password": self.auth_password,
            "auth_token": self.auth_token,
            "storage_format": self.storage_format.name,
            "output_dir": self.output_dir,
            "follow_redirects": self.follow_redirects,
            "respect_robots_txt": self.respect_robots_txt,
            "verify_ssl": self.verify_ssl,
            "extract_metadata": self.extract_metadata,
            "javascript_enabled": self.javascript_enabled,
            "concurrent_requests": self.concurrent_requests,
            "depth_limit": self.depth_limit,
            "same_domain_only": self.same_domain_only,
            "url_patterns": self.url_patterns,
            "url_exclusions": self.url_exclusions
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "name": self.name,
            "version": self.version,
            "log_level": self.log_level.name,
            "requests_per_minute": self.requests_per_minute,
            "timeout": self.timeout,
            "retries": self.retries,
            "backoff_factor": self.backoff_factor,
            "user_agent": self.user_agent.name,
            "custom_user_agent": self.custom_user_agent,
            "proxy": self.proxy,
            "proxy_rotation": self.proxy_rotation,
            "proxy_list": self.proxy_list,
            "auth_enabled": self.auth_enabled,
            "auth_type": self.auth_type,
            "auth_username": self.auth_username,
            "auth_password": None if self.auth_password else None,  # Don't include actual password
            "auth_token": None if self.auth_token else None,  # Don't include actual token
            "storage_format": self.storage_format.name,
            "output_dir": self.output_dir,
            "follow_redirects": self.follow_redirects,
            "respect_robots_txt": self.respect_robots_txt,
            "verify_ssl": self.verify_ssl,
            "extract_metadata": self.extract_metadata,
            "javascript_enabled": self.javascript_enabled,
            "concurrent_requests": self.concurrent_requests,
            "depth_limit": self.depth_limit,
            "same_domain_only": self.same_domain_only,
            "url_patterns": self.url_patterns,
            "url_exclusions": self.url_exclusions
        }


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # Start from the current file
    current_file = Path(__file__).resolve()
    
    # Go up to find the project root (where src directory is)
    for parent in current_file.parents:
        if (parent / "src").exists():
            return parent
            
    # Fallback to two directories up from current file
    return current_file.parent.parent.parent


def get_default_config_path() -> Path:
    """
    Get the default configuration file path.
    
    Returns:
        Path to the default configuration file
    """
    return get_project_root() / "config" / "scraper_config.json"


def load_config(config_path: Optional[Union[str, Path]] = None) -> ScraperConfig:
    """
    Load configuration from a file or create a default configuration.
    
    Args:
        config_path: Path to the configuration file (optional)
        
    Returns:
        ScraperConfig instance
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    try:
        return ScraperConfig.from_file(config_path)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default configuration if file doesn't exist or is invalid
        return ScraperConfig()

