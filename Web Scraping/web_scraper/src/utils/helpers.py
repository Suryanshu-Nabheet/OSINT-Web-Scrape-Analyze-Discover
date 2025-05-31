#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utility functions for both Python and TypeScript scrapers.
"""

import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
    return logger


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_user_agent(user_agent_type: str = "random") -> str:
    """
    Get a user agent string based on the specified type.
    
    Args:
        user_agent_type: Type of user agent to return
        
    Returns:
        User agent string
    """
    user_agents = {
        "chrome": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "firefox": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "safari": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "edge": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
    }
    
    if user_agent_type.lower() == "random":
        return random.choice(list(user_agents.values()))
    
    return user_agents.get(user_agent_type.lower(), user_agents["chrome"])


def create_directory(directory_path: Union[str, Path]) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the created directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, file_path: Union[str, Path], pretty: bool = True) -> None:
    """
    Save data as JSON to a file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        pretty: Whether to format the JSON with indentation
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2 if pretty else None)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_filename(base_name: str, extension: str = "json", timestamp: bool = True) -> str:
    """
    Generate a filename with optional timestamp.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        timestamp: Whether to include a timestamp
        
    Returns:
        Generated filename
    """
    # Remove any invalid characters from the base name
    base_name = re.sub(r'[\\/*?:"<>|]', "_", base_name)
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp_str}.{extension}"
    
    return f"{base_name}.{extension}"


def url_to_filename(url: str) -> str:
    """
    Convert a URL to a valid filename.
    
    Args:
        url: URL to convert
        
    Returns:
        Valid filename derived from the URL
    """
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Create a base name from the domain and path
    domain = parsed_url.netloc.replace("www.", "")
    path = parsed_url.path.strip("/").replace("/", "_")
    
    # Combine and clean up
    base_name = f"{domain}_{path}" if path else domain
    base_name = re.sub(r'[\\/*?:"<>|]', "_", base_name)
    
    # Handle empty or too long filenames
    if not base_name:
        base_name = "unnamed"
    elif len(base_name) > 100:
        # If too long, use a hash of the URL
        base_name = f"{base_name[:50]}_{hashlib.md5(url.encode()).hexdigest()[:10]}"
    
    return base_name


def extract_domain(url: str) -> str:
    """
    Extract the domain from a URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        Domain name
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc


def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dictionaries(result[key], value)
        else:
            result[key] = value
            
    return result


def retry_function(
    func: callable,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Exception, ...] = (Exception,),
    on_retry: Optional[callable] = None
) -> Any:
    """
    Retry a function call with exponential backoff.
    
    Args:
        func: Function to call
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay for subsequent retries
        exceptions: Exceptions that trigger a retry
        on_retry: Optional callback function to call before each retry
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    retries = 0
    delay = retry_delay
    
    while True:
        try:
            return func()
        except exceptions as e:
            retries += 1
            if retries > max_retries:
                raise e
                
            if on_retry:
                on_retry(retries, delay, e)
                
            time.sleep(delay)
            delay *= backoff_factor


def download_file(url: str, destination: Union[str, Path], timeout: int = 30) -> Path:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        timeout: Timeout in seconds
        
    Returns:
        Path to the downloaded file
        
    Raises:
        requests.RequestException: If the download fails
    """
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    return dest_path


def is_binary_content(content: Union[str, bytes]) -> bool:
    """
    Check if content is binary.
    
    Args:
        content: Content to check
        
    Returns:
        True if content is binary, False otherwise
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
        
    # Check for null bytes and high ratio of non-printable characters
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
    return bool(content.translate(None, textchars))


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Text to extract emails from
        
    Returns:
        List of extracted email addresses
    """
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return list(set(re.findall(email_pattern, text)))


def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text.
    
    Args:
        text: Text to extract phone numbers from
        
    Returns:
        List of extracted phone numbers
    """
    # This is a simplified pattern and might need adjustment for specific formats
    phone_pattern = r'(?:(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?)(?:(?:\d{1,4})[-.\s]?){1,4}\d{1,4}'
    return list(set(re.findall(phone_pattern, text)))


def get_file_extension(url: str) -> str:
    """
    Get the file extension from a URL.
    
    Args:
        url: URL to extract extension from
        
    Returns:
        File extension or empty string if none found
    """
    parsed = urlparse(url)
    path = parsed.path
    
    # Extract the file extension
    _, ext = os.path.splitext(path)
    
    # Remove the dot and return lowercase
    return ext[1:].lower() if ext else ""


def format_filesize(size_in_bytes: int) -> str:
    """
    Format file size in a human-readable format.
    
    Args:
        size_in_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024 or unit == 'TB':
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024


def remove_html_tags(html: str) -> str:
    """
    Remove HTML tags from a string.
    
    Args:
        html: HTML string
        
    Returns:
        Text content without HTML tags
    """
    # Simple regex to remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html)
    
    # Replace multiple whitespaces with a single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to make it a valid filename.
    
    Args:
        filename: Input string
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    
    # Limit length and ensure it's not empty
    if not sanitized:
        return "unnamed"
        
    return sanitized[:100]

