"""
Data storage utilities for the web scraper framework.

This module provides classes and utilities for storing, retrieving, and validating
scraped data using various backends like files (JSON, CSV) and SQLite.
"""

import abc
import csv
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union, Generic, Callable

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Define a fallback BaseModel if pydantic is not available
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Type variables for generics
T = TypeVar('T')  # For data items
K = TypeVar('K')  # For keys


class StorageFormat(str, Enum):
    """Supported storage formats."""
    JSON = "json"
    CSV = "csv"
    SQLITE = "sqlite"
    MEMORY = "memory"


class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass


class ValidationError(StorageError):
    """Exception raised when data validation fails."""
    pass


class DataNotFoundError(StorageError):
    """Exception raised when requested data is not found."""
    pass


class StorageConnectionError(StorageError):
    """Exception raised when connection to storage fails."""
    pass


if PYDANTIC_AVAILABLE:
    class StorageConfig(BaseModel):
        """Configuration for data storage.
        
        Attributes:
            format: Storage format to use (json, csv, sqlite, memory)
            path: Path to the storage file or directory
            table_name: Table name for SQLite storage
            schema: Optional schema for data validation
            batch_size: Number of items to process in a batch operation
            max_workers: Maximum number of workers for parallel operations
            auto_commit: Whether to automatically commit changes (for SQLite)
            encoding: Character encoding for file operations
            create_if_missing: Create storage file/directory if it doesn't exist
        """
        format: StorageFormat = Field(default=StorageFormat.JSON, description="Storage format to use")
        path: Optional[str] = Field(default=None, description="Path to storage file or directory")
        table_name: str = Field(default="scraped_data", description="Table name for SQLite storage")
        schema: Optional[Dict[str, Any]] = Field(default=None, description="Optional schema for data validation")
        batch_size: int = Field(default=100, description="Number of items to process in a batch")
        max_workers: int = Field(default=4, description="Maximum number of workers for parallel operations")
        auto_commit: bool = Field(default=True, description="Whether to automatically commit changes")
        encoding: str = Field(default="utf-8", description="Character encoding for file operations")
        create_if_missing: bool = Field(default=True, description="Create storage file/directory if missing")
        
        @validator('path')
        def validate_path(cls, v, values):
            if v is None and values.get('format') != StorageFormat.MEMORY:
                raise ValueError("Path must be provided for non-memory storage")
            return v
else:
    # Simple fallback implementation if Pydantic is not available
    class StorageConfig:
        def __init__(
            self,
            format: str = "json",
            path: Optional[str] = None,
            table_name: str = "scraped_data",
            schema: Optional[Dict[str, Any]] = None,
            batch_size: int = 100,
            max_workers: int = 4,
            auto_commit: bool = True,
            encoding: str = "utf-8",
            create_if_missing: bool = True,
        ):
            self.format = format
            self.path = path
            self.table_name = table_name
            self.schema = schema
            self.batch_size = batch_size
            self.max_workers = max_workers
            self.auto_commit = auto_commit
            self.encoding = encoding
            self.create_if_missing = create_if_missing
            
            if self.path is None and self.format != "memory":
                raise ValueError("Path must be provided for non-memory storage")


class DataValidator:
    """Validates data against a schema.
    
    This class provides methods to validate data against a schema definition,
    ensuring data integrity before storage.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """Initialize the validator with an optional schema.
        
        Args:
            schema: Dictionary defining the expected data structure and types
        """
        self.schema = schema
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a single data item against the schema.
        
        Args:
            data: Data item to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.schema:
            return True, None
            
        for field, field_schema in self.schema.items():
            # Check required fields
            if field_schema.get("required", False) and field not in data:
                return False, f"Required field '{field}' is missing"
                
            # Skip validation if field is not present and not required
            if field not in data:
                continue
                
            value = data[field]
            field_type = field_schema.get("type")
            
            # Type validation
            if field_type and not self._validate_type(value, field_type):
                return False, f"Field '{field}' has invalid type. Expected {field_type}"
                
            # Range validation for numeric fields
            if field_type in ["integer", "number"]:
                min_value = field_schema.get("minimum")
                max_value = field_schema.get("maximum")
                
                if min_value is not None and value < min_value:
                    return False, f"Field '{field}' is less than minimum value {min_value}"
                    
                if max_value is not None and value > max_value:
                    return False, f"Field '{field}' exceeds maximum value {max_value}"
            
            # String length validation
            if field_type == "string":
                min_length = field_schema.get("minLength")
                max_length = field_schema.get("maxLength")
                pattern = field_schema.get("pattern")
                
                if min_length is not None and len(value) < min_length:
                    return False, f"Field '{field}' is shorter than minimum length {min_length}"
                    
                if max_length is not None and len(value) > max_length:
                    return False, f"Field '{field}' exceeds maximum length {max_length}"
                    
                if pattern is not None:
                    import re
                    if not re.match(pattern, value):
                        return False, f"Field '{field}' does not match required pattern"
            
            # Enum validation
            enum_values = field_schema.get("enum")
            if enum_values is not None and value not in enum_values:
                return False, f"Field '{field}' must be one of {enum_values}"
        
        return True, None
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate that a value matches the expected type.
        
        Args:
            value: The value to validate
            expected_type: Type name (string, integer, number, boolean, array, object)
            
        Returns:
            True if the value matches the expected type, False otherwise
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, (list, tuple))
        elif expected_type == "object":
            return isinstance(value, dict)
        return True  # Unknown type, assume valid


class DataStorage(Generic[T], abc.ABC):
    """Abstract base class for data storage implementations.
    
    This class defines the interface that all storage backends must implement.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize the storage with configuration.
        
        Args:
            config: Configuration for the storage backend
        """
        self.config = config
        self.validator = DataValidator(config.schema)
        self._lock = threading.RLock()
    
    @abc.abstractmethod
    def save(self, data: T) -> bool:
        """Save a single data item.
        
        Args:
            data: The data to save
            
        Returns:
            True if saved successfully, False otherwise
            
        Raises:
            ValidationError: If data fails validation
            StorageError: If storage operation fails
        """
        pass
    
    @abc.abstractmethod
    def save_many(self, items: List[T]) -> int:
        """Save multiple data items.
        
        Args:
            items: List of data items to save
            
        Returns:
            Number of items successfully saved
            
        Raises:
            ValidationError: If any data fails validation
            StorageError: If storage operation fails
        """
        pass
    
    @abc.abstractmethod
    def get(self, key: K) -> T:
        """Retrieve a data item by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The retrieved data item
            
        Raises:
            DataNotFoundError: If the data is not found
            StorageError: If retrieval operation fails
        """
        pass
    
    @abc.abstractmethod
    def get_many(self, keys: List[K]) -> List[T]:
        """Retrieve multiple data items by their keys.
        
        Args:
            keys: List of keys to look up
            
        Returns:
            List of retrieved data items
            
        Raises:
            DataNotFoundError: If any key is not found
            StorageError: If retrieval operation fails
        """
        pass
    
    @abc.abstractmethod
    def query(self, filter_func: Callable[[T], bool]) -> List[T]:
        """Query data items using a filter function.
        
        Args:
            filter_func: Function that takes a data item and returns True if it should be included
            
        Returns:
            List of data items matching the filter
            
        Raises:
            StorageError: If query operation fails
        """
        pass
    
    @abc.abstractmethod
    def update(self, key: K, data: T) -> bool:
        """Update a data item by key.
        
        Args:
            key: The key of the item to update
            data: The new data
            
        Returns:
            True if updated successfully, False otherwise
            
        Raises:
            DataNotFoundError: If the item to update is not found
            ValidationError: If the new data fails validation
            StorageError: If update operation fails
        """
        pass
    
    @abc.abstractmethod
    def delete(self, key: K) -> bool:
        """Delete a data item by key.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            True if deleted successfully, False otherwise
            
        Raises:
            DataNotFoundError: If the item to delete is not found
            StorageError: If delete operation fails
        """
        pass
    
    @abc.abstractmethod
    def clear(self) -> bool:
        """Clear all data from storage.
        
        Returns:
            True if cleared successfully, False otherwise
            
        Raises:
            StorageError: If clear operation fails
        """
        pass
    
    @abc.abstractmethod
    def count(self) -> int:
        """Count the number of items in storage.
        
        Returns:
            Number of items
            
        Raises:
            StorageError: If count operation fails
        """
        pass
    
    def export_data(self, format: StorageFormat, path: str) -> bool:
        """Export all data to a file in the specified format.
        
        Args:
            format: The format to export to
            path: Path to save the exported data
            
        Returns:
            True if exported successfully, False otherwise
            
        Raises:
            StorageError: If export operation fails
        """
        try:
            data = self.query(lambda _: True)
            
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            if format == StorageFormat.JSON:
                with open(path, 'w', encoding=self.config.encoding) as f:
                    json.dump(data, f, indent=2)
            elif format == StorageFormat.CSV:
                if not data:
                    with open(path, 'w', encoding=self.config.encoding) as f:
                        f.write("")
                    return True
                    
                # Get field names from the first item
                fieldnames = list(data[0].keys()) if data else []
                
                with open(path, 'w', newline='', encoding=self.config.encoding) as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
            else:
                raise StorageError(f"Unsupported export format: {format}")
                
            return True
        except Exception as e:
            raise StorageError(f"Failed to export data: {str(e)}") from e
    
    def import_data(self, format: StorageFormat, path: str, clear_existing: bool = False) -> int:
        """Import data from a file.
        
        Args:
            format: The format of the file to import
            path: Path to the file to import
            clear_existing: Whether to clear existing data before import
            
        Returns:
            Number of items successfully imported
            
        Raises:
            StorageError: If import operation fails
        """
        try:
            if clear_existing:
                self.clear()
                
            if format == StorageFormat.JSON:
                with open(path, 'r', encoding=self.config.encoding) as f:
                    data = json.load(f)
            elif format == StorageFormat.CSV:
                data = []
                with open(path, 'r', newline='', encoding=self.config.encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data.append(row)
            else:
                raise StorageError(f"Unsupported import format: {format}")
                
            return self.save_many(data)
        except Exception as e:
            raise StorageError(f"Failed to import data: {str(e)}") from e


class MemoryStorage(DataStorage[Dict[str, Any]]):
    """In-memory storage implementation.
    
    This class stores data in memory using a dictionary.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize in-memory storage.
        
        Args:
            config: Storage configuration
        """
        super().__init__(config)
        self._data: Dict[str, Dict[str, Any]] = {}
    
    def save(self, data: Dict[str, Any]) -> bool:
        """Save a single data item.
        
        Args:
            data: Data to save (must contain an 'id' field)
            
        Returns:
            True if saved successfully
            
        Raises:
            ValidationError: If data fails validation
            StorageError: If data doesn't contain an ID
        """
        # Validate data
        is_valid, error = self.validator.validate(data)
        if not is_valid:
            raise ValidationError(error)
            
        # Check for ID
        if 'id' not in data:
            raise StorageError("Data must contain an 'id' field")
            
        with self._lock:
            self._data[str(data['id'])] = data.copy()
            
        return True
    
    def save_many(self, items: List[Dict[str, Any]]) -> int:
        """Save multiple data items.
        
        Args:
            items: List of data items to save
            
        Returns:
            Number of items successfully saved
        """
        saved_count = 0
        
        for item in items:
            try:
                if self.save(item):
                    saved_count += 1
            except (ValidationError, StorageError):
                continue
                
        return saved_count
    
    def get(self, key: str) -> Dict[str, Any]:
        """Retrieve a data item by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The retrieved data item
            
        Raises:
            DataNotFoundError: If the key is not found
        """
        with self._lock:
            if key not in self._data:
                raise DataNotFoundError(f"Data with key '{key}' not found")
                
            return self._data[key].copy()
    
    def get_many(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple data items by their keys.
        
        Args:
            keys: List of keys to look up
            
        Returns:
            List of retrieved data items
            
        Raises:
            DataNotFoundError: If any key is not found
        """
        result = []
        missing_keys = []
        
        with self._lock:
            for key in keys:
                if key in self._data:
                    result.append(self._data[key].copy())
                else:
                    missing_keys.append(key)
                    
        if missing_keys:
            raise DataNotFoundError(f"Data not found for keys: {missing_keys}")
            
        return result
    
    def query(self, filter_func: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """Query data items using a filter function.
        
        Args:
            filter_func: Function that takes a data item and returns True if it should be included
            
        Returns:
            List of data items matching the filter
        """
        result = []
        
        with self._lock:
            for item in self._data.values():
                if filter_func(item):
                    result.append(item.copy())
                    
        return result
    
    def update(self, key: str, data: Dict[str, Any]) -> bool:
        """Update a data item by key.
        
        Args:
            key: The key of the item to update
            data: The new data
            
        Returns:
            True if updated successfully
            
        Raises:
            DataNotFoundError: If the key is not found
            ValidationError: If the new data fails validation
        """
        # Validate data
        is_valid, error = self.validator.validate(data)
        if not is_valid:
            raise ValidationError(error)
            
        with self._lock:
            if key not in self._data:
                raise DataNotFoundError(f"Data with key '{key}' not found")
                
            self._data[key] = data.copy()
            
        return True
    
    def delete(self, key: str) -> bool:
        """Delete a data item by key.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            DataNotFoundError: If the key is not found
        """
        with self._lock:
            if key not in self._data:
                raise DataNotFoundError(f"Data with key '{key}' not found")
                
            del self._data[key]
            
        return True
    
    def clear(self) -> bool:
        """Clear all data from storage.
        
        Returns:
            True if cleared successfully
        """
        with self._lock:
            self._data.clear()
            
        return True
    
    def count(self) -> int:
        """Count the number of items in storage.
        
        Returns:
            Number of items
        """
        with self._lock:
            return len(self._data)


class FileStorage(DataStorage[Dict[str, Any]]):
    """File-based storage implementation.
    
    This class can store data in JSON or CSV files.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize file storage.
        
        Args:
            config: Storage configuration
            
        Raises:
            StorageError: If the storage format is not supported
        """
        super().__init__(config)
        
        if config.format not in [StorageFormat.JSON, StorageFormat.CSV]:
            raise StorageError(f"Unsupported file storage format: {config.format}")
            
        self._path = Path(config.path)
        self._ensure_storage_exists()
        self._data: Dict[str, Dict[str, Any]] = self._load_data()
    
    def _ensure_storage_exists(self) -> None:
        """Ensure the storage file exists."""
        if not self.config.create_if_missing:
            return
            
        # Create parent directories if they don't exist
        os.makedirs(self._path.parent, exist_ok=True)
        
        # Create the file if it doesn't exist
        if not self._path.exists():
            if self.config.format == StorageFormat.JSON:
                with open(self._path, 'w', encoding=self.config.encoding) as f:
                    json.dump({}, f)
            elif self.config.format == StorageFormat.CSV:
                # For CSV, we'll create it when we first write data
                with open(self._path, 'w', encoding=self.config.encoding) as f:
                    pass
    
    def _load_data(self) -> Dict[str, Dict[str, Any]]:
        """Load data from the file into memory.
        
        Returns:
            Dictionary of data items indexed by ID
        """
        data = {}
        
        try:
            if not self._path.exists():
                return data
                
            if self.config.format == StorageFormat.JSON:
                with open(self._path, 'r', encoding=self.config.encoding) as f:
                    file_content = f.read().strip()
                    if not file_content:
                        return data
                    json_data = json.loads(file_content)
                    
                    # Handle both object and array formats
                    if isinstance(json_data, dict):
                        data = json_data
                    elif isinstance(json_data, list):
                        for item in json_data:
                            if 'id' in item:
                                data[str(item['id'])] = item
                    
            elif self.config.format == StorageFormat.CSV:
                items = []
                with open(self._path, 'r', newline='', encoding=self.config.encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        items.append(row)
                        
                for item in items:
                    if 'id' in item:
                        data[str(item['id'])] = item
        except Exception as e:
            # If we can't load the file, start with empty data
            print(f"Warning: Failed to load data from {self._path}: {str(e)}")
            
        return data
    
    def _save_data(self) -> None:
        """Save all data to the file."""
        try:
            if self.config.format == StorageFormat.JSON:
                with open(self._path, 'w', encoding=self.config.encoding) as f:
                    json.dump(self._data, f, indent=2)
            elif self.config.format == StorageFormat.CSV:
                if not self._data:
                    with open(self._path, 'w', encoding=self.config.encoding) as f:
                        f.write("")
                    return
                    
                # Get all possible fieldnames from all items
                fieldnames = set()
                for item in self._data.values():
                    fieldnames.update(item.keys())
                    
                with open(self._path, 'w', newline='', encoding=self.config.encoding) as f:
                    writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                    writer.writeheader()
                    writer.writerows(self._data.values())
        except Exception as e:
            raise StorageError(f"Failed to save data to {self._path}: {str(e)}") from e
    
    def save(self, data: Dict[str, Any]) -> bool:
        """Save a single data item.
        
        Args:
            data: Data to save (must contain an 'id' field)
            
        Returns:
            True if saved successfully
            
        Raises:
            ValidationError: If data fails validation
            StorageError: If data doesn't contain an ID
        """
        # Validate data
        is_valid, error = self.validator.validate(data)
        if not is_valid:
            raise ValidationError(error)
            
        # Check for ID
        if 'id' not in data:
            raise StorageError("Data must contain an 'id' field")
            
        with self._lock:
            self._data[str(data['id'])] = data.copy()
            self._save_data()
            
        return True
    
    def save_many(self, items: List[Dict[str, Any]]) -> int:
        """Save multiple data items.
        
        Args:
            items: List of data items to save
            
        Returns:
            Number of items successfully saved
        """
        saved_count = 0
        valid_items = []
        
        # Validate all items first
        for item in items:
            is_valid, _ = self.validator.validate(item)
            if is_valid and 'id' in item:
                valid_items.append(item)
                saved_count += 1
        
        # Save all valid items at once
        with self._lock:
            for item in valid_items:
                self._data[str(item['id'])] = item.copy()
            
            self._save_data()
            
        return saved_count
    
    def get(self, key: str) -> Dict[str, Any]:
        """Retrieve a data item by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The retrieved data item
            
        Raises:
            DataNotFoundError: If the key is not found
        """
        with self._lock:
            if key not in self._data:
                raise DataNotFoundError(f"Data with key '{key}' not found")
                
            return self._data[key].copy()
    
    def get_many(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple data items by their keys.
        
        Args:
            keys: List of keys to look up
            
        Returns:
            List of retrieved data items
            
        Raises:
            DataNotFoundError: If any key is not found
        """
        result = []
        missing_keys = []
        
        with self._lock:
            for key in keys:
                if key in self._data:
                    result.append(self._data[key].copy())
                else:
                    missing_keys.append(key)
                    
        if missing_keys:
            raise DataNotFoundError(f"Data not found for keys: {missing_keys}")
            
        return result
    
    def query(self, filter_func: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """Query data items using a filter function.
        
        Args:
            filter_func: Function that takes a data item and returns True if it should be included
            
        Returns:
            List of data items matching the filter
        """
        result = []
        
        with self._lock:
            for item in self._data.values():
                if filter_func(item):
                    result.append(item.copy())
                    
        return result
    
    def update(self, key: str, data: Dict[str, Any]) -> bool:
        """Update a data item by key.
        
        Args:
            key: The key of the item to update
            data: The new data
            
        Returns:
            True if updated successfully
            
        Raises:
            DataNotFoundError: If the key is not found
            ValidationError: If the new data fails validation
        """
        # Validate data
        is_valid, error = self.validator.validate(data)
        if not is_valid:
            raise ValidationError(error)
            
        with self._lock:
            if key not in self._data:
                raise DataNotFoundError(f"Data with key '{key}' not found")
                
            self._data[key] = data.copy()
            self._save_data()
            
        return True
    
    def delete(self, key: str) -> bool:
        """Delete a data item by key.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            DataNotFoundError: If the key is not found
        """
        with self._lock:
            if key not in self._data:
                raise DataNotFoundError(f"Data with key '{key}' not found")
                
            del self._data[key]
            self._save_data()
            
        return True
    
    def clear(self) -> bool:
        """Clear all data from storage.
        
        Returns:
            True if cleared successfully
        """
        with self._lock:
            self._data.clear()
            self._save_data()
            
        return True
    
    def count(self) -> int:
        """Count the number of items in storage.
        
        Returns:
            Number of items
        """
        with self._lock:
            return len(self._data)


class SQLiteStorage(DataStorage[Dict[str, Any]]):
    """SQLite-based storage implementation.
    
    This class stores data in a SQLite database.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize SQLite storage.
        
        Args:
            config: Storage configuration
            
        Raises:
            StorageConnectionError: If connection to the database fails
        """
        super().__init__(config)
        
        if config.format != StorageFormat.SQLITE:
            raise StorageError("SQLiteStorage requires format to be set to 'sqlite'")
            
        self._table_name = config.table_name
        self._path = config.path
        
        try:
            self._ensure_storage_exists()
        except Exception as e:
            raise StorageConnectionError(f"Failed to initialize SQLite storage: {str(e)}") from e
    
    def _ensure_storage_exists(self) -> None:
        """Ensure the database and table exist."""
        if not self.config.create_if_missing:
            return
            
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        
        # Create the table if it doesn't exist
        self._execute_query(
            f'''
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            '''
        )
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database.
        
        Returns:
            SQLite connection object
        """
        try:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            raise StorageConnectionError(f"Failed to connect to SQLite database: {str(e)}") from e
    
    def _execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of rows returned by the query
            
        Raises:
            StorageError: If query execution fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            raise StorageError(f"Failed to execute query: {str(e)}") from e
    
    def _execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute a SQL update query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Number of rows affected
            
        Raises:
            StorageError: If query execution fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if self.config.auto_commit:
                    conn.commit()
                return cursor.rowcount
        except Exception as e:
            raise StorageError(f"Failed to execute update: {str(e)}") from e
    
    def _execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute a SQL query with multiple parameter sets.
        
        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
            
        Returns:
            Number of rows affected
            
        Raises:
            StorageError: If query execution fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                if self.config.auto_commit:
                    conn.commit()
                return cursor.rowcount
        except Exception as e:
            raise StorageError(f"Failed to execute batch update: {str(e)}") from e
    
    def save(self, data: Dict[str, Any]) -> bool:
        """Save a single data item.
        
        Args:
            data: Data to save (must contain an 'id' field)
            
        Returns:
            True if saved successfully
            
        Raises:
            ValidationError: If data fails validation
            StorageError: If data doesn't contain an ID
        """
        # Validate data
        is_valid, error = self.validator.validate(data)
        if not is_valid:
            raise ValidationError(error)
            
        # Check for ID
        if 'id' not in data:
            raise StorageError("Data must contain an 'id' field")
            
        with self._lock:
            data_json = json.dumps(data)
            query = f"INSERT OR REPLACE INTO {self._table_name} (id, data) VALUES (?, ?)"
            self._execute_update(query, (str(data['id']), data_json))
            
        return True
    
    def save_many(self, items: List[Dict[str, Any]]) -> int:
        """Save multiple data items.
        
        Args:
            items: List of data items to save
            
        Returns:
            Number of items successfully saved
        """
        saved_count = 0
        valid_params = []
        
        # Validate all items first
        for item in items:
            is_valid, _ = self.validator.validate(item)
            if is_valid and 'id' in item:
                data_json = json.dumps(item)
                valid_params.append((str(item['id']), data_json))
                saved_count += 1
        
        # Process in batches
        if valid_params:
            with self._lock:
                batch_size = self.config.batch_size
                for i in range(0, len(valid_params), batch_size):
                    batch = valid_params[i:i + batch_size]
                    query = f"INSERT OR REPLACE INTO {self._table_name} (id, data) VALUES (?, ?)"
                    self._execute_many(query, batch)
            
        return saved_count
    
    def get(self, key: str) -> Dict[str, Any]:
        """Retrieve a data item by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The retrieved data item
            
        Raises:
            DataNotFoundError: If the key is not found
        """
        query = f"SELECT data FROM {self._table_name} WHERE id = ?"
        rows = self._execute_query(query, (key,))
        
        if not rows:
            raise DataNotFoundError(f"Data with key '{key}' not found")
            
        return json.loads(rows[0]['data'])
    
    def get_many(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple data items by their keys.
        
        Args:
            keys: List of keys to look up
            
        Returns:
            List of retrieved data items
            
        Raises:
            DataNotFoundError: If any key is not found
        """
        if not keys:
            return []
            
        placeholders = ','.join(['?'] * len(keys))
        query = f"SELECT id, data FROM {self._table_name} WHERE id IN ({placeholders})"
        rows = self._execute_query(query, tuple(keys))
        
        result = []
        found_keys = set()
        
        for row in rows:
            found_keys.add(row['id'])
            result.append(json.loads(row['data']))
            
        missing_keys = set(keys) - found_keys
        if missing_keys:
            raise DataNotFoundError(f"Data not found for keys: {missing_keys}")
            
        return result
    
    def query(self, filter_func: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """Query data items using a filter function.
        
        Args:
            filter_func: Function that takes a data item and returns True if it should be included
            
        Returns:
            List of data items matching the filter
        """
        query = f"SELECT data FROM {self._table_name}"
        rows = self._execute_query(query)
        
        result = []
        for row in rows:
            item = json.loads(row['data'])
            if filter_func(item):
                result.append(item)
                
        return result
    
    def update(self, key: str, data: Dict[str, Any]) -> bool:
        """Update a data item by key.
        
        Args:
            key: The key of the item to update
            data: The new data
            
        Returns:
            True if updated successfully
            
        Raises:
            DataNotFoundError: If the key is not found
            ValidationError: If the new data fails validation
        """
        # Check if the item exists
        exists_query = f"SELECT 1 FROM {self._table_name} WHERE id = ?"
        rows = self._execute_query(exists_query, (key,))
        
        if not rows:
            raise DataNotFoundError(f"Data with key '{key}' not found")
            
        # Validate data
        is_valid, error = self.validator.validate(data)
        if not is_valid:
            raise ValidationError(error)
            
        with self._lock:
            data_json = json.dumps(data)
            query = f"UPDATE {self._table_name} SET data = ? WHERE id = ?"
            self._execute_update(query, (data_json, key))
            
        return True
    
    def delete(self, key: str) -> bool:
        """Delete a data item by key.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            DataNotFoundError: If the key is not found
        """
        # Check if the item exists
        exists_query = f"SELECT 1 FROM {self._table_name} WHERE id = ?"
        rows = self._execute_query(exists_query, (key,))
        
        if not rows:
            raise DataNotFoundError(f"Data with key '{key}' not found")
            
        with self._lock:
            query = f"DELETE FROM {self._table_name} WHERE id = ?"
            self._execute_update(query, (key,))
            
        return True
    
    def clear(self) -> bool:
        """Clear all data from storage.
        
        Returns:
            True if cleared successfully
        """
        with self._lock:
            query = f"DELETE FROM {self._table_name}"
            self._execute_update(query)
            
        return True
    
    def count(self) -> int:
        """Count the number of items in storage.
        
        Returns:
            Number of items
        """
        query = f"SELECT COUNT(*) as count FROM {self._table_name}"
        rows = self._execute_query(query)
        
        return rows[0]['count'] if rows else 0


def create_storage(config: StorageConfig) -> DataStorage:
    """Factory function to create a storage instance based on configuration.
    
    Args:
        config: Storage configuration
        
    Returns:
        Appropriate DataStorage implementation
        
    Raises:
        StorageError: If the storage format is not supported
    """
    if config.format == StorageFormat.MEMORY:
        return MemoryStorage(config)
    elif config.format in [StorageFormat.JSON, StorageFormat.CSV]:
        return FileStorage(config)
    elif config.format == StorageFormat.SQLITE:
        return SQLiteStorage(config)
    else:
        raise StorageError(f"Unsupported storage format: {config.format}")

