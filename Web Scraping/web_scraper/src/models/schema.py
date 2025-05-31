#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data models for scraping results.
Defines structured data classes for storing and processing scraped data.
"""

import datetime
import enum
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, HttpUrl, validator


class ContentType(enum.Enum):
    """Enum representing different content types."""
    HTML = "html"
    JSON = "json"
    XML = "xml"
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class ResourceType(enum.Enum):
    """Enum representing different resource types."""
    WEBPAGE = "webpage"
    API = "api"
    IMAGE = "image"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"
    OTHER = "other"


@dataclass
class ScrapedResource:
    """Base class for all scraped resources."""
    
    url: str
    status_code: int
    content_type: ContentType
    resource_type: ResourceType
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['content_type'] = self.content_type.value
        data['resource_type'] = self.resource_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedResource':
        """Create from dictionary."""
        # Convert string values to enums
        data['content_type'] = ContentType(data['content_type'])
        data['resource_type'] = ResourceType(data['resource_type'])
        
        # Convert timestamp string to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
            
        return cls(**data)


@dataclass
class ScrapedWebpage(ScrapedResource):
    """Model for scraped web pages."""
    
    title: Optional[str] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default values after initialization."""
        if not hasattr(self, 'resource_type') or self.resource_type is None:
            self.resource_type = ResourceType.WEBPAGE
            
        if not hasattr(self, 'content_type') or self.content_type is None:
            self.content_type = ContentType.HTML


@dataclass
class ScrapedApiResponse(ScrapedResource):
    """Model for scraped API responses."""
    
    json_content: Optional[Dict[str, Any]] = None
    raw_content: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default values after initialization."""
        if not hasattr(self, 'resource_type') or self.resource_type is None:
            self.resource_type = ResourceType.API
            
        if not hasattr(self, 'content_type') or self.content_type is None:
            self.content_type = ContentType.JSON


@dataclass
class ScrapedDataPoint:
    """Model for individual data points extracted from resources."""
    
    name: str
    value: Any
    source_url: str
    source_id: str
    data_type: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedDataPoint':
        """Create from dictionary."""
        # Convert timestamp string to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
            
        return cls(**data)


@dataclass
class ScrapeResult:
    """Container for a complete scrape operation result."""
    
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    resources: List[ScrapedResource] = field(default_factory=list)
    data_points: List[ScrapedDataPoint] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    
    def complete(self) -> None:
        """Mark the scrape operation as complete."""
        self.end_time = datetime.datetime.now()
        self.stats['duration'] = (self.end_time - self.start_time).total_seconds()
        self.stats['resources_count'] = len(self.resources)
        self.stats['data_points_count'] = len(self.data_points)
        self.stats['errors_count'] = len(self.errors)
        self.stats['success_rate'] = sum(1 for r in self.resources if r.success) / max(1, len(self.resources))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'stats': self.stats,
            'errors': self.errors,
            'resources': [r.to_dict() for r in self.resources],
            'data_points': [d.to_dict() for d in self.data_points]
        }
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapeResult':
        """Create from dictionary."""
        # Convert timestamp strings to datetime
        start_time = datetime.datetime.fromisoformat(data['start_time'])
        end_time = datetime.datetime.fromisoformat(data['end_time']) if data['end_time'] else None
        
        # Create basic instance
        result = cls(
            id=data['id'],
            name=data['name'],
            start_time=start_time,
            end_time=end_time,
            stats=data['stats'],
            errors=data['errors']
        )
        
        # Add resources
        for r_data in data['resources']:
            if r_data.get('resource_type') == ResourceType.WEBPAGE.value:
                resource = ScrapedWebpage.from_dict(r_data)
            elif r_data.get('resource_type') == ResourceType.API.value:
                resource = ScrapedApiResponse.from_dict(r_data)
            else:
                resource = ScrapedResource.from_dict(r_data)
            
            result.resources.append(resource)
        
        # Add data points
        for d_data in data['data_points']:
            data_point = ScrapedDataPoint.from_dict(d_data)
            result.data_points.append(data_point)
            
        return result


# Pydantic models for validation and TypeScript interface generation

class PydanticScrapedResource(BaseModel):
    """Pydantic model for scraped resources."""
    
    url: HttpUrl
    status_code: int
    content_type: str
    resource_type: str
    timestamp: datetime.datetime
    id: str
    success: bool
    error: Optional[str] = None
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate content type."""
        if v not in [ct.value for ct in ContentType]:
            raise ValueError(f"Invalid content type: {v}")
        return v
    
    @validator('resource_type')
    def validate_resource_type(cls, v):
        """Validate resource type."""
        if v not in [rt.value for rt in ResourceType]:
            raise ValueError(f"Invalid resource type: {v}")
        return v


class PydanticScrapedWebpage(PydanticScrapedResource):
    """Pydantic model for scraped web pages."""
    
    title: Optional[str] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PydanticScrapedApiResponse(PydanticScrapedResource):
    """Pydantic model for scraped API responses."""
    
    json_content: Optional[Dict[str, Any]] = None
    raw_content: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    request_params: Dict[str, Any] = Field(default_factory=dict)


class PydanticScrapedDataPoint(BaseModel):
    """Pydantic model for individual data points."""
    
    name: str
    value: Any
    source_url: str
    source_id: str
    data_type: str
    timestamp: datetime.datetime
    id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PydanticScrapeResult(BaseModel):
    """Pydantic model for a complete scrape operation result."""
    
    id: str
    name: Optional[str] = None
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    stats: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    resources: List[Union[PydanticScrapedResource, PydanticScrapedWebpage, PydanticScrapedApiResponse]] = Field(default_factory=list)
    data_points: List[PydanticScrapedDataPoint] = Field(default_factory=list)

