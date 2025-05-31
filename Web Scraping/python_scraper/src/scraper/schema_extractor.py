#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema Extractor Module

This module provides functionality for detecting and extracting schemas from
various sources including JSON responses, GraphQL schemas, HTML forms, and 
OpenAPI/Swagger documentation.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Type
from urllib.parse import urljoin, urlparse
import copy
import inspect

import requests
from bs4 import BeautifulSoup
import jsonschema
from pydantic import BaseModel, Field, create_model
from jsonschema.exceptions import ValidationError

from .base import BaseScraper, ScraperResult, ScraperConfig

# Set up logging
logger = logging.getLogger("webscraper.schema_extractor")


class SchemaType(BaseModel):
    """
    Model for representing a schema type with validation rules.
    
    This is used to build structured schema representations that can be
    converted to various schema formats (JSON Schema, GraphQL, etc.)
    """
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False
    format: Optional[str] = None
    pattern: Optional[str] = None
    enum: Optional[List[Any]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    multiple_of: Optional[float] = None
    properties: Optional[Dict[str, 'SchemaType']] = None
    items: Optional['SchemaType'] = None
    additional_properties: Optional[bool] = None
    example: Optional[Any] = None
    nullable: bool = False
    

class SchemaExtractorConfig(BaseModel):
    """Configuration specific to schema extraction."""
    
    # General configuration
    infer_types: bool = Field(True, description="Whether to infer types from sample data")
    min_samples_for_inference: int = Field(3, description="Minimum number of samples needed for reliable type inference")
    
    # JSON schema extraction
    json_schema_draft: str = Field("draft-07", description="JSON Schema draft version to use")
    detect_string_formats: bool = Field(True, description="Whether to attempt to detect string formats (e.g., date, email)")
    detect_enums: bool = Field(True, description="Whether to detect enum values from repeated occurrences")
    enum_detection_threshold: float = Field(0.8, description="Threshold for enum detection (percentage of samples with the same values)")
    
    # GraphQL schema extraction
    use_graphql_introspection: bool = Field(True, description="Whether to use GraphQL introspection to extract schemas")
    
    # HTML form extraction
    extract_input_validation: bool = Field(True, description="Whether to extract validation rules from HTML input elements")
    
    # OpenAPI/Swagger extraction
    detect_openapi_endpoints: bool = Field(True, description="Whether to look for OpenAPI/Swagger documentation")
    openapi_paths: List[str] = Field(
        default_factory=lambda: [
            "/swagger.json", "/swagger/v1/swagger.json", "/api-docs", "/api-docs.json",
            "/openapi.json", "/swagger-ui.html", "/docs", "/redoc"
        ],
        description="Paths to check for OpenAPI/Swagger documentation"
    )


class SchemaExtractor(BaseScraper):
    """
    Schema Extractor for detecting and extracting schemas from various sources.
    
    This class extends the BaseScraper to provide specialized functionality
    for schema extraction, validation, and documentation generation.
    """
    
    def __init__(
        self, 
        config: Optional[Union[Dict[str, Any], ScraperConfig]] = None,
        schema_config: Optional[Union[Dict[str, Any], SchemaExtractorConfig]] = None
    ):
        """
        Initialize the schema extractor with the given configuration.
        
        Args:
            config: General scraper configuration
            schema_config: Schema extractor specific configuration
        """
        super().__init__(config)
        
        # Set up schema extractor configuration
        if schema_config is None:
            self.schema_config = SchemaExtractorConfig()
        elif isinstance(schema_config, dict):
            self.schema_config = SchemaExtractorConfig.model_validate(schema_config)
        else:
            self.schema_config = schema_config
        
        # Set up string format pattern detectors
        self._setup_format_detectors()
        
        # Track extracted schemas to avoid duplicates
        self.extracted_schemas: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Schema Extractor initialized with configuration: %s", self.schema_config.model_dump())
    
    def _setup_format_detectors(self):
        """Set up regex patterns for detecting string formats."""
        self.format_patterns = {
            "email": re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'),
            "date": re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            "date-time": re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'),
            "time": re.compile(r'^\d{2}:\d{2}:\d{2}$'),
            "uri": re.compile(r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'),
            "hostname": re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'),
            "ipv4": re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$'),
            "uuid": re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE),
        }
    
    def scrape(self, url: str, **kwargs) -> ScraperResult:
        """
        Scrape schemas from the given URL.
        
        This method discovers schemas from various sources on the target website.
        
        Args:
            url: The URL to scrape
            **kwargs: Additional scraper-specific parameters
            
        Returns:
            A ScraperResult object containing the discovered schemas
        """
        logger.info("Extracting schemas from %s", url)
        
        # Fetch the HTML page
        status, soup = self.fetch_html(url)
        
        if status < 200 or status >= 300 or soup is None:
            logger.warning("Failed to fetch HTML from %s: HTTP %d", url, status)
            return ScraperResult(
                url=url,
                status_code=status,
                content_type="text/html",
                data={"error": f"Failed to fetch HTML (HTTP {status})"},
                metadata={}
            )
        
        schemas = {}
        
        # Extract schemas from different sources
        form_schemas = self.extract_form_schemas(soup, url)
        if form_schemas:
            schemas["forms"] = form_schemas
        
        # Try to find OpenAPI/Swagger documentation
        if self.schema_config.detect_openapi_endpoints:
            openapi_schemas = self.extract_openapi_schemas(url)
            if openapi_schemas:
                schemas["openapi"] = openapi_schemas
        
        # Try to find GraphQL schema if configured
        if self.schema_config.use_graphql_introspection:
            graphql_schemas = self.extract_graphql_schema(url)
            if graphql_schemas:
                schemas["graphql"] = graphql_schemas
        
        # Look for JSON data in the page
        json_schemas = self.extract_json_schemas(soup, url)
        if json_schemas:
            schemas["json"] = json_schemas
        
        # Extract metadata about the schemas
        metadata = {
            "url": url,
            "form_count": len(form_schemas) if form_schemas else 0,
            "has_openapi": bool(openapi_schemas) if "openapi" in schemas else False,
            "has_graphql": bool(graphql_schemas) if "graphql" in schemas else False,
            "json_schema_count": len(json_schemas) if json_schemas else 0,
        }
        
        return ScraperResult(
            url=url,
            status_code=status,
            content_type="application/json",
            data=schemas,
            metadata=metadata
        )
    
    def extract_form_schemas(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract schema information from HTML forms.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of form schemas
        """
        form_schemas = []
        
        # Find all forms in the page
        forms = soup.find_all("form")
        logger.info("Found %d forms to analyze", len(forms))
        
        for i, form in enumerate(forms):
            form_id = form.get("id", f"form_{i}")
            form_name = form.get("name", form_id)
            form_action = form.get("action", "")
            form_method = form.get("method", "get").upper()
            
            # Resolve relative action URL
            form_action = urljoin(base_url, form_action)
            
            form_schema = {
                "id": form_id,
                "name": form_name,
                "action": form_action,
                "method": form_method,
                "fields": []
            }
            
            # Extract fields from the form
            fields = []
            
            # Process input elements
            for input_elem in form.find_all("input"):
                field = self._extract_input_field(input_elem)
                if field:
                    fields.append(field)
            
            # Process select elements
            for select_elem in form.find_all("select"):
                field = self._extract_select_field(select_elem)
                if field:
                    fields.append(field)
            
            # Process textarea elements
            for textarea_elem in form.find_all("textarea"):
                field = self._extract_textarea_field(textarea_elem)
                if field:
                    fields.append(field)
            
            # Add fields to the form schema
            form_schema["fields"] = fields
            
            # Generate JSON Schema for the form
            properties = {}
            required = []
            
            for field in fields:
                field_name = field["name"]
                properties[field_name] = {
                    "type": field["type"],
                    "description": field.get("description", "")
                }
                
                # Add format if present
                if "format" in field:
                    properties[field_name]["format"] = field["format"]
                
                # Add enum if present
                if "enum" in field:
                    properties[field_name]["enum"] = field["enum"]
                
                # Add pattern if present
                if "pattern" in field:
                    properties[field_name]["pattern"] = field["pattern"]
                
                # Add min and max if present
                if "minLength" in field:
                    properties[field_name]["minLength"] = field["minLength"]
                if "maxLength" in field:
                    properties[field_name]["maxLength"] = field["maxLength"]
                if "minimum" in field:
                    properties[field_name]["minimum"] = field["minimum"]
                if "maximum" in field:
                    properties[field_name]["maximum"] = field["maximum"]
                
                # Add required fields
                if field.get("required", False):
                    required.append(field_name)
            
            form_schema["json_schema"] = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            }
            
            form_schemas.append(form_schema)
        
        return form_schemas
    
    def _extract_input_field(self, input_elem) -> Optional[Dict[str, Any]]:
        """
        Extract schema information from an HTML input element.
        
        Args:
            input_elem: BeautifulSoup element representing an input
            
        Returns:
            Dictionary with field information or None if the field should be skipped
        """
        input_type = input_elem.get("type", "text").lower()
        input_name = input_elem.get("name", "")
        
        # Skip inputs without a name or hidden inputs
        if not input_name or input_type in ["hidden", "submit", "button", "image", "reset"]:
            return None
        
        field = {
            "name": input_name,
            "label": self._find_label_for_input(input_elem),
            "required": input_elem.has_attr("required"),
            "html_type": input_type
        }
        
        # Set JSON Schema type based on input type
        if input_type in ["text", "search", "tel", "url", "email", "password"]:
            field["type"] = "string"
            
            # Add format if available
            if input_type == "email":
                field["format"] = "email"
            elif input_type == "url":
                field["format"] = "uri"
            
            # Add pattern if available
            if input_elem.has_attr("pattern"):
                field["pattern"] = input_elem.get("pattern")
            
            # Add min and max length if available
            if input_elem.has_attr("minlength"):
                try:
                    field["minLength"] = int(input_elem.get("minlength"))
                except ValueError:
                    pass
            
            if input_elem.has_attr("maxlength"):
                try:
                    field["maxLength"] = int(input_elem.get("maxlength"))
                except ValueError:
                    pass
            
        elif input_type in ["number", "range"]:
            field["type"] = "number"
            
            # Add min and max if available
            if input_elem.has_attr("min"):
                try:
                    field["minimum"] = float(input_elem.get("min"))
                except ValueError:
                    pass
            
            if input_elem.has_attr("max"):
                try:
                    field["maximum"] = float(input_elem.get("max"))
                except ValueError:
                    pass
            
            # Add step if available
            if input_elem.has_attr("step"):
                try:
                    step = float(input_elem.get("step"))
                    if step > 0:
                        field["multipleOf"] = step
                except ValueError:
                    pass
            
        elif input_type == "checkbox":
            field["type"] = "boolean"
            
        elif input_type == "date":
            field["type"] = "string"
            field["format"] = "date"
            
        elif input_type == "datetime-local":
            field["type"] = "string"
            field["format"] = "date-time"
            
        elif input_type == "time":
            field["type"] = "string"
            field["format"] = "time"
            
        elif input_type == "month":
            field["type"] = "string"
            field["pattern"] = "^\\d{4}-\\d{2}$"
            
        elif input_type == "week":
            field["type"] = "string"
            field["pattern"] = "^\\d{4}-W\\d{2}$"
            
        elif input_type == "color":
            field["type"] = "string"
            field["pattern"] = "^#[0-9a-f]{6}$"
            
        elif input_type == "file":
            field["type"] = "string"
            field["format"] = "binary"
            
            # Add accept attribute if available
            if input_elem.has_attr("accept"):
                field["accept"] = input_elem.get("accept").split(",")
            
            # Check if multiple files are allowed
            if input_elem.has_attr("multiple"):
                field["type"] = "array"
                field["items"] = {"type": "string", "format": "binary"}
            
        elif input_type == "radio":
            field["type"] = "string"
            
            # Find all radio buttons with the same name to build an enum
            parent_form = input_elem.find_parent("form")
            if parent_form:
                same_name_radios = parent_form.find_all("input", {"type": "radio", "name": input_name})
                enum_values = []
                
                for radio in same_name_radios:
                    if radio.has_attr("value"):
                        enum_values.append(radio.get("value"))
                
                if enum_values:
                    field["enum"] = enum_values
        
        else:
            # Default to string for unknown input types
            field["type"] = "string"
        
        # Add default value if available
        if input_elem.has_attr("value") and input_type not in ["checkbox", "radio"]:
            field["default"] = input_elem.get("value")
        
        # Add placeholder as description if available
        if input_elem.has_attr("placeholder"):
            field["description"] = input_elem.get("placeholder")
        
        return field
    
    def _extract_select_field(self, select_elem) -> Optional[Dict[str, Any]]:
        """
        Extract schema information from an HTML select element.
        
        Args:
            select_elem: BeautifulSoup element representing a select
            
        Returns:
            Dictionary with field information or None if the field should be skipped
        """
        select_name = select_elem.get("name", "")
        
        # Skip selects without a name
        if not select_name:
            return None
        
        field = {
            "name": select_name,
            "label": self._find_label_for_input(select_elem),
            "required": select_elem.has_attr("required"),
            "html_type": "select"
        }
        
        # Check if multiple selection is allowed
        if select_elem.has_attr("multiple"):
            field["type"] = "array"
            field["items"] = {"type": "string"}
        else:
            field["type"] = "string"
        
        # Extract options
        options = select_elem.find_all("option")
        enum_values = []
        
        for option in options:
            if option.has_attr("value"):
                enum_values.append(option.get("value"))
        
        if enum_values:
            if field["type"] == "array":
                field["items"]["enum"] = enum_values
            else:
                field["enum"] = enum_values
        
        return field
    
    def _extract_textarea_field(self, textarea_elem) -> Optional[Dict[str, Any]]:
        """
        Extract schema information from an HTML textarea element.
        
        Args:
            textarea_elem: BeautifulSoup element representing a textarea
            
        Returns:
            Dictionary with field information or None if the field should be skipped
        """
        textarea_name = textarea_elem.get("name", "")
        
        # Skip textareas without a name
        if not textarea_name:
            return None
        
        field = {
            "name": textarea_name,
            "label": self._find_label_for_input(textarea_elem),
            "required": textarea_elem.has_attr("required"),
            "html_type": "textarea",
            "type": "string"
        }
        
        # Add min and max length if available
        if textarea_elem.has_attr("minlength"):
            try:
                field["minLength"] = int(textarea_elem.get("minlength"))
            except ValueError:
                pass
        
        if textarea_elem.has_attr("maxlength"):
            try:
                field["maxLength"] = int(textarea_elem.get("maxlength"))
            except ValueError:
                pass
        
        # Add placeholder as description if available
        if textarea_elem.has_attr("placeholder"):
            field["description"] = textarea_elem.get("placeholder")
        
        # Add default value if content exists
        if textarea_elem.string:
            field["default"] = textarea_elem.string.strip()
        
        return field
    
    def _find_label_for_input(self, input_elem) -> Optional[str]:
        """
        Find the label text for an input element.
        
        Args:
            input_elem: BeautifulSoup element representing an input
            
        Returns:
            Label text or None if no label is found
        """
        # Check if the input has an id attribute
        input_id = input_elem.get("id")
        
        if input_id:
            # Find a label with a matching for attribute
            label_elem = input_elem.find_parent("form").find("label", {"for": input_id})
            if label_elem:
                return label_elem.get_text(strip=True)
        
        # Check if the input is a child of a label
        parent_label = input_elem.find_parent("label")
        if parent_label:
            # Remove the text of any child inputs/selects from the label text
            label_text = parent_label.get_text(strip=True)
            for child in parent_label.find_all(["input", "select", "textarea"]):
                child_text = child.get_text(strip=True)
                if child_text:
                    label_text = label_text.replace(child_text, "").strip()
            return label_text
        
        return None
    
    def extract_openapi_schemas(self, base_url: str) -> Dict[str, Any]:
        """
        Extract OpenAPI/Swagger schema information.
        
        Args:
            base_url: Base URL for the website
            
        Returns:
            Dictionary with OpenAPI schema information
        """
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Try to find OpenAPI/Swagger documentation
        for path in self.schema_config.openapi_paths:
            openapi_url = urljoin(base_domain, path)
            
            # Try to fetch OpenAPI documentation
            status, content_type, json_data, text = self.make_request(openapi_url)
            
            if status >= 200 and status < 300:
                # Check if it's a JSON response
                if json_data is not None:
                    # Check if it looks like an OpenAPI schema
                    if self._validate_openapi_schema(json_data):
                        logger.info("Found OpenAPI schema at %s", openapi_url)
                        return {
                            "url": openapi_url,
                            "version": json_data.get("openapi", json_data.get("swagger", "unknown")),
                            "schema": json_data
                        }
                
                # Check if it's an HTML page with Swagger UI or ReDoc
                elif text and "text/html" in content_type:
                    # Check for common Swagger UI or ReDoc patterns
                    if re.search(r'swagger-ui|swaggerUi|openapi|api-docs|redoc', text, re.IGNORECASE):
                        # Try to extract schema URL from the HTML
                        schema_urls = self._extract_schema_urls_from_html(text, openapi_url)
                        
                        if schema_urls:
                            # Try to fetch the schema
                            for schema_url in schema_urls:
                                status, _, json_data, _ = self.make_request(schema_url)
                                
                                if status >= 200 and status < 300 and json_data is not None:
                                    if self._validate_openapi_schema(json_data):
                                        logger.info("Found OpenAPI schema at %s via %s", schema_url, openapi_url)
                                        return {
                                            "url": schema_url,
                                            "documentation_url": openapi_url,
                                            "version": json_data.get("openapi", json_data.get("swagger", "unknown")),
                                            "schema": json_data
                                        }
        
        logger.info("No OpenAPI schema found")
        return {}
    
    def _validate_openapi_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Validate that a JSON object is an OpenAPI schema.
        
        Args:
            schema: JSON object to validate
            
        Returns:
            True if the schema appears to be an OpenAPI schema, False otherwise
        """
        # Check for OpenAPI 3.x
        if "openapi" in schema and isinstance(schema.get("paths"), dict):
            return True
        
        # Check for Swagger 2.0
        if "swagger" in schema and schema.get("swagger", "").startswith("2.") and isinstance(schema.get("paths"), dict):
            return True
        
        return False
    
    def _extract_schema_urls_from_html(self, html: str, base_url: str) -> List[str]:
        """
        Extract OpenAPI schema URLs from HTML content.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of potential schema URLs
        """
        soup = BeautifulSoup(html, "html.parser")
        schema_urls = []
        
        # Look for script tags with potential schema URLs
        for script in soup.find_all("script"):
            if script.string:
                # Look for common patterns in JavaScript
                url_matches = re.findall(r'url:\s*[\'"]([^\'"]+)[\'"]', script.string)
                schema_matches = re.findall(r'schemaUrl\s*=\s*[\'"]([^\'"]+)[\'"]', script.string)
                spec_matches = re.findall(r'spec:\s*[\'"]([^\'"]+)[\'"]', script.string)
                
                for match in url_matches + schema_matches + spec_matches:
                    if any(ext in match.lower() for ext in [".json", ".yaml", ".yml"]) or \
                       any(term in match.lower() for term in ["api-docs", "swagger", "openapi"]):
                        schema_urls.append(urljoin(base_url, match))
        
        # Look for links to potential schema files
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if any(ext in href.lower() for ext in [".json", ".yaml", ".yml"]) or \
               any(term in href.lower() for term in ["api-docs", "swagger", "openapi"]):
                schema_urls.append(urljoin(base_url, href))
        
        return list(set(schema_urls))  # Remove duplicates
    
    def extract_graphql_schema(self, base_url: str) -> Dict[str, Any]:
        """
        Extract GraphQL schema information using introspection.
        
        Args:
            base_url: Base URL for the website
            
        Returns:
            Dictionary with GraphQL schema information
        """
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Common GraphQL endpoint paths
        graphql_paths = ["/graphql", "/gql", "/api/graphql", "/v1/graphql", "/query"]
        
        # Introspection query
        introspection_query = {
            "query": """
                query IntrospectionQuery {
                  __schema {
                    queryType { name }
                    mutationType { name }
                    subscriptionType { name }
                    types {
                      ...FullType
                    }
                    directives {
                      name
                      description
                      locations
                      args {
                        ...InputValue
                      }
                    }
                  }
                }
                
                fragment FullType on __Type {
                  kind
                  name
                  description
                  fields(includeDeprecated: true) {
                    name
                    description
                    args {
                      ...InputValue
                    }
                    type {
                      ...TypeRef
                    }
                    isDeprecated
                    deprecationReason
                  }
                  inputFields {
                    ...InputValue
                  }
                  interfaces {
                    ...TypeRef
                  }
                  enumValues(includeDeprecated: true) {
                    name
                    description
                    isDeprecated
                    deprecationReason
                  }
                  possibleTypes {
                    ...TypeRef
                  }
                }
                
                fragment InputValue on __InputValue {
                  name
                  description
                  type { ...TypeRef }
                  defaultValue
                }
                
                fragment TypeRef on __Type {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                        ofType {
                          kind
                          name
                          ofType {
                            kind
                            name
                            ofType {
                              kind
                              name
                              ofType {
                                kind
                                name
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
            """
        }
        
        # Try each potential GraphQL endpoint
        for path in graphql_paths:
            graphql_url = urljoin(base_domain, path)
            
            # Try to execute introspection query
            status, content_type, json_data, _ = self.make_request(
                graphql_url, 
                method="POST",
                json_data=introspection_query,
                headers={"Content-Type": "application/json"}
            )
            
            if status >= 200 and status < 300 and json_data is not None:
                # Check if it looks like a GraphQL response
                if "data" in json_data and "__schema" in json_data.get("data", {}):
                    logger.info("Found GraphQL schema at %s", graphql_url)
                    
                    # Convert GraphQL schema to a more usable format
                    schema = self._process_graphql_schema(json_data["data"]["__schema"])
                    
                    return {
                        "url": graphql_url,
                        "raw_schema": json_data["data"]["__schema"],
                        "processed_schema": schema
                    }
        
        logger.info("No GraphQL schema found")
        return {}
    
    def _process_graphql_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a GraphQL schema into a more usable format.
        
        Args:
            schema: Raw GraphQL schema from introspection
            
        Returns:
            Processed schema with types and fields
        """
        processed_schema = {
            "query": schema.get("queryType", {}).get("name"),
            "mutation": schema.get("mutationType", {}).get("name"),
            "subscription": schema.get("subscriptionType", {}).get("name"),
            "types": {}
        }
        
        # Process each type
        for type_info in schema.get("types", []):
            # Skip introspection types (starting with __)
            if type_info.get("name", "").startswith("__"):
                continue
            
            kind = type_info.get("kind")
            name = type_info.get("name")
            
            if not name:
                continue
            
            type_data = {
                "kind": kind,
                "description": type_info.get("description"),
            }
            
            # Add fields for object and interface types
            if kind in ["OBJECT", "INTERFACE"] and "fields" in type_info:
                type_data["fields"] = {}
                for field in type_info.get("fields", []):
                    field_name = field.get("name")
                    if field_name:
                        type_data["fields"][field_name] = {
                            "description": field.get("description"),
                            "type": self._process_graphql_type(field.get("type", {})),
                            "args": self._process_graphql_args(field.get("args", [])),
                            "isDeprecated": field.get("isDeprecated", False),
                            "deprecationReason": field.get("deprecationReason"),
                        }
            
            # Add enum values for enum types
            if kind == "ENUM" and "enumValues" in type_info:
                type_data["values"] = []
                for enum_value in type_info.get("enumValues", []):
                    enum_name = enum_value.get("name")
                    if enum_name:
                        type_data["values"].append({
                            "name": enum_name,
                            "description": enum_value.get("description"),
                            "isDeprecated": enum_value.get("isDeprecated", False),
                            "deprecationReason": enum_value.get("deprecationReason"),
                        })
            
            # Add input fields for input object types
            if kind == "INPUT_OBJECT" and "inputFields" in type_info:
                type_data["inputFields"] = {}
                for input_field in type_info.get("inputFields", []):
                    field_name = input_field.get("name")
                    if field_name:
                        type_data["inputFields"][field_name] = {
                            "description": input_field.get("description"),
                            "type": self._process_graphql_type(input_field.get("type", {})),
                            "defaultValue": input_field.get("defaultValue"),
                        }
            
            # Add possible types for union types
            if kind == "UNION" and "possibleTypes" in type_info:
                type_data["possibleTypes"] = []
                for possible_type in type_info.get("possibleTypes", []):
                    type_name = self._process_graphql_type(possible_type)
                    if type_name:
                        type_data["possibleTypes"].append(type_name)
            
            processed_schema["types"][name] = type_data
        
        return processed_schema
    
    def _process_graphql_type(self, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a GraphQL type definition.
        
        Args:
            type_info: GraphQL type information
            
        Returns:
            Processed type information
        """
        kind = type_info.get("kind")
        name = type_info.get("name")
        of_type = type_info.get("ofType")
        
        if kind == "NON_NULL" and of_type:
            inner_type = self._process_graphql_type(of_type)
            return {
                "kind": "NON_NULL",
                "ofType": inner_type
            }
        elif kind == "LIST" and of_type:
            inner_type = self._process_graphql_type(of_type)
            return {
                "kind": "LIST",
                "ofType": inner_type
            }
        elif name:
            return {
                "kind": kind,
                "name": name
            }
        
        return {}
    
    def _process_graphql_args(self, args: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Process GraphQL field arguments.
        
        Args:
            args: List of GraphQL argument definitions
            
        Returns:
            Dictionary of processed arguments
        """
        result = {}
        
        for arg in args:
            arg_name = arg.get("name")
            if arg_name:
                result[arg_name] = {
                    "description": arg.get("description"),
                    "type": self._process_graphql_type(arg.get("type", {})),
                    "defaultValue": arg.get("defaultValue"),
                }
        
        return result
    
    def extract_json_schemas(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        """
        Extract JSON schema information from a page.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            Dictionary with JSON schema information
        """
        schemas = {}
        
        # Look for JSON data in script tags
        for script in soup.find_all("script", {"type": "application/json"}):
            if script.string:
                try:
                    data = json.loads(script.string)
                    schema_id = script.get("id", f"json_{len(schemas)}")
                    schema = self.infer_json_schema(data, schema_id)
                    schemas[schema_id] = schema
                except json.JSONDecodeError:
                    pass
        
        # Look for JSON data in script tags with embedded JSON
        for script in soup.find_all("script"):
            if script.string:
                # Look for JSON object assignments in JavaScript
                json_matches = re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*({.+?});', script.string, re.DOTALL)
                
                for var_name, json_str in json_matches:
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, (dict, list)) and data:  # Only process non-empty objects/arrays
                            schema = self.infer_json_schema(data, var_name)
                            schemas[var_name] = schema
                    except json.JSONDecodeError:
                        pass
        
        # Look for JSON-LD data
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            if script.string:
                try:
                    data = json.loads(script.string)
                    schema_id = "jsonld"
                    schema = self.infer_json_schema(data, schema_id)
                    schemas[schema_id] = schema
                except json.JSONDecodeError:
                    pass
        
        return schemas
    
    def infer_json_schema(self, data: Any, schema_id: str = "schema") -> Dict[str, Any]:
        """
        Infer a JSON schema from a data sample.
        
        Args:
            data: JSON data
            schema_id: Identifier for the schema
            
        Returns:
            JSON Schema
        """
        schema = {
            "$schema": f"http://json-schema.org/draft-07/schema#",
            "title": schema_id,
            "description": f"Schema inferred from data sample"
        }
        
        # Generate schema from data
        type_schema = self._infer_type(data)
        schema.update(type_schema)
        
        return schema
    
    def _infer_type(self, data: Any, path: str = "") -> Dict[str, Any]:
        """
        Infer the type of a JSON value.
        
        Args:
            data: JSON value
            path: Current path in the JSON structure (for debugging)
            
        Returns:
            JSON Schema for the value
        """
        if data is None:
            return {"type": "null"}
        
        if isinstance(data, bool):
            return {"type": "boolean"}
        
        if isinstance(data, int):
            return {"type": "integer"}
        
        if isinstance(data, float):
            return {"type": "number"}
        
        if isinstance(data, str):
            schema = {"type": "string"}
            
            # Try to detect string format if enabled
            if self.schema_config.detect_string_formats:
                detected_format = self._detect_string_format(data)
                if detected_format:
                    schema["format"] = detected_format
            
            return schema
        
        if isinstance(data, list):
            if not data:
                return {"type": "array", "items": {}}
            
            # Infer schema from the first item (simplified approach)
            # A more robust approach would check all items and potentially use oneOf for mixed types
            items_schema = self._infer_type(data[0], f"{path}[0]")
            
            # If all items are of the same type, we can use a single schema
            all_same_type = all(isinstance(item, type(data[0])) for item in data)
            
            if all_same_type:
                return {"type": "array", "items": items_schema}
            else:
                # For mixed types, use oneOf
                item_schemas = []
                for i, item in enumerate(data[:10]):  # Limit to first 10 items for performance
                    item_schema = self._infer_type(item, f"{path}[{i}]")
                    if item_schema not in item_schemas:
                        item_schemas.append(item_schema)
                
                return {"type": "array", "items": {"oneOf": item_schemas}}
        
        if isinstance(data, dict):
            properties = {}
            required = []
            
            for key, value in data.items():
                prop_schema = self._infer_type(value, f"{path}.{key}")
                properties[key] = prop_schema
                
                # Consider all properties required for simplicity
                # A more robust approach would analyze multiple samples
                required.append(key)
            
            return {
                "type": "object",
                "properties": properties,
                "required": required
            }
        
        # Fallback for other types
        return {"type": "string"}
    
    def _detect_string_format(self, value: str) -> Optional[str]:
        """
        Detect the format of a string value.
        
        Args:
            value: String value
            
        Returns:
            Detected format or None
        """
        for format_name, pattern in self.format_patterns.items():
            if pattern.match(value):
                return format_name
        
        return None
    
    def validate_against_schema(
        self, 
        data: Any, 
        schema: Dict[str, Any],
        format_checker: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON Schema
            format_checker: Whether to use format checker
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if format_checker:
                jsonschema.validate(
                    instance=data, 
                    schema=schema,
                    format_checker=jsonschema.FormatChecker()
                )
            else:
                jsonschema.validate(instance=data, schema=schema)
            
            return True, None
        except ValidationError as e:
            return False, str(e)
    
    def generate_pydantic_model(
        self, 
        schema: Dict[str, Any], 
        model_name: str = "Model"
    ) -> Type[BaseModel]:
        """
        Generate a Pydantic model from a JSON schema.
        
        Args:
            schema: JSON Schema
            model_name: Name for the generated model
            
        Returns:
            Pydantic model class
        """
        # Only support object schemas for now
        if schema.get("type") != "object":
            raise ValueError("Schema must be an object type")
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        model_fields = {}
        
        for prop_name, prop_schema in properties.items():
            field_type = self._json_schema_to_python_type(prop_schema)
            field_kwargs = {}
            
            # Set field description
            if "description" in prop_schema:
                field_kwargs["description"] = prop_schema["description"]
            
            # Set default value if property is not required
            if prop_name not in required:
                field_kwargs["default"] = None
            
            model_fields[prop_name] = (field_type, Field(**field_kwargs))
        
        # Create and return the model
        model = create_model(model_name, **model_fields)
        return model
    
    def _json_schema_to_python_type(self, schema: Dict[str, Any]) -> Any:
        """
        Convert JSON Schema type to Python type.
        
        Args:
            schema: JSON Schema
            
        Returns:
            Python type
        """
        schema_type = schema.get("type")
        
        if schema_type == "null":
            return None
        elif schema_type == "boolean":
            return bool
        elif schema_type == "integer":
            return int
        elif schema_type == "number":
            return float
        elif schema_type == "string":
            return str
        elif schema_type == "array":
            items_schema = schema.get("items", {})
            items_type = self._json_schema_to_python_type(items_schema)
            return List[items_type]
        elif schema_type == "object":
            return Dict[str, Any]
        
        # Handle oneOf, anyOf, allOf
        if "oneOf" in schema:
            # Just use the first type for simplicity
            return self._json_schema_to_python_type(schema["oneOf"][0])
        
        if "anyOf" in schema:
            # Just use the first type for simplicity
            return self._json_schema_to_python_type(schema["anyOf"][0])
        
        if "allOf" in schema:
            # Just use the first type for simplicity
            return self._json_schema_to_python_type(schema["allOf"][0])
        
        # Fallback
        return Any
    
    def generate_schema_documentation(
        self, 
        schema: Dict[str, Any], 
        format: str = "markdown"
    ) -> str:
        """
        Generate documentation from a schema.
        
        Args:
            schema: Schema to document
            format: Output format ("markdown", "html", or "text")
            
        Returns:
            Documentation string
        """
        if format.lower() == "markdown":
            return self._generate_markdown_docs(schema)
        elif format.lower() == "html":
            return self._generate_html_docs(schema)
        elif format.lower() == "text":
            return self._generate_text_docs(schema)
        else:
            raise ValueError(f"Unsupported documentation format: {format}")
    
    def _generate_markdown_docs(self, schema: Dict[str, Any]) -> str:
        """
        Generate Markdown documentation from a schema.
        
        Args:
            schema: Schema to document
            
        Returns:
            Markdown documentation
        """
        if "title" in schema:
            docs = f"# {schema['title']}\n\n"
        else:
            docs = "# Schema Documentation\n\n"
        
        if "description" in schema:
            docs += f"{schema['description']}\n\n"
        
        # Document properties for object schemas
        if schema.get("type") == "object" and "properties" in schema:
            docs += "## Properties\n\n"
            
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            docs += "| Property | Type | Required | Description |\n"
            docs += "|----------|------|----------|-------------|\n"
            
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "any")
                is_required = "Yes" if prop_name in required else "No"
                description = prop_schema.get("description", "")
                
                docs += f"| {prop_name} | {prop_type} | {is_required} | {description} |\n"
            
            docs += "\n"
            
            # Document nested objects
            for prop_name, prop_schema in properties.items():
                if prop_schema.get("type") == "object" and "properties" in prop_schema:
                    docs += f"### {prop_name}\n\n"
                    
                    if "description" in prop_schema:
                        docs += f"{prop_schema['description']}\n\n"
                    
                    nested_props = prop_schema.get("properties", {})
                    nested_required = prop_schema.get("required", [])
                    
                    docs += "| Property | Type | Required | Description |\n"
                    docs += "|----------|------|----------|-------------|\n"
                    
                    for nested_name, nested_schema in nested_props.items():
                        nested_type = nested_schema.get("type", "any")
                        is_required = "Yes" if nested_name in nested_required else "No"
                        description = nested_schema.get("description", "")
                        
                        docs += f"| {nested_name} | {nested_type} | {is_required} | {description} |\n"
                    
                    docs += "\n"
                
                # Document array items
                if prop_schema.get("type") == "array" and "items" in prop_schema:
                    items_schema = prop_schema.get("items", {})
                    
                    docs += f"### {prop_name} (array items)\n\n"
                    
                    if "description" in items_schema:
                        docs += f"{items_schema['description']}\n\n"
                    
                    if items_schema.get("type") == "object" and "properties" in items_schema:
                        items_props = items_schema.get("properties", {})
                        items_required = items_schema.get("required", [])
                        
                        docs += "| Property | Type | Required | Description |\n"
                        docs += "|----------|------|----------|-------------|\n"
                        
                        for item_name, item_schema in items_props.items():
                            item_type = item_schema.get("type", "any")
                            is_required = "Yes" if item_name in items_required else "No"
                            description = item_schema.get("description", "")
                            
                            docs += f"| {item_name} | {item_type} | {is_required} | {description} |\n"
                        
                        docs += "\n"
        
        # Document array schemas
        elif schema.get("type") == "array" and "items" in schema:
            items_schema = schema.get("items", {})
            
            docs += "## Array Items\n\n"
            
            if "description" in items_schema:
                docs += f"{items_schema['description']}\n\n"
            
            if items_schema.get("type") == "object" and "properties" in items_schema:
                items_props = items_schema.get("properties", {})
                items_required = items_schema.get("required", [])
                
                docs += "| Property | Type | Required | Description |\n"
                docs += "|----------|------|----------|-------------|\n"
                
                for item_name, item_schema in items_props.items():
                    item_type = item_schema.get("type", "any")
                    is_required = "Yes" if item_name in items_required else "No"
                    description = item_schema.get("description", "")
                    
                    docs += f"| {item_name} | {item_type} | {is_required} | {description} |\n"
                
                docs += "\n"
        
        return docs
    
    def _generate_html_docs(self, schema: Dict[str, Any]) -> str:
        """
        Generate HTML documentation from a schema.
        
        This is a simplified implementation. A real implementation would generate
        more comprehensive HTML with styling.
        
        Args:
            schema: Schema to document
            
        Returns:
            HTML documentation
        """
        # Convert Markdown to HTML (simplified implementation)
        markdown = self._generate_markdown_docs(schema)
        
        # Very basic conversion - a real implementation would use a proper
        # Markdown to HTML converter like markdown2 or mistune
        html = "<!DOCTYPE html>\n<html>\n<head>\n<title>Schema Documentation</title>\n"
        html += "<style>\n"
        html += "body { font-family: Arial, sans-serif; margin: 20px; }\n"
        html += "table { border-collapse: collapse; width: 100%; }\n"
        html += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n"
        html += "th { background-color: #f2f2f2; }\n"
        html += "</style>\n"
        html += "</head>\n<body>\n"
        
        # Convert markdown headers
        markdown = re.sub(r'^# (.+)$', r'<h1>\1</h1>', markdown, flags=re.MULTILINE)
        markdown = re.sub(r'^## (.+)$', r'<h2>\1</h2>', markdown, flags=re.MULTILINE)
        markdown = re.sub(r'^### (.+)$', r'<h3>\1</h3>', markdown, flags=re.MULTILINE)
        
        # Convert markdown tables (simplified)
        table_pattern = r'\| (.+) \|\n\|[-\|]+\|\n((?:\| .+ \|\n)+)'
        
        def replace_table(match):
            header = match.group(1)
            rows = match.group(2).strip().split('\n')
            
            header_cells = [cell.strip() for cell in header.split('|')]
            
            table_html = '<table>\n<thead>\n<tr>\n'
            for cell in header_cells:
                if cell:  # Skip empty cells from start/end
                    table_html += f'<th>{cell}</th>\n'
            table_html += '</tr>\n</thead>\n<tbody>\n'
            
            for row in rows:
                row_cells = [cell.strip() for cell in row.split('|')]
                table_html += '<tr>\n'
                for cell in row_cells:
                    if cell:  # Skip empty cells from start/end
                        table_html += f'<td>{cell}</td>\n'
                table_html += '</tr>\n'
            
            table_html += '</tbody>\n</table>\n'
            return table_html
        
        markdown = re.sub(table_pattern, replace_table, markdown, flags=re.MULTILINE)
        
        # Convert paragraph breaks
        markdown = re.sub(r'\n\n', r'</p>\n<p>', markdown)
        markdown = f"<p>{markdown}</p>"
        
        html += markdown
        html += "\n</body>\n</html>"
        
        return html
    
    def _generate_text_docs(self, schema: Dict[str, Any]) -> str:
        """
        Generate plain text documentation from a schema.
        
        Args:
            schema: Schema to document
            
        Returns:
            Plain text documentation
        """
        if "title" in schema:
            docs = f"{schema['title']}\n{'=' * len(schema['title'])}\n\n"
        else:
            docs = "Schema Documentation\n====================\n\n"
        
        if "description" in schema:
            docs += f"{schema['description']}\n\n"
        
        # Document properties for object schemas
        if schema.get("type") == "object" and "properties" in schema:
            docs += "Properties\n----------\n\n"
            
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "any")
                is_required = "Required" if prop_name in required else "Optional"
                description = prop_schema.get("description", "")
                
                docs += f"{prop_name} ({prop_type}, {is_required})\n"
                if description:
                    docs += f"    {description}\n"
                docs += "\n"
            
            # Document nested objects
            for prop_name, prop_schema in properties.items():
                if prop_schema.get("type") == "object" and "properties" in prop_schema:
                    docs += f"{prop_name} (Object)\n{'-' * (len(prop_name) + 9)}\n\n"
                    
                    if "description" in prop_schema:
                        docs += f"{prop_schema['description']}\n\n"
                    
                    nested_props = prop_schema.get("properties", {})
                    nested_required = prop_schema.get("required", [])
                    
                    for nested_name, nested_schema in nested_props.items():
                        nested_type = nested_schema.get("type", "any")
                        is_required = "Required" if nested_name in nested_required else "Optional"
                        description = nested_schema.get("description", "")
                        
                        docs += f"{nested_name} ({nested_type}, {is_required})\n"
                        if description:
                            docs += f"    {description}\n"
                        docs += "\n"
                
                # Document array items
                if prop_schema.get("type") == "array" and "items" in prop_schema:
                    items_schema = prop_schema.get("items", {})
                    
                    docs += f"{prop_name} (Array Items)\n{'-' * (len(prop_name) + 14)}\n\n"
                    
                    if "description" in items_schema:
                        docs += f"{items_schema['description']}\n\n"
                    
                    if items_schema.get("type") == "object" and "properties" in items_schema:
                        items_props = items_schema.get("properties", {})
                        items_required = items_schema.get("required", [])
                        
                        for item_name, item_schema in items_props.items():
                            item_type = item_schema.get("type", "any")
                            is_required = "Required" if item_name in items_required else "Optional"
                            description = item_schema.get("description", "")
                            
                            docs += f"{item_name} ({item_type}, {is_required})\n"
                            if description:
                                docs += f"    {description}\n"
                            docs += "\n"
        
        # Document array schemas
        elif schema.get("type") == "array" and "items" in schema:
            items_schema = schema.get("items", {})
            
            docs += "Array Items\n-----------\n\n"
            
            if "description" in items_schema:
                docs += f"{items_schema['description']}\n\n"
            
            if items_schema.get("type") == "object" and "properties" in items_schema:
                items_props = items_schema.get("properties", {})
                items_required = items_schema.get("required", [])
                
                for item_name, item_schema in items_props.items():
                    item_type = item_schema.get("type", "any")
                    is_required = "Required" if item_name in items_required else "Optional"
                    description = item_schema.get("description", "")
                    
                    docs += f"{item_name} ({item_type}, {is_required})\n"
                    if description:
                        docs += f"    {description}\n"
                    docs += "\n"
        
        return docs

