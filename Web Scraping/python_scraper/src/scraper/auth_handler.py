#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authentication Handler Module

This module provides functionality for handling various authentication methods
including Basic Auth, Token-based Auth, and OAuth2, as well as session management
and secure credential storage.
"""

import os
import json
import time
import logging
import base64
import hashlib
import hmac
import urllib.parse
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
import inspect
from abc import ABC, abstractmethod
import getpass
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import requests
from requests.auth import AuthBase, HTTPBasicAuth, HTTPDigestAuth
import keyring
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import yaml
from oauthlib.oauth2 import WebApplicationClient, BackendApplicationClient
from requests_oauthlib import OAuth2Session

from .base import BaseScraper, ScraperConfig

# Set up logging
logger = logging.getLogger("webscraper.auth_handler")


class AuthMethod(ABC):
    """
    Abstract base class for authentication methods.
    
    This class defines the interface that all authentication methods must implement.
    """
    
    @abstractmethod
    def apply_auth(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply authentication to a request.
        
        Args:
            request_kwargs: Request arguments for requests.request
            
        Returns:
            Updated request arguments with authentication applied
        """
        pass
    
    @abstractmethod
    def is_token_expired(self) -> bool:
        """
        Check if the authentication token is expired.
        
        Returns:
            True if the token is expired or not present, False otherwise
        """
        pass
    
    @abstractmethod
    def refresh_token(self) -> bool:
        """
        Refresh the authentication token if possible.
        
        Returns:
            True if the token was refreshed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers.
        
        Returns:
            Dictionary of authentication headers
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert authentication method to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the authentication method
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthMethod':
        """
        Create an authentication method from a dictionary.
        
        Args:
            data: Dictionary representation of the authentication method
            
        Returns:
            Authentication method instance
        """
        pass


class BasicAuth(AuthMethod):
    """Basic authentication method using username and password."""
    
    def __init__(self, username: str, password: str):
        """
        Initialize basic authentication.
        
        Args:
            username: Username for authentication
            password: Password for authentication
        """
        self.username = username
        self.password = password
        self._auth = HTTPBasicAuth(username, password)
    
    def apply_auth(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply basic authentication to a request.
        
        Args:
            request_kwargs: Request arguments for requests.request
            
        Returns:
            Updated request arguments with authentication applied
        """
        request_kwargs["auth"] = self._auth
        return request_kwargs
    
    def is_token_expired(self) -> bool:
        """
        Basic auth doesn't use tokens, so it never expires.
        
        Returns:
            False (Basic auth never expires)
        """
        return False
    
    def refresh_token(self) -> bool:
        """
        Basic auth doesn't use tokens, so no refresh is needed.
        
        Returns:
            True (Basic auth is always valid)
        """
        return True
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get basic auth headers.
        
        Returns:
            Dictionary with Authorization header
        """
        auth_str = f"{self.username}:{self.password}"
        encoded = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {encoded}"}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert basic auth to a dictionary.
        
        Returns:
            Dictionary representation of basic auth
        """
        return {
            "type": "basic",
            "username": self.username,
            "password": self.password
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasicAuth':
        """
        Create a basic auth instance from a dictionary.
        
        Args:
            data: Dictionary representation of basic auth
            
        Returns:
            BasicAuth instance
        """
        return cls(
            username=data.get("username", ""),
            password=data.get("password", "")
        )


class TokenAuth(AuthMethod):
    """Token-based authentication method."""
    
    def __init__(
        self,
        token: str,
        token_type: str = "Bearer",
        expiry: Optional[datetime] = None,
        refresh_token: Optional[str] = None,
        refresh_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ):
        """
        Initialize token authentication.
        
        Args:
            token: Authentication token
            token_type: Type of token (Bearer, JWT, etc.)
            expiry: Token expiry datetime
            refresh_token: Refresh token (if available)
            refresh_url: URL for refreshing the token
            client_id: Client ID for token refresh
            client_secret: Client secret for token refresh
        """
        self.token = token
        self.token_type = token_type
        self.expiry = expiry
        self.refresh_token = refresh_token
        self.refresh_url = refresh_url
        self.client_id = client_id
        self.client_secret = client_secret
    
    def apply_auth(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply token authentication to a request.
        
        Args:
            request_kwargs: Request arguments for requests.request
            
        Returns:
            Updated request arguments with authentication applied
        """
        if "headers" not in request_kwargs:
            request_kwargs["headers"] = {}
        
        request_kwargs["headers"].update(self.get_auth_headers())
        return request_kwargs
    
    def is_token_expired(self) -> bool:
        """
        Check if the token is expired.
        
        Returns:
            True if the token is expired, False otherwise
        """
        if not self.expiry:
            return False
        
        return datetime.now() > self.expiry
    
    def refresh_token(self) -> bool:
        """
        Refresh the authentication token if possible.
        
        Returns:
            True if the token was refreshed successfully, False otherwise
        """
        if not self.refresh_token or not self.refresh_url:
            logger.warning("Token refresh requested but no refresh token or URL available")
            return False
        
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token
            }
            
            # Add client credentials if available
            if self.client_id and self.client_secret:
                auth = (self.client_id, self.client_secret)
            else:
                auth = None
            
            response = requests.post(
                self.refresh_url,
                data=data,
                auth=auth,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                
                self.token = token_data.get("access_token")
                
                # Update refresh token if provided
                if "refresh_token" in token_data:
                    self.refresh_token = token_data["refresh_token"]
                
                # Update expiry if provided
                if "expires_in" in token_data:
                    self.expiry = datetime.now() + timedelta(seconds=token_data["expires_in"])
                
                logger.info("Token refreshed successfully, expires at %s", self.expiry)
                return True
            else:
                logger.error("Failed to refresh token: HTTP %d", response.status_code)
                return False
        
        except Exception as e:
            logger.exception("Error refreshing token: %s", str(e))
            return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get token auth headers.
        
        Returns:
            Dictionary with Authorization header
        """
        return {"Authorization": f"{self.token_type} {self.token}"}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert token auth to a dictionary.
        
        Returns:
            Dictionary representation of token auth
        """
        return {
            "type": "token",
            "token": self.token,
            "token_type": self.token_type,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "refresh_token": self.refresh_token,
            "refresh_url": self.refresh_url,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenAuth':
        """
        Create a token auth instance from a dictionary.
        
        Args:
            data: Dictionary representation of token auth
            
        Returns:
            TokenAuth instance
        """
        expiry = None
        if data.get("expiry"):
            try:
                expiry = datetime.fromisoformat(data["expiry"])
            except (ValueError, TypeError):
                pass
        
        return cls(
            token=data.get("token", ""),
            token_type=data.get("token_type", "Bearer"),
            expiry=expiry,
            refresh_token=data.get("refresh_token"),
            refresh_url=data.get("refresh_url"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret")
        )


class OAuth2Auth(AuthMethod):
    """OAuth2 authentication method."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token: Optional[Dict[str, Any]] = None,
        auth_url: Optional[str] = None,
        token_url: Optional[str] = None,
        refresh_url: Optional[str] = None,
        redirect_uri: str = "http://localhost:8080/callback",
        scope: Optional[List[str]] = None,
        grant_type: str = "authorization_code"
    ):
        """
        Initialize OAuth2 authentication.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            token: OAuth2 token dictionary
            auth_url: Authorization URL
            token_url: Token URL
            refresh_url: Refresh URL
            redirect_uri: Redirect URI for authorization flow
            scope: OAuth2 scopes
            grant_type: OAuth2 grant type
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = token or {}
        self.auth_url = auth_url
        self.token_url = token_url
        self.refresh_url = refresh_url or token_url
        self.redirect_uri = redirect_uri
        self.scope = scope or []
        self.grant_type = grant_type
        
        # Create OAuth2 session
        if self.grant_type == "authorization_code":
            self.client = WebApplicationClient(client_id)
            self.session = OAuth2Session(
                client_id=client_id,
                redirect_uri=redirect_uri,
                scope=scope,
                token=token
            )
        elif self.grant_type == "client_credentials":
            self.client = BackendApplicationClient(client_id=client_id)
            self.session = OAuth2Session(client=self.client)
        else:
            raise ValueError(f"Unsupported grant type: {grant_type}")
    
    def apply_auth(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply OAuth2 authentication to a request.
        
        Args:
            request_kwargs: Request arguments for requests.request
            
        Returns:
            Updated request arguments with authentication applied
        """
        if "headers" not in request_kwargs:
            request_kwargs["headers"] = {}
        
        request_kwargs["headers"].update(self.get_auth_headers())
        return request_kwargs
    
    def is_token_expired(self) -> bool:
        """
        Check if the OAuth2 token is expired.
        
        Returns:
            True if the token is expired, False otherwise
        """
        if not self.token:
            return True
        
        expires_at = self.token.get("expires_at")
        if not expires_at:
            return False
        
        # Add a buffer of 10 seconds to avoid timing issues
        return datetime.now().timestamp() > (expires_at - 10)
    
    def refresh_token(self) -> bool:
        """
        Refresh the OAuth2 token.
        
        Returns:
            True if the token was refreshed successfully, False otherwise
        """
        if not self.token or not self.refresh_url:
            logger.warning("Token refresh requested but no token or refresh URL available")
            return False
        
        try:
            if self.grant_type == "authorization_code":
                self.token = self.session.refresh_token(
                    self.refresh_url,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            elif self.grant_type == "client_credentials":
                self.token = self.session.fetch_token(
                    self.token_url,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            
            logger.info("OAuth2 token refreshed successfully")
            return True
        except Exception as e:
            logger.exception("Error refreshing OAuth2 token: %s", str(e))
            return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get OAuth2 auth headers.
        
        Returns:
            Dictionary with Authorization header
        """
        if not self.token:
            return {}
        
        return {"Authorization": f"Bearer {self.token.get('access_token', '')}"}
    
    def authorize(self) -> bool:
        """
        Perform OAuth2 authorization flow.
        
        Returns:
            True if authorization was successful, False otherwise
        """
        if not self.auth_url or not self.token_url:
            logger.error("Authorization URL or token URL not provided")
            return False
        
        try:
            if self.grant_type == "authorization_code":
                # Use a local server to handle the redirect
                authorization_url, state = self.session.authorization_url(self.auth_url)
                
                print(f"\nPlease go to the following URL to authorize the application:\n{authorization_url}\n")
                webbrowser.open(authorization_url)
                
                # Set up a local server to handle the redirect
                code = self._wait_for_redirect()
                
                if not code:
                    logger.error("Failed to get authorization code")
                    return False
                
                self.token = self.session.fetch_token(
                    self.token_url,
                    code=code,
                    client_secret=self.client_secret
                )
                
            elif self.grant_type == "client_credentials":
                self.token = self.session.fetch_token(
                    self.token_url,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            
            logger.info("OAuth2 authorization successful")
            return True
        
        except Exception as e:
            logger.exception("Error during OAuth2 authorization: %s", str(e))
            return False
    
    def _wait_for_redirect(self) -> Optional[str]:
        """
        Wait for OAuth2 redirect and extract the authorization code.
        
        Returns:
            Authorization code or None if failed
        """
        code_container = {"code": None}
        redirect_received = threading.Event()
        
        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = urllib.parse.urlparse(self.path).query
                query_components = urllib.parse.parse_qs(query)
                
                if "code" in query_components:
                    code_container["code"] = query_components["code"][0]
                    
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    
                    self.wfile.write(b"<html><head><title>Authorization Successful</title></head>")
                    self.wfile.write(b"<body><h1>Authorization Successful</h1>")
                    self.wfile.write(b"<p>You can close this window and return to the application.</p>")
                    self.wfile.write(b"</body></html>")
                else:
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    
                    self.wfile.write(b"<html><head><title>Authorization Failed</title></head>")
                    self.wfile.write(b"<body><h1>Authorization Failed</h1>")
                    self.wfile.write(b"<p>No authorization code received.</p>")
                    self.wfile.write(b"</body></html>")
                
                redirect_received.set()
            
            def log_message(self, format, *args):
                # Suppress log messages
                pass
        
        # Extract port from redirect URI
        redirect_uri_parts = urllib.parse.urlparse(self.redirect_uri)
        port = redirect_uri_parts.port or 8080
        
        # Start local server
        server = HTTPServer(("localhost", port), RedirectHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        try:
            # Wait for the redirect (timeout after 5 minutes)
            redirect_received.wait(300)
            return code_container["code"]
        finally:
            server.shutdown()
            server_thread.join()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert OAuth2 auth to a dictionary.
        
        Returns:
            Dictionary representation of OAuth2 auth
        """
        return {
            "type": "oauth2",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "token": self.token,
            "auth_url": self.auth_url,
            "token_url": self.token_url,
            "refresh_url": self.refresh_url,
            "redirect_uri": self.redirect_uri,
            "scope": self.scope,
            "grant_type": self.grant_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OAuth2Auth':
        """
        Create an OAuth2 auth instance from a dictionary.
        
        Args:
            data: Dictionary representation of OAuth2 auth
            
        Returns:
            OAuth2Auth instance
        """
        return cls(
            client_id=data.get("client_id", ""),
            client_secret=data.get("client_secret", ""),
            token=data.get("token"),
            auth_url=data.get("auth_url"),
            token_url=data.get("token_url"),
            refresh_url=data.get("refresh_url"),
            redirect_uri=data.get("redirect_uri", "http://localhost:8080/callback"),
            scope=data.get("scope"),
            grant_type=data.get("grant_type", "authorization_code")
        )


class ApiKeyAuth(AuthMethod):
    """API key authentication method."""
    
    def __init__(
        self,
        api_key: str,
        param_name: str = "api_key",
        location: str = "header",
        prefix: Optional[str] = None
    ):
        """
        Initialize API key authentication.
        
        Args:
            api_key: API key value
            param_name: Name of the parameter to use for the API key
            location: Where to place the API key (header, query, or cookie)
            prefix: Optional prefix for the API key value
        """
        self.api_key = api_key
        self.param_name = param_name
        self.location = location.lower()
        self.prefix = prefix
        
        if self.location not in ["header", "query", "cookie"]:
            raise ValueError("Location must be one of: header, query, cookie")
    
    def apply_auth(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply API key authentication to a request.
        
        Args:
            request_kwargs: Request arguments for requests.request
            
        Returns:
            Updated request arguments with authentication applied
        """
        if self.location == "header":
            if "headers" not in request_kwargs:
                request_kwargs["headers"] = {}
            
            value = f"{self.prefix} {self.api_key}" if self.prefix else self.api_key
            request_kwargs["headers"][self.param_name] = value
            
        elif self.location == "query":
            if "params" not in request_kwargs:
                request_kwargs["params"] = {}
            
            request_kwargs["params"][self.param_name] = self.api_key
            
        elif self.location == "cookie":
            if "cookies" not in request_kwargs:
                request_kwargs["cookies"] = {}
            
            request_kwargs["cookies"][self.param_name] = self.api_key
        
        return request_kwargs
    
    def is_token_expired(self) -> bool:
        """
        API keys don't expire.
        
        Returns:
            False (API keys don't expire)
        """
        return False
    
    def refresh_token(self) -> bool:
        """
        API keys can't be refreshed.
        
        Returns:
            True (API keys don't need refreshing)
        """
        return True
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get API key auth headers.
        
        Returns:
            Dictionary with API key header (if location is header)
        """
        if self.location == "header":
            value = f"{self.prefix} {self.api_key}" if self.prefix else self.api_key
            return {self.param_name: value}
        
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert API key auth to a dictionary.
        
        Returns:
            Dictionary representation of API key auth
        """
        return {
            "type": "apikey",
            "api_key": self.api_key,
            "param_name": self.param_name,
            "location": self.location,
            "prefix": self.prefix
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiKeyAuth':
        """
        Create an API key auth instance from a dictionary.
        
        Args:
            data: Dictionary representation of API key auth
            
        Returns:
            ApiKeyAuth instance
        """
        return cls(
            api_key=data.get("api_key", ""),
            param_name=data.get("param_name", "api_key"),
            location=data.get("location", "header"),
            prefix=data.get("prefix")
        )


class CredentialManager:
    """
    Secure credential manager for storing and retrieving authentication credentials.
    
    This class provides methods for securely storing and retrieving credentials
    using either the system keyring or encrypted files.
    """
    
    def __init__(self, service_name: str = "webscraper", use_keyring: bool = True):
        """
        Initialize the credential manager.
        
        Args:
            service_name: Service name for keyring
            use_keyring: Whether to use system keyring for credential storage
        """
        self.service_name = service_name
        self.use_keyring = use_keyring
        
        # Set up encryption for file-based storage
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Set up encryption for file-based credential storage."""
        # Use a fixed salt for derivation
        salt = b'webscraper_salt_2023'
        
        # Try to get or create an encryption key
        try:
            key_path = Path.home() / ".webscraper" / "key"
            key_path.parent.mkdir(parents=True, exist_ok=True)
            
            if key_path.exists():
                with open(key_path, "rb") as f:
                    key_material = f.read()
            else:
                # Generate a random key
                key_material = os.urandom(32)
                with open(key_path, "wb") as f:
                    f.write(key_material)
                
                # Restrict permissions to the current user
                os.chmod(key_path, 0o600)
            
            # Derive a key for Fernet encryption
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key_material))
            self.cipher = Fernet(key)
            
        except Exception as e:
            logger.exception("Error setting up encryption: %s", str(e))
            self.cipher = None
    
    def store_credentials(self, domain: str, credentials: Dict[str, Any]) -> bool:
        """
        Store credentials securely.
        
        Args:
            domain: Domain for which to store credentials
            credentials: Credential dictionary
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            if self.use_keyring:
                # Store credentials in system keyring
                keyring.set_password(
                    self.service_name,
                    domain,
                    json.dumps(credentials)
                )
            else:
                # Store credentials in encrypted file
                if not self.cipher:
                    logger.error("Encryption not available for file-based storage")
                    return False
                
                creds_dir = Path.home() / ".webscraper" / "credentials"
                creds_dir.mkdir(parents=True, exist_ok=True)
                
                creds_file = creds_dir / f"{domain}.enc"
                
                # Encrypt credentials
                encrypted = self.cipher.encrypt(json.dumps(credentials).encode("utf-8"))
                
                with open(creds_file, "wb") as f:
                    f.write(encrypted)
                
                # Restrict permissions to the current user
                os.chmod(creds_file, 0o600)
            
            return True
        
        except Exception as e:
            logger.exception("Error storing credentials: %s", str(e))
            return False
    
    def get_credentials(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve credentials securely.
        
        Args:
            domain: Domain for which to retrieve credentials
            
        Returns:
            Credential dictionary or None if not found
        """
        try:
            if self.use_keyring:
                # Retrieve credentials from system keyring
                creds_json = keyring.get_password(self.service_name, domain)
                
                if creds_json:
                    return json.loads(creds_json)
                
                return None
            else:
                # Retrieve credentials from encrypted file
                if not self.cipher:
                    logger.error("Encryption not available for file-based storage")
                    return None
                
                creds_file = Path.home() / ".webscraper" / "credentials" / f"{domain}.enc"
                
                if not creds_file.exists():
                    return None
                
                with open(creds_file, "rb") as f:
                    encrypted = f.read()
                
                # Decrypt credentials
                decrypted = self.cipher.decrypt(encrypted).decode("utf-8")
                return json.loads(decrypted)
        
        except Exception as e:
            logger.exception("Error retrieving credentials: %s", str(e))
            return None
    
    def delete_credentials(self, domain: str) -> bool:
        """
        Delete stored credentials.
        
        Args:
            domain: Domain for which to delete credentials
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if self.use_keyring:
                # Delete credentials from system keyring
                keyring.delete_password(self.service_name, domain)
            else:
                # Delete credentials file
                creds_file = Path.home() / ".webscraper" / "credentials" / f"{domain}.enc"
                
                if creds_file.exists():
                    os.remove(creds_file)
            
            return True
        
        except Exception as e:
            logger.exception("Error deleting credentials: %s", str(e))
            return False
    
    def list_domains(self) -> List[str]:
        """
        List domains with stored credentials.
        
        Returns:
            List of domains with stored credentials
        """
        try:
            if not self.use_keyring:
                # List credential files
                creds_dir = Path.home() / ".webscraper" / "credentials"
                
                if not creds_dir.exists():
                    return []
                
                return [f.stem for f in creds_dir.glob("*.enc")]
            
            # Keyring doesn't provide a way to list all stored credentials
            # This is a limitation of the keyring API
            return []
        
        except Exception as e:
            logger.exception("Error listing credential domains: %s", str(e))
            return []


class AuthHandlerConfig(BaseModel):
    """Configuration specific to the authentication handler."""
    
    # Credential storage
    use_keyring: bool = True
    credential_service_name: str = "webscraper"
    
    # Session management
    session_timeout: int = 3600  # 1 hour
    verify_ssl: bool = True
    
    # Proxy authentication
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    
    # Token refresh
    auto_refresh: bool = True
    refresh_before_expiry: int = 300  # 5 minutes


class AuthHandler(BaseScraper):
    """
    Authentication handler for web scraping.
    
    This class provides functionality for handling various authentication methods,
    managing sessions, and storing credentials securely.
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ScraperConfig]] = None,
        auth_config: Optional[Union[Dict[str, Any], AuthHandlerConfig]] = None
    ):
        """
        Initialize the authentication handler.
        
        Args:
            config: General scraper configuration
            auth_config: Authentication handler specific configuration
        """
        super().__init__(config)
        
        # Set up authentication configuration
        if auth_config is None:
            self.auth_config = AuthHandlerConfig()
        elif isinstance(auth_config, dict):
            self.auth_config = AuthHandlerConfig.model_validate(auth_config)
        else:
            self.auth_config = auth_config
        
        # Set up credential manager
        self.credential_manager = CredentialManager(
            service_name=self.auth_config.credential_service_name,
            use_keyring=self.auth_config.use_keyring
        )
        
        # Authentication methods by domain
        self.auth_methods: Dict[str, AuthMethod] = {}
        
        # Session management
        self.session_start = datetime.now()
        
        logger.info("Authentication handler initialized with configuration: %s", self.auth_config.model_dump())
    
    def scrape(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Authenticate with the specified URL and test the authentication.
        
        Args:
            url: URL to authenticate with
            **kwargs: Additional arguments for authentication
            
        Returns:
            Dictionary with authentication results
        """
        domain = self._get_domain(url)
        
        # Get existing authentication method or create a new one
        auth_method = self._get_auth_method(domain)
        
        # Test authentication
        if auth_method:
            status, data = self._test_authentication(url, auth_method)
            
            return {
                "url": url,
                "domain": domain,
                "authenticated": status < 400,
                "status": status,
                "response": data
            }
        else:
            return {
                "url": url,
                "domain": domain,
                "authenticated": False,
                "error": "No authentication method available for this domain"
            }
    
    def authenticate(
        self,
        url: str,
        auth_type: str,
        credentials: Dict[str, Any],
        store_credentials: bool = True
    ) -> bool:
        """
        Authenticate with the specified URL using the provided credentials.
        
        Args:
            url: URL to authenticate with
            auth_type: Authentication type (basic, token, oauth2, apikey)
            credentials: Authentication credentials
            store_credentials: Whether to store credentials securely
            
        Returns:
            True if authentication was successful, False otherwise
        """
        domain = self._get_domain(url)
        
        try:
            # Create authentication method based on type
            if auth_type.lower() == "basic":
                auth_method = BasicAuth(
                    username=credentials.get("username", ""),
                    password=credentials.get("password", "")
                )
            
            elif auth_type.lower() == "token":
                # Parse expiry datetime if provided
                expiry = None
                if "expiry" in credentials:
                    try:
                        expiry = datetime.fromisoformat(credentials["expiry"])
                    except (ValueError, TypeError):
                        pass
                
                auth_method = TokenAuth(
                    token=credentials.get("token", ""),
                    token_type=credentials.get("token_type", "Bearer"),
                    expiry=expiry,
                    refresh_token=credentials.get("refresh_token"),
                    refresh_url=credentials.get("refresh_url"),
                    client_id=credentials.get("client_id"),
                    client_secret=credentials.get("client_secret")
                )
            
            elif auth_type.lower() == "oauth2":
                auth_method = OAuth2Auth(
                    client_id=credentials.get("client_id", ""),
                    client_secret=credentials.get("client_secret", ""),
                    token=credentials.get("token"),
                    auth_url=credentials.get("auth_url"),
                    token_url=credentials.get("token_url"),
                    refresh_url=credentials.get("refresh_url"),
                    redirect_uri=credentials.get("redirect_uri", "http://localhost:8080/callback"),
                    scope=credentials.get("scope"),
                    grant_type=credentials.get("grant_type", "authorization_code")
                )
                
                # If no token is provided, perform authorization flow
                if not credentials.get("token"):
                    if not auth_method.authorize():
                        logger.error("OAuth2 authorization failed")
                        return False
            
            elif auth_type.lower() == "apikey":
                auth_method = ApiKeyAuth(
                    api_key=credentials.get("api_key", ""),
                    param_name=credentials.get("param_name", "api_key"),
                    location=credentials.get("location", "header"),
                    prefix=credentials.get("prefix")
                )
            
            else:
                logger.error("Unsupported authentication type: %s", auth_type)
                return False
            
            # Test authentication
            status, _ = self._test_authentication(url, auth_method)
            
            if status < 400:
                # Store authentication method
                self.auth_methods[domain] = auth_method
                
                # Store credentials if requested
                if store_credentials:
                    self.credential_manager.store_credentials(
                        domain=domain,
                        credentials={
                            "type": auth_type.lower(),
                            "credentials": auth_method.to_dict()
                        }
                    )
                
                logger.info("Authentication successful for %s using %s", domain, auth_type)
                return True
            else:
                logger.error("Authentication failed for %s: HTTP %d", domain, status)
                return False
        
        except Exception as e:
            logger.exception("Error during authentication: %s", str(e))
            return False
    
    def apply_auth(self, request_kwargs: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Apply authentication to a request.
        
        Args:
            request_kwargs: Request arguments for requests.request
            url: URL to authenticate with
            
        Returns:
            Updated request arguments with authentication applied
        """
        domain = self._get_domain(url)
        auth_method = self._get_auth_method(domain)
        
        if auth_method:
            # Check if token needs to be refreshed
            if self.auth_config.auto_refresh and auth_method.is_token_expired():
                logger.info("Token expired for %s, attempting refresh", domain)
                auth_method.refresh_token()
            
            # Apply authentication
            request_kwargs = auth_method.apply_auth(request_kwargs)
            
            # Apply proxy authentication if configured
            if self.auth_config.proxy_username and self.auth_config.proxy_password:
                if "proxies" in request_kwargs:
                    for protocol, proxy_url in request_kwargs["proxies"].items():
                        if isinstance(proxy_url, str) and "://" in proxy_url:
                            parts = proxy_url.split("://")
                            auth_proxy = f"{parts[0]}://{self.auth_config.proxy_username}:{self.auth_config.proxy_password}@{parts[1]}"
                            request_kwargs["proxies"][protocol] = auth_proxy
        
        # Set SSL verification
        request_kwargs["verify"] = self.auth_config.verify_ssl
        
        return request_kwargs
    
    def _test_authentication(self, url: str, auth_method: AuthMethod) -> Tuple[int, Any]:
        """
        Test authentication with the specified URL.
        
        Args:
            url: URL to test authentication with
            auth_method: Authentication method to test
            
        Returns:
            Tuple of (status_code, response_data)
        """
        try:
            # Apply authentication to request
            request_kwargs = {
                "method": "GET",
                "url": url,
                "timeout": 30,
                "allow_redirects": True
            }
            
            request_kwargs = auth_method.apply_auth(request_kwargs)
            
            # Apply proxy authentication if configured
            if self.auth_config.proxy_username and self.auth_config.proxy_password:
                proxy_auth = requests.auth.HTTPProxyAuth(
                    self.auth_config.proxy_username,
                    self.auth_config.proxy_password
                )
                request_kwargs["auth"] = proxy_auth
            
            # Make request
            response = requests.request(**request_kwargs)
            
            # Try to parse response data
            try:
                data = response.json()
            except:
                data = response.text
            
            return response.status_code, data
        
        except Exception as e:
            logger.exception("Error testing authentication: %s", str(e))
            return 500, {"error": str(e)}
    
    def _get_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        parsed_url = urllib.parse.urlparse(url)
        return parsed_url.netloc
    
    def _get_auth_method(self, domain: str) -> Optional[AuthMethod]:
        """
        Get authentication method for the specified domain.
        
        If no authentication method is found in memory, try to load it from stored credentials.
        
        Args:
            domain: Domain to get authentication method for
            
        Returns:
            Authentication method or None if not found
        """
        # Check if we already have an authentication method for this domain
        if domain in self.auth_methods:
            return self.auth_methods[domain]
        
        # Try to load from stored credentials
        stored_creds = self.credential_manager.get_credentials(domain)
        
        if stored_creds:
            auth_type = stored_creds.get("type")
            creds = stored_creds.get("credentials", {})
            
            if auth_type == "basic":
                auth_method = BasicAuth.from_dict(creds)
            elif auth_type == "token":
                auth_method = TokenAuth.from_dict(creds)
            elif auth_type == "oauth2":
                auth_method = OAuth2Auth.from_dict(creds)
            elif auth_type == "apikey":
                auth_method = ApiKeyAuth.from_dict(creds)
            else:
                return None
            
            # Store in memory
            self.auth_methods[domain] = auth_method
            
            return auth_method
        
        return None
    
    def is_session_expired(self) -> bool:
        """
        Check if the current session is expired.
        
        Returns:
            True if the session is expired, False otherwise
        """
        elapsed = (datetime.now() - self.session_start).total_seconds()
        return elapsed > self.auth_config.session_timeout
    
    def refresh_session(self) -> None:
        """Refresh the current session by updating the session start time."""
        self.session_start = datetime.now()
    
    def clear_session(self) -> None:
        """Clear all authentication methods and session data."""
        self.auth_methods.clear()
        self.session_start = datetime.now()
    
    def list_authenticated_domains(self) -> List[str]:
        """
        List all domains with active authentication.
        
        Returns:
            List of authenticated domains
        """
        return list(self.auth_methods.keys())
    
    def export_auth_config(self, filename: str) -> bool:
        """
        Export authentication configuration to a file.
        
        This exports only the configuration, not the credentials.
        
        Args:
            filename: File to export configuration to
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            config_data = {
                "auth_config": self.auth_config.model_dump(),
                "domains": self.list_authenticated_domains()
            }
            
            # Determine format based on file extension
            _, ext = os.path.splitext(filename)
            
            if ext.lower() == ".json":
                with open(filename, "w") as f:
                    json.dump(config_data, f, indent=2)
            elif ext.lower() in [".yaml", ".yml"]:
                with open(filename, "w") as f:
                    yaml.dump(config_data, f)
            else:
                logger.error("Unsupported file format: %s", ext)
                return False
            
            return True
        
        except Exception as e:
            logger.exception("Error exporting authentication configuration: %s", str(e))
            return False
    
    def import_auth_config(self, filename: str) -> bool:
        """
        Import authentication configuration from a file.
        
        Args:
            filename: File to import configuration from
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            # Determine format based on file extension
            _, ext = os.path.splitext(filename)
            
            if ext.lower() == ".json":
                with open(filename, "r") as f:
                    config_data = json.load(f)
            elif ext.lower() in [".yaml", ".yml"]:
                with open(filename, "r") as f:
                    config_data = yaml.safe_load(f)
            else:
                logger.error("Unsupported file format: %s", ext)
                return False
            
            # Update configuration
            if "auth_config" in config_data:
                self.auth_config = AuthHandlerConfig.model_validate(config_data["auth_config"])
            
            return True
        
        except Exception as e:
            logger.exception("Error importing authentication configuration: %s", str(e))
            return False
    
    def prompt_for_credentials(
        self,
        auth_type: str,
        domain: str = "",
        secure_input: bool = True
    ) -> Dict[str, Any]:
        """
        Prompt the user for authentication credentials.
        
        Args:
            auth_type: Authentication type (basic, token, oauth2, apikey)
            domain: Domain to authenticate with (for display purposes)
            secure_input: Whether to hide sensitive input like passwords
            
        Returns:
            Dictionary with entered credentials
        """
        credentials = {}
        domain_str = f" for {domain}" if domain else ""
        
        print(f"\nEnter {auth_type.upper()} authentication credentials{domain_str}:\n")
        
        if auth_type.lower() == "basic":
            username = input("Username: ")
            
            if secure_input:
                password = getpass.getpass("Password: ")
            else:
                password = input("Password: ")
            
            credentials = {
                "username": username,
                "password": password
            }
        
        elif auth_type.lower() == "token":
            if secure_input:
                token = getpass.getpass("Token: ")
            else:
                token = input("Token: ")
            
            token_type = input("Token type (default: Bearer): ") or "Bearer"
            
            credentials = {
                "token": token,
                "token_type": token_type
            }
            
            # Ask if refresh token is available
            has_refresh = input("Do you have a refresh token? (y/n): ").lower() == "y"
            
            if has_refresh:
                if secure_input:
                    refresh_token = getpass.getpass("Refresh token: ")
                else:
                    refresh_token = input("Refresh token: ")
                
                refresh_url = input("Refresh URL: ")
                
                credentials.update({
                    "refresh_token": refresh_token,
                    "refresh_url": refresh_url
                })
                
                # Ask for client credentials if needed
                has_client_creds = input("Do you need client credentials for token refresh? (y/n): ").lower() == "y"
                
                if has_client_creds:
                    client_id = input("Client ID: ")
                    
                    if secure_input:
                        client_secret = getpass.getpass("Client secret: ")
                    else:
                        client_secret = input("Client secret: ")
                    
                    credentials.update({
                        "client_id": client_id,
                        "client_secret": client_secret
                    })
        
        elif auth_type.lower() == "oauth2":
            client_id = input("Client ID: ")
            
            if secure_input:
                client_secret = getpass.getpass("Client secret: ")
            else:
                client_secret = input("Client secret: ")
            
            auth_url = input("Authorization URL: ")
            token_url = input("Token URL: ")
            redirect_uri = input("Redirect URI (default: http://localhost:8080/callback): ") or "http://localhost:8080/callback"
            
            # Get scopes
            scope_input = input("Scopes (space-separated): ")
            scope = scope_input.split() if scope_input else []
            
            # Get grant type
            grant_type = input("Grant type (authorization_code or client_credentials, default: authorization_code): ") or "authorization_code"
            
            credentials = {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_url": auth_url,
                "token_url": token_url,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "grant_type": grant_type
            }
        
        elif auth_type.lower() == "apikey":
            if secure_input:
                api_key = getpass.getpass("API key: ")
            else:
                api_key = input("API key: ")
            
            param_name = input("Parameter name (default: api_key): ") or "api_key"
            location = input("Location (header, query, or cookie, default: header): ") or "header"
            prefix = input("Prefix (optional): ")
            
            credentials = {
                "api_key": api_key,
                "param_name": param_name,
                "location": location
            }
            
            if prefix:
                credentials["prefix"] = prefix
        
        return credentials

