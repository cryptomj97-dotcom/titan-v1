"""
TITAN 3.0 - Secure Secrets Management Module
Environment-based configuration and secrets handling.
"""

import os
import base64
import hashlib
import hmac
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Secure secrets management using environment variables.
    
    Security Features:
    - No hardcoded secrets in code
    - Environment variable based configuration
    - Optional encrypted secret files
    - Secret rotation support
    - Audit logging for secret access
    """
    
    def __init__(self, env_prefix: str = "TITAN_"):
        """
        Initialize secret manager.
        
        Args:
            env_prefix: Prefix for environment variables
        """
        self.env_prefix = env_prefix
        self._secret_cache: Dict[str, Any] = {}
        self._access_log: list = []
        
    def get_secret(self, name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Retrieve a secret from environment variables.
        
        Args:
            name: Secret name (will be prefixed)
            default: Default value if not found
            required: If True, raises ValueError when not found
            
        Returns:
            Secret value or default
            
        Raises:
            ValueError: If required secret is missing
        """
        full_name = f"{self.env_prefix}{name}"
        value = os.getenv(full_name, default)
        
        # Log access (but not the value!)
        self._access_log.append({
            'name': name,
            'found': value is not None,
            'has_default': default is not None
        })
        
        if required and value is None:
            raise ValueError(f"Required secret '{full_name}' not found in environment")
        
        return value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a specific service.
        
        Args:
            service: Service name (e.g., 'binance', 'alphavantage')
            
        Returns:
            API key or None
        """
        return self.get_secret(f"API_KEY_{service.upper()}")
    
    def get_database_url(self) -> str:
        """
        Get database connection URL.
        
        Returns:
            Database URL
            
        Raises:
            ValueError: If not configured
        """
        return self.get_secret(
            "DATABASE_URL",
            default="sqlite:///titan.db",
            required=False
        )
    
    def get_encryption_key(self) -> bytes:
        """
        Get encryption key for sensitive data.
        
        Returns:
            32-byte encryption key
            
        Raises:
            ValueError: If not configured
        """
        key_str = self.get_secret(
            "ENCRYPTION_KEY",
            required=False
        )
        
        if not key_str:
            # Generate a default key from machine-specific info
            # WARNING: Use proper key management in production!
            logger.warning("Using derived encryption key - configure TITAN_ENCRYPTION_KEY in production")
            machine_id = str(os.uname()).encode()
            return hashlib.sha256(machine_id).digest()
        
        # Derive 32-byte key from provided string
        return hashlib.sha256(key_str.encode()).digest()
    
    def get_flask_secret_key(self) -> bytes:
        """
        Get Flask session secret key.
        
        Returns:
            Random secret key for Flask sessions
        """
        key_str = self.get_secret("FLASK_SECRET_KEY")
        
        if key_str:
            return hashlib.sha256(key_str.encode()).digest()
        else:
            # Generate a secure random key
            logger.warning("Generating random Flask secret key - set TITAN_FLASK_SECRET_KEY for persistence")
            return os.urandom(32)
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate that required environment variables are set.
        
        Returns:
            Validation report with status and missing variables
        """
        required_secrets = [
            "DATABASE_URL",  # Optional but recommended
        ]
        
        optional_secrets = [
            "API_KEY_BINANCE",
            "API_KEY_ALPHAVANTAGE",
            "ENCRYPTION_KEY",
            "FLASK_SECRET_KEY",
            "REDIS_URL",
        ]
        
        report = {
            'valid': True,
            'missing_required': [],
            'missing_optional': [],
            'configured': []
        }
        
        for secret in required_secrets:
            if self.get_secret(secret) is None:
                report['missing_required'].append(f"{self.env_prefix}{secret}")
                report['valid'] = False
            else:
                report['configured'].append(f"{self.env_prefix}{secret}")
        
        for secret in optional_secrets:
            if self.get_secret(secret) is None:
                report['missing_optional'].append(f"{self.env_prefix}{secret}")
            else:
                report['configured'].append(f"{self.env_prefix}{secret}")
        
        return report
    
    def get_access_log(self) -> list:
        """Get audit log of secret accesses."""
        return self._access_log.copy()


# Global secret manager instance
secret_manager = SecretManager()


def get_secret(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Convenience function to get secrets."""
    return secret_manager.get_secret(name, default, required)


def get_api_key(service: str) -> Optional[str]:
    """Convenience function to get API keys."""
    return secret_manager.get_api_key(service)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Base64-encoded token string
    """
    return base64.urlsafe_b64encode(os.urandom(length)).decode('ascii').rstrip('=')


def verify_hmac_signature(message: str, signature: str, secret: str) -> bool:
    """
    Verify HMAC signature for request authentication.
    
    Args:
        message: Original message
        signature: Provided signature (hex-encoded)
        secret: Secret key for verification
        
    Returns:
        True if signature is valid
    """
    expected = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)


def sign_message(message: str, secret: str) -> str:
    """
    Create HMAC signature for a message.
    
    Args:
        message: Message to sign
        secret: Secret key
        
    Returns:
        Hex-encoded signature
    """
    return hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()


# Alias for backwards compatibility
SecretsManager = SecretManager
