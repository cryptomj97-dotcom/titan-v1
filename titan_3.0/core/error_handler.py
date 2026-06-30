"""
TITAN 3.0 - Secure Error Handling Module
Comprehensive error handling with security-conscious logging and responses.
"""

import traceback
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base class for security-related errors."""
    pass


class ValidationError(SecurityError):
    """Input validation failed."""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed."""
    pass


class AuthorizationError(SecurityError):
    """Authorization failed."""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded."""
    pass


class TITANErrorHandler:
    """
    Centralized error handling with security best practices.
    
    Features:
    - Safe error messages (no internal details leaked)
    - Comprehensive internal logging
    - Error categorization
    - Automatic alerting for critical errors
    """
    
    # Error codes for client responses
    ERROR_CODES = {
        'VALIDATION_ERROR': 400,
        'AUTHENTICATION_ERROR': 401,
        'AUTHORIZATION_ERROR': 403,
        'NOT_FOUND': 404,
        'RATE_LIMIT': 429,
        'INTERNAL_ERROR': 500,
        'SERVICE_UNAVAILABLE': 503,
    }
    
    # Safe messages for client responses
    SAFE_MESSAGES = {
        'VALIDATION_ERROR': 'Invalid input provided',
        'AUTHENTICATION_ERROR': 'Authentication required',
        'AUTHORIZATION_ERROR': 'Access denied',
        'NOT_FOUND': 'Resource not found',
        'RATE_LIMIT': 'Too many requests, please try again later',
        'INTERNAL_ERROR': 'An unexpected error occurred',
        'SERVICE_UNAVAILABLE': 'Service temporarily unavailable',
    }
    
    def __init__(self, log_level: int = logging.ERROR, 
                 alert_on_critical: bool = True):
        """
        Initialize error handler.
        
        Args:
            log_level: Logging level for errors
            alert_on_critical: Whether to alert on critical errors
        """
        self.log_level = log_level
        self.alert_on_critical = alert_on_critical
        self._error_count = 0
        self._critical_errors = []
    
    def handle_error(self, error: Exception, context: Dict = None,
                    is_critical: bool = False) -> Dict[str, Any]:
        """
        Handle an error and return safe response.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            is_critical: Whether this is a critical error
            
        Returns:
            Safe error response dictionary
        """
        self._error_count += 1
        
        # Log full error internally
        self._log_error(error, context, is_critical)
        
        # Categorize error
        error_type = self._categorize_error(error)
        
        # Get safe message
        safe_message = self.SAFE_MESSAGES.get(
            error_type, 
            'An unexpected error occurred'
        )
        
        # Build safe response
        response = {
            'error': safe_message,
            'error_code': error_type,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Add error ID for tracking (but don't expose internals)
        error_id = f"ERR-{self._error_count:06d}"
        response['error_id'] = error_id
        
        # Track critical errors
        if is_critical:
            self._critical_errors.append({
                'id': error_id,
                'type': str(type(error).__name__),
                'timestamp': datetime.utcnow(),
                'context': self._sanitize_context(context)
            })
            
            # Alert if configured
            if self.alert_on_critical:
                self._alert_critical(error, context, error_id)
        
        return response
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for appropriate response."""
        error_name = type(error).__name__
        
        if isinstance(error, (ValidationError, ValueError, TypeError)):
            return 'VALIDATION_ERROR'
        elif isinstance(error, AuthenticationError):
            return 'AUTHENTICATION_ERROR'
        elif isinstance(error, AuthorizationError):
            return 'AUTHORIZATION_ERROR'
        elif isinstance(error, RateLimitError):
            return 'RATE_LIMIT'
        elif isinstance(error, FileNotFoundError):
            return 'NOT_FOUND'
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return 'SERVICE_UNAVAILABLE'
        else:
            return 'INTERNAL_ERROR'
    
    def _log_error(self, error: Exception, context: Dict = None,
                   is_critical: bool = False) -> None:
        """Log full error details internally."""
        log_level = logging.CRITICAL if is_critical else self.log_level
        
        # Build detailed log message
        log_parts = [
            f"Error: {type(error).__name__}",
            f"Message: {str(error)}",
            f"Traceback:\n{traceback.format_exc()}"
        ]
        
        if context:
            sanitized = self._sanitize_context(context)
            log_parts.append(f"Context: {sanitized}")
        
        log_message = "\n".join(log_parts)
        
        logger.log(log_level, log_message)
    
    def _sanitize_context(self, context: Dict) -> Dict:
        """Remove sensitive data from context for logging."""
        if not context:
            return {}
        
        sensitive_keys = {
            'password', 'secret', 'token', 'api_key', 'apikey',
            'authorization', 'credential', 'private_key'
        }
        
        sanitized = {}
        for key, value in context.items():
            if key.lower() in sensitive_keys:
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_context(value)
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long strings
                sanitized[key] = value[:100] + '...'
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _alert_critical(self, error: Exception, context: Dict,
                        error_id: str) -> None:
        """Send alert for critical errors."""
        # In production, integrate with monitoring systems
        alert_message = (
            f"CRITICAL ERROR ALERT\n"
            f"Error ID: {error_id}\n"
            f"Type: {type(error).__name__}\n"
            f"Time: {datetime.utcnow().isoformat()}\n"
            f"Details: {str(error)}"
        )
        
        # Log alert
        logger.critical(alert_message)
        
        # TODO: Integrate with external alerting (PagerDuty, Slack, etc.)
        # send_to_monitoring_system(alert_message)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'total_errors': self._error_count,
            'critical_errors': len(self._critical_errors),
            'recent_critical': self._critical_errors[-10:]  # Last 10
        }
    
    def clear_critical_log(self) -> None:
        """Clear critical error log."""
        self._critical_errors.clear()


# Global error handler instance
error_handler = TITANErrorHandler()


def safe_error_response(error: Exception, context: Dict = None) -> tuple:
    """
    Convenience function for Flask routes.
    
    Args:
        error: Exception that occurred
        context: Optional context
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    result = error_handler.handle_error(error, context)
    error_type = result.get('error_code', 'INTERNAL_ERROR')
    status_code = TITANErrorHandler.ERROR_CODES.get(error_type, 500)
    return result, status_code


def create_error_response(message: str, error_code: str = 'VALIDATION_ERROR',
                          status_code: int = 400) -> tuple:
    """
    Create a standardized error response.
    
    Args:
        message: Error message (will be used safely)
        error_code: Error code identifier
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'error': message,
        'error_code': error_code,
        'timestamp': datetime.utcnow().isoformat(),
    }
    return response, status_code


def catch_errors(func):
    """
    Decorator to catch and handle errors in Flask routes.
    
    Usage:
        @app.route('/api/endpoint')
        @catch_errors
        def endpoint():
            ...
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            response, status = safe_error_response(e)
            # For Flask, we need to return the response properly
            from flask import jsonify
            return jsonify(response), status
    
    return wrapper
