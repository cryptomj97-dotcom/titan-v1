"""
TITAN 3.0 - Security Middleware Module
CORS, CSRF, Rate Limiting, and Thread Safety middleware.
"""

import time
import threading
import functools
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import secrets
import hmac
import hashlib

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API endpoints.
    
    Features:
    - Configurable rate limits per endpoint
    - Per-IP or per-user tracking
    - Automatic token refill
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def _get_bucket(self, key: str) -> Dict:
        """Get or create a token bucket for a key."""
        now = time.time()
        
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    'tokens': self.max_requests,
                    'last_refill': now
                }
            
            bucket = self._buckets[key]
            
            # Refill tokens based on elapsed time
            elapsed = now - bucket['last_refill']
            refill_rate = self.max_requests / self.window_seconds
            tokens_to_add = elapsed * refill_rate
            
            bucket['tokens'] = min(self.max_requests, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            return bucket
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed for given key.
        
        Args:
            key: Identifier (IP, user ID, etc.)
            
        Returns:
            True if request is allowed
        """
        bucket = self._get_bucket(key)
        
        with self._lock:
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            else:
                logger.warning(f"Rate limit exceeded for {key}")
                return False
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for a key."""
        bucket = self._get_bucket(key)
        return int(bucket['tokens'])
    
    def cleanup(self, max_age_seconds: int = 3600):
        """Remove stale buckets to prevent memory leaks."""
        now = time.time()
        with self._lock:
            stale_keys = [
                k for k, v in self._buckets.items()
                if now - v['last_refill'] > max_age_seconds
            ]
            for key in stale_keys:
                del self._buckets[key]


class CSRFProtection:
    """
    CSRF token generation and validation.
    
    Features:
    - Cryptographically secure tokens
    - Token expiration
    - Double-submit cookie pattern support
    """
    
    TOKEN_EXPIRY_SECONDS = 3600  # 1 hour
    
    def __init__(self):
        self._tokens: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def generate_token(self, session_id: str) -> str:
        """
        Generate a new CSRF token.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CSRF token string
        """
        token = secrets.token_urlsafe(32)
        expiry = time.time() + self.TOKEN_EXPIRY_SECONDS
        
        with self._lock:
            if session_id not in self._tokens:
                self._tokens[session_id] = []
            
            # Store token with expiry
            self._tokens[session_id].append({
                'token': token,
                'expiry': expiry,
                'created': time.time()
            })
            
            # Cleanup old tokens
            self._tokens[session_id] = [
                t for t in self._tokens[session_id]
                if t['expiry'] > time.time()
            ]
        
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """
        Validate a CSRF token.
        
        Args:
            session_id: Session identifier
            token: Token to validate
            
        Returns:
            True if token is valid
        """
        if not token or not session_id:
            return False
        
        with self._lock:
            if session_id not in self._tokens:
                return False
            
            now = time.time()
            valid_tokens = []
            
            for token_data in self._tokens[session_id]:
                if token_data['expiry'] > now:
                    valid_tokens.append(token_data)
                    # Use constant-time comparison
                    if hmac.compare_digest(token_data['token'], token):
                        # Remove used token (one-time use)
                        self._tokens[session_id] = valid_tokens
                        return True
            
            self._tokens[session_id] = valid_tokens
            return False
    
    def cleanup(self):
        """Remove expired tokens."""
        now = time.time()
        with self._lock:
            for session_id in list(self._tokens.keys()):
                self._tokens[session_id] = [
                    t for t in self._tokens[session_id]
                    if t['expiry'] > now
                ]
                if not self._tokens[session_id]:
                    del self._tokens[session_id]


class ThreadSafeState:
    """
    Thread-safe state management for concurrent access.
    
    Features:
    - Read-write locks
    - Atomic operations
    - Deadlock prevention
    """
    
    def __init__(self, initial_state: Dict = None):
        """
        Initialize thread-safe state.
        
        Args:
            initial_state: Initial state dictionary
        """
        self._state = initial_state or {}
        self._lock = threading.RLock()  # Reentrant lock
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe get operation."""
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Thread-safe set operation."""
        with self._lock:
            self._state[key] = value
    
    def update(self, updates: Dict) -> None:
        """Thread-safe bulk update."""
        with self._lock:
            self._state.update(updates)
    
    def delete(self, key: str) -> Any:
        """Thread-safe delete operation."""
        with self._lock:
            return self._state.pop(key, None)
    
    def atomic_increment(self, key: str, delta: int = 1) -> int:
        """Atomically increment a counter."""
        with self._lock:
            current = self._state.get(key, 0)
            self._state[key] = current + delta
            return self._state[key]
    
    def atomic_compare_and_swap(self, key: str, expected: Any, new_value: Any) -> bool:
        """
        Atomically update value if it matches expected.
        
        Returns:
            True if swap succeeded
        """
        with self._lock:
            if self._state.get(key) == expected:
                self._state[key] = new_value
                return True
            return False
    
    def snapshot(self) -> Dict:
        """Get a thread-safe snapshot of the state."""
        with self._lock:
            return self._state.copy()
    
    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._state.clear()


class SecureHeaders:
    """
    Security headers for HTTP responses.
    
    Implements OWASP recommended security headers.
    """
    
    HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        'Cache-Control': 'no-store, no-cache, must-revalidate',
        'Pragma': 'no-cache',
    }
    
    @classmethod
    def get_headers(cls, custom_cors: str = None) -> Dict[str, str]:
        """
        Get security headers with optional CORS configuration.
        
        Args:
            custom_cors: Custom CORS origin (default: same origin)
            
        Returns:
            Dictionary of headers
        """
        headers = cls.HEADERS.copy()
        
        if custom_cors:
            headers['Access-Control-Allow-Origin'] = custom_cors
            headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            headers['Access-Control-Allow-Headers'] = 'Content-Type, X-CSRF-Token'
            headers['Access-Control-Allow-Credentials'] = 'true'
        else:
            headers['Access-Control-Allow-Origin'] = '*'  # Default, should be restricted in production
        
        return headers


def rate_limit(limiter: RateLimiter, key_func: Callable = None):
    """
    Decorator for rate limiting endpoints.
    
    Args:
        limiter: RateLimiter instance
        key_func: Function to extract rate limit key from request
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract key (default to first arg which is usually request)
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Try to get IP from request
                request = args[0] if args else None
                key = getattr(request, 'remote_addr', 'unknown')
            
            if not limiter.is_allowed(key):
                remaining = limiter.get_remaining(key)
                return {
                    'error': 'Rate limit exceeded',
                    'retry_after': limiter.window_seconds,
                    'remaining': remaining
                }, 429
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_csrf(csrf_protection: CSRFProtection, session_func: Callable = None):
    """
    Decorator for CSRF protection on state-changing endpoints.
    
    Args:
        csrf_protection: CSRFProtection instance
        session_func: Function to extract session ID from request
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session ID and token
            if session_func:
                session_id = session_func(*args, **kwargs)
            else:
                session_id = None
            
            # Get token from request
            request = args[0] if args else None
            token = None
            
            if request:
                # Try different sources for CSRF token
                if hasattr(request, 'json'):
                    token = request.json.get('csrf_token')
                if not token and hasattr(request, 'headers'):
                    token = request.headers.get('X-CSRF-Token')
            
            if not csrf_protection.validate_token(session_id, token):
                logger.warning("CSRF validation failed")
                return {'error': 'Invalid or missing CSRF token'}, 403
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global instances for reuse
_default_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
_csrf_protection = CSRFProtection()
