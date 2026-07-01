"""Unit Tests for TITAN 3.0 Security Modules."""

import pytest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.safe_eval import safe_eval_expression, SafeEvalError
from core.validators import validate_input, InputValidationError
from core.secrets import SecretsManager
from core.security_middleware import RateLimiter, CSRFProtection, ThreadSafeState


class TestSafeEval:
    """Test AST-based safe expression evaluator."""
    
    def test_safe_arithmetic(self):
        """Test basic arithmetic operations."""
        assert safe_eval_expression("2 + 2") == 4
        assert safe_eval_expression("10 * 5") == 50
        assert safe_eval_expression("100 / 4") == 25.0
        assert safe_eval_expression("2 ** 3") == 8
    
    def test_safe_math_functions(self):
        """Test allowed math functions."""
        result = safe_eval_expression("math.sqrt(16)")
        assert abs(result - 4.0) < 0.001
        
        result = safe_eval_expression("math.sin(0)")
        assert abs(result) < 0.001
    
    def test_block_function_calls(self):
        """Test that dangerous function calls are blocked."""
        with pytest.raises(SafeEvalError):
            safe_eval_expression("__import__('os').system('ls')")
        
        with pytest.raises(SafeEvalError):
            safe_eval_expression("eval('1+1')")
        
        with pytest.raises(SafeEvalError):
            safe_eval_expression("exec('print(1)')")
    
    def test_block_dunder_access(self):
        """Test that dunder attributes are blocked."""
        with pytest.raises(SafeEvalError):
            safe_eval_expression("(1).__class__.__mro__")
        
        with pytest.raises(SafeEvalError):
            safe_eval_expression("'test'.__class__)")
    
    def test_invalid_syntax(self):
        """Test invalid expressions."""
        with pytest.raises(SafeEvalError):
            safe_eval_expression("2 + + 2")
        
        with pytest.raises(SafeEvalError):
            safe_eval_expression("import os")


class TestValidators:
    """Test input validation module."""
    
    def test_valid_inputs(self):
        """Test valid inputs pass validation."""
        assert validate_input("normal text", "text") is True
        assert validate_input("user123", "alphanumeric") is True
        assert validate_input("/safe/path/file.txt", "path") is True
        assert validate_input("user@example.com", "email") is True
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        with pytest.raises(InputValidationError):
            validate_input("'; DROP TABLE users; --", "text")
        
        with pytest.raises(InputValidationError):
            validate_input("1 OR 1=1", "text")
    
    def test_xss_detection(self):
        """Test XSS attack detection."""
        with pytest.raises(InputValidationError):
            validate_input("<script>alert('xss')</script>", "text")
        
        with pytest.raises(InputValidationError):
            validate_input("javascript:alert(1)", "text")
    
    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        with pytest.raises(InputValidationError):
            validate_input("../../../etc/passwd", "path")
        
        with pytest.raises(InputValidationError):
            validate_input("..\\..\\windows\\system32", "path")
    
    def test_command_injection_detection(self):
        """Test command injection detection."""
        with pytest.raises(InputValidationError):
            validate_input("test; rm -rf /", "text")
        
        with pytest.raises(InputValidationError):
            validate_input("test && cat /etc/passwd", "text")


class TestSecretsManager:
    """Test secrets management module."""
    
    def test_get_secret(self, monkeypatch):
        """Test retrieving secrets from environment."""
        monkeypatch.setenv("TEST_SECRET", "my_secret_value")
        
        manager = SecretsManager()
        value = manager.get_secret("TEST_SECRET")
        assert value == "my_secret_value"
    
    def test_missing_required_secret(self, monkeypatch):
        """Test missing required secret raises error."""
        if "MISSING_SECRET" in os.environ:
            monkeypatch.delenv("MISSING_SECRET")
        
        manager = SecretsManager()
        with pytest.raises(ValueError):
            manager.get_secret("MISSING_SECRET", required=True)
    
    def test_optional_secret_default(self, monkeypatch):
        """Test optional secret returns default."""
        if "OPTIONAL_SECRET" in os.environ:
            monkeypatch.delenv("OPTIONAL_SECRET")
        
        manager = SecretsManager()
        value = manager.get_secret("OPTIONAL_SECRET", default="default_value")
        assert value == "default_value"
    
    def test_validate_environment(self, monkeypatch):
        """Test environment validation."""
        monkeypatch.setenv("TITAN_DEBUG", "false")
        monkeypatch.setenv("TITAN_ALLOWED_ORIGINS", "https://example.com")
        monkeypatch.setenv("TITAN_FLASK_SECRET_KEY", "test_key_12345678901234567890123456789012")
        
        manager = SecretsManager()
        errors = manager.validate_environment()
        assert len(errors) == 0


class TestRateLimiter:
    """Test rate limiting middleware."""
    
    def test_rate_limiting(self):
        """Test rate limiter blocks after limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # First 5 requests should succeed
        for i in range(5):
            assert limiter.is_allowed(f"client_{i}") is True
        
        # 6th request should be blocked
        assert limiter.is_allowed("client_5") is False
    
    def test_different_clients(self):
        """Test different clients have separate limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Client 1 uses all requests
        assert limiter.is_allowed("client_1") is True
        assert limiter.is_allowed("client_1") is True
        assert limiter.is_allowed("client_1") is False
        
        # Client 2 should still have requests
        assert limiter.is_allowed("client_2") is True
    
    def test_window_reset(self):
        """Test rate limit window resets."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # Use all requests
        assert limiter.is_allowed("client_1") is True
        assert limiter.is_allowed("client_1") is True
        assert limiter.is_allowed("client_1") is False
        
        # Wait for window to reset
        import time
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.is_allowed("client_1") is True


class TestCSRFProtection:
    """Test CSRF protection middleware."""
    
    def test_generate_token(self):
        """Test CSRF token generation."""
        csrf = CSRFProtection()
        token = csrf.generate_token()
        
        assert len(token) > 32
        assert isinstance(token, str)
    
    def test_validate_token(self):
        """Test CSRF token validation."""
        csrf = CSRFProtection()
        token = csrf.generate_token()
        
        assert csrf.validate_token(token) is True
    
    def test_invalid_token(self):
        """Test invalid token rejection."""
        csrf = CSRFProtection()
        
        assert csrf.validate_token("invalid_token") is False
        assert csrf.validate_token("") is False


class TestThreadSafeState:
    """Test thread-safe state management."""
    
    def test_atomic_operations(self):
        """Test atomic get/set operations."""
        state = ThreadSafeState()
        
        state.set("key1", "value1")
        assert state.get("key1") == "value1"
    
    def test_concurrent_updates(self):
        """Test concurrent updates don't cause race conditions."""
        state = ThreadSafeState()
        state.set("counter", 0)
        
        import threading
        
        def increment():
            for _ in range(1000):
                current = state.get("counter") or 0
                state.set("counter", current + 1)
        
        threads = [threading.Thread(target=increment) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should be exactly 10000 (10 threads * 1000 increments)
        assert state.get("counter") == 10000
    
    def test_delete_operation(self):
        """Test delete operation."""
        state = ThreadSafeState()
        
        state.set("key1", "value1")
        state.delete("key1")
        assert state.get("key1") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
