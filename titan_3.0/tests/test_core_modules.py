"""
Comprehensive Unit Tests for TITAN 3.0 Core Modules

This test suite covers:
- Validators (input validation, injection detection)
- Safe Eval (AST-based expression evaluation)
- Secrets Management (environment variable handling)
- Security Middleware (rate limiting, CSRF, thread safety)
- Error Handler (sensitive data redaction)
"""

import pytest
import os
import time
import threading
from unittest.mock import patch, MagicMock

# Import modules to test
from core.validators import InputValidator, ValidationError
from core.safe_eval import safe_eval_expression, SafeEvalError
from core.secrets import SecretsManager
from core.security_middleware import RateLimiter, CSRFProtection, ThreadSafeState
from core.error_handler import SecureErrorHandler


class TestInputValidator:
    """Test input validation and injection detection"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = InputValidator()
    
    def test_validate_string_valid(self):
        """Test valid string passes validation"""
        result = self.validator.validate_string("Hello World", min_length=1, max_length=100)
        assert result is True
    
    def test_validate_string_too_short(self):
        """Test string below minimum length fails"""
        with pytest.raises(ValidationError):
            self.validator.validate_string("Hi", min_length=5, max_length=100)
    
    def test_validate_string_too_long(self):
        """Test string above maximum length fails"""
        with pytest.raises(ValidationError):
            self.validator.validate_string("A" * 200, min_length=1, max_length=100)
    
    def test_detect_sql_injection_basic(self):
        """Test basic SQL injection detection"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "SELECT * FROM users",
            "UNION SELECT password FROM users",
            "'; DELETE FROM accounts WHERE '1'='1"
        ]
        
        for input_str in malicious_inputs:
            assert self.validator.detect_sql_injection(input_str) is True
    
    def test_detect_sql_injection_clean(self):
        """Test clean inputs pass SQL injection check"""
        clean_inputs = [
            "SELECT my portfolio performance",
            "User requested data export",
            "Normal text without SQL keywords"
        ]
        
        for input_str in clean_inputs:
            assert self.validator.detect_sql_injection(input_str) is False
    
    def test_detect_xss_script_tags(self):
        """Test XSS detection with script tags"""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')"
        ]
        
        for input_str in malicious_inputs:
            assert self.validator.detect_xss(input_str) is True
    
    def test_detect_xss_clean(self):
        """Test clean inputs pass XSS check"""
        clean_inputs = [
            "Regular text content",
            "Email: user@example.com",
            "Description: <p>This is normal HTML</p>"
        ]
        
        for input_str in clean_inputs:
            assert self.validator.detect_xss(input_str) is False
    
    def test_detect_path_traversal(self):
        """Test path traversal detection"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "....//....//etc/passwd"
        ]
        
        for path in malicious_paths:
            assert self.validator.detect_path_traversal(path) is True
    
    def test_detect_path_traversal_clean(self):
        """Test clean paths pass traversal check"""
        clean_paths = [
            "/home/user/data/file.txt",
            "C:\\Users\\Documents\\report.pdf",
            "./config/settings.yaml"
        ]
        
        for path in clean_paths:
            assert self.validator.detect_path_traversal(path) is False
    
    def test_comprehensive_input_validation(self):
        """Test full input validation pipeline"""
        # Valid input
        valid_data = {"symbol": "AAPL", "quantity": "100", "description": "Tech stock"}
        assert self.validator.validate_input(valid_data) is True
        
        # Invalid input with SQL injection
        invalid_data = {"symbol": "AAPL'; DROP TABLE stocks; --", "quantity": "100"}
        with pytest.raises(ValidationError):
            self.validator.validate_input(invalid_data)


class TestSafeEval:
    """Test AST-based safe expression evaluation"""
    
    def test_evaluate_arithmetic(self):
        """Test basic arithmetic operations"""
        assert safe_eval_expression("2 + 2") == 4
        assert safe_eval_expression("10 - 3") == 7
        assert safe_eval_expression("5 * 4") == 20
        assert safe_eval_expression("20 / 4") == 5.0
        assert safe_eval_expression("2 ** 3") == 8
    
    def test_evaluate_complex_expression(self):
        """Test complex mathematical expressions"""
        result = safe_eval_expression("(10 + 5) * 2 - 3 / 1.5")
        assert result == 28.0
    
    def test_block_function_calls(self):
        """Test that function calls are blocked"""
        dangerous_expressions = [
            "__import__('os').system('ls')",
            "eval('1+1')",
            "exec('print(\"hello\")')",
            "open('/etc/passwd').read()",
            "globals()['__builtins__']"
        ]
        
        for expr in dangerous_expressions:
            with pytest.raises(SafeEvalError):
                safe_eval_expression(expr)
    
    def test_block_dunder_access(self):
        """Test that dunder attribute access is blocked"""
        dangerous_expressions = [
            "(1).__class__.__mro__",
            "[].__class__.__bases__",
            "{}['__class__']",
            "().__dict__"
        ]
        
        for expr in dangerous_expressions:
            with pytest.raises(SafeEvalError):
                safe_eval_expression(expr)
    
    def test_block_import_statements(self):
        """Test that import statements are blocked"""
        with pytest.raises(SafeEvalError):
            safe_eval_expression("import os")
        
        with pytest.raises(SafeEvalError):
            safe_eval_expression("from sys import exit")
    
    def test_invalid_syntax(self):
        """Test that invalid syntax raises appropriate error"""
        with pytest.raises(SafeEvalError):
            safe_eval_expression("2 + + 2")
        
        with pytest.raises(SafeEvalError):
            safe_eval_expression("if True then print('hi')")


class TestSecretsManager:
    """Test environment-based secrets management"""
    
    def setup_method(self):
        """Set up test environment"""
        self.test_env = {
            'TITAN_API_KEY': 'test_api_key_12345',
            'TITAN_DB_PASSWORD': 'secure_password_67890',
            'TITAN_ENCRYPTION_KEY': 'encryption_key_abcdef',
            'TITAN_DEBUG': 'false'
        }
    
    @patch.dict(os.environ, {'TITAN_API_KEY': 'test_key'})
    def test_get_secret_exists(self):
        """Test retrieving existing secret"""
        manager = SecretsManager()
        secret = manager.get_secret('TITAN_API_KEY')
        assert secret == 'test_key'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_secret_not_found(self):
        """Test retrieving non-existent secret"""
        manager = SecretsManager()
        secret = manager.get_secret('NON_EXISTENT_KEY')
        assert secret is None
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_secret_with_default(self):
        """Test retrieving secret with default value"""
        manager = SecretsManager()
        secret = manager.get_secret('MISSING_KEY', default='default_value')
        assert secret == 'default_value'
    
    @patch.dict(os.environ, {'TITAN_DEBUG': 'true'})
    def test_is_debug_mode_enabled(self):
        """Test debug mode detection when enabled"""
        manager = SecretsManager()
        assert manager.is_debug_mode() is True
    
    @patch.dict(os.environ, {'TITAN_DEBUG': 'false'})
    def test_is_debug_mode_disabled(self):
        """Test debug mode detection when disabled"""
        manager = SecretsManager()
        assert manager.is_debug_mode() is False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_required_secrets_missing(self):
        """Test validation fails when required secrets missing"""
        manager = SecretsManager()
        required = ['TITAN_API_KEY', 'TITAN_DB_PASSWORD']
        is_valid, missing = manager.validate_required_secrets(required)
        assert is_valid is False
        assert len(missing) == 2
    
    @patch.dict(os.environ, {'TITAN_API_KEY': 'key', 'TITAN_DB_PASSWORD': 'pass'})
    def test_validate_required_secrets_present(self):
        """Test validation passes when required secrets present"""
        manager = SecretsManager()
        required = ['TITAN_API_KEY', 'TITAN_DB_PASSWORD']
        is_valid, missing = manager.validate_required_secrets(required)
        assert is_valid is True
        assert len(missing) == 0


class TestRateLimiter:
    """Test token bucket rate limiting"""
    
    def test_initial_requests_allowed(self):
        """Test that initial requests within limit are allowed"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        for i in range(5):
            assert limiter.is_allowed("test_ip") is True
    
    def test_limit_exceeded_blocks(self):
        """Test that requests exceeding limit are blocked"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # Use up all tokens
        for i in range(3):
            limiter.is_allowed("test_ip")
        
        # Next request should be blocked
        assert limiter.is_allowed("test_ip") is False
    
    def test_different_ips_tracked_separately(self):
        """Test that different IPs have separate limits"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # IP1 uses its limit
        limiter.is_allowed("ip1")
        limiter.is_allowed("ip1")
        assert limiter.is_allowed("ip1") is False
        
        # IP2 should still have tokens
        assert limiter.is_allowed("ip2") is True
    
    def test_token_refill_over_time(self):
        """Test that tokens refill after window expires"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # Use up tokens
        limiter.is_allowed("test_ip")
        limiter.is_allowed("test_ip")
        assert limiter.is_allowed("test_ip") is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should have tokens again
        assert limiter.is_allowed("test_ip") is True


class TestCSRFProtection:
    """Test CSRF token generation and validation"""
    
    def test_generate_token_unique(self):
        """Test that generated tokens are unique"""
        csrf = CSRFProtection()
        token1 = csrf.generate_token()
        token2 = csrf.generate_token()
        assert token1 != token2
    
    def test_generate_token_format(self):
        """Test token format (should be 32-char hex)"""
        csrf = CSRFProtection()
        token = csrf.generate_token()
        assert len(token) == 64  # 32 bytes = 64 hex chars
        assert all(c in '0123456789abcdef' for c in token.lower())
    
    def test_validate_token_valid(self):
        """Test validation of valid token"""
        csrf = CSRFProtection()
        token = csrf.generate_token()
        assert csrf.validate_token(token) is True
    
    def test_validate_token_invalid(self):
        """Test validation rejects invalid tokens"""
        csrf = CSRFProtection()
        assert csrf.validate_token("invalid_token") is False
        assert csrf.validate_token("") is False
        assert csrf.validate_token(None) is False
    
    def test_validate_token_single_use(self):
        """Test that tokens are single-use"""
        csrf = CSRFProtection()
        token = csrf.generate_token()
        
        # First validation should succeed
        assert csrf.validate_token(token) is True
        
        # Second validation should fail (token consumed)
        assert csrf.validate_token(token) is False


class TestThreadSafeState:
    """Test thread-safe state management"""
    
    def test_set_and_get_value(self):
        """Test basic set and get operations"""
        state = ThreadSafeState()
        state.set("key1", "value1")
        assert state.get("key1") == "value1"
    
    def test_concurrent_writes(self):
        """Test concurrent writes don't cause race conditions"""
        state = ThreadSafeState()
        results = []
        
        def writer(value):
            for i in range(100):
                state.set("counter", value)
                results.append(state.get("counter"))
        
        threads = [
            threading.Thread(target=writer, args=("A",)),
            threading.Thread(target=writer, args=("B",))
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All operations should complete without errors
        assert len(results) == 200
    
    def test_atomic_increment(self):
        """Test atomic increment operation"""
        state = ThreadSafeState()
        state.set("counter", 0)
        
        def incrementer():
            for _ in range(1000):
                state.atomic_increment("counter")
        
        threads = [threading.Thread(target=incrementer) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should equal 10000 (10 threads * 1000 increments)
        assert state.get("counter") == 10000
    
    def test_delete_key(self):
        """Test key deletion"""
        state = ThreadSafeState()
        state.set("key1", "value1")
        state.delete("key1")
        assert state.get("key1") is None
    
    def test_clear_all(self):
        """Test clearing all state"""
        state = ThreadSafeState()
        state.set("key1", "value1")
        state.set("key2", "value2")
        state.clear()
        assert state.get("key1") is None
        assert state.get("key2") is None


class TestSecureErrorHandler:
    """Test secure error handling with sensitive data redaction"""
    
    def setup_method(self):
        """Set up error handler"""
        self.handler = SecureErrorHandler(log_file=None)  # Don't write to file in tests
    
    def test_redact_sensitive_data_password(self):
        """Test password redaction"""
        message = "Error connecting with password=secret123 to database"
        redacted = self.handler._redact_sensitive_data(message)
        assert "secret123" not in redacted
        assert "password=" in redacted or "PASSWORD" in redacted.upper()
    
    def test_redact_sensitive_data_api_key(self):
        """Test API key redaction"""
        message = "API call failed with key: sk_test_abc123xyz"
        redacted = self.handler._redact_sensitive_data(message)
        assert "sk_test_abc123xyz" not in redacted
    
    def test_redact_sensitive_data_token(self):
        """Test token redaction"""
        message = "Authorization Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = self.handler._redact_sensitive_data(message)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
    
    def test_handle_exception_logs_safely(self):
        """Test exception handling logs safely"""
        try:
            raise ValueError("Test error with password=secret in message")
        except Exception as e:
            safe_message = self.handler.handle_exception(e, context="Testing")
            assert "secret" not in safe_message
    
    def test_get_safe_context_filters_keys(self):
        """Test context filtering removes sensitive keys"""
        context = {
            "user_id": 123,
            "password": "secret",
            "api_key": "key123",
            "action": "login"
        }
        safe_context = self.handler._get_safe_context(context)
        assert "password" not in safe_context
        assert "api_key" not in safe_context
        assert "user_id" in safe_context
        assert "action" in safe_context


class TestIntegration:
    """Integration tests combining multiple security modules"""
    
    def test_full_request_validation_pipeline(self):
        """Test complete request validation flow"""
        validator = InputValidator()
        csrf = CSRFProtection()
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        
        # Simulate valid request
        ip = "192.168.1.1"
        data = {"symbol": "AAPL", "action": "buy"}
        token = csrf.generate_token()
        
        # Check rate limit
        assert limiter.is_allowed(ip) is True
        
        # Validate input
        assert validator.validate_input(data) is True
        
        # Validate CSRF token
        assert csrf.validate_token(token) is True
    
    def test_blocked_request_scenarios(self):
        """Test various blocked request scenarios"""
        validator = InputValidator()
        csrf = CSRFProtection()
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Scenario 1: Rate limit exceeded
        ip = "10.0.0.1"
        limiter.is_allowed(ip)
        limiter.is_allowed(ip)
        assert limiter.is_allowed(ip) is False
        
        # Scenario 2: SQL injection attempt
        malicious_data = {"symbol": "AAPL'; DROP TABLE stocks; --"}
        with pytest.raises(ValidationError):
            validator.validate_input(malicious_data)
        
        # Scenario 3: Invalid CSRF token
        assert csrf.validate_token("fake_token") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
