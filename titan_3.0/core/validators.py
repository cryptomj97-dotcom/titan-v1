"""
TITAN 3.0 - Input Validation Module
Comprehensive input validation and sanitization for API endpoints.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(message)


# Alias for backwards compatibility
InputValidationError = ValidationError


def validate_input(value: str, input_type: str = "text") -> bool:
    """
    Convenience function for validating input strings.
    
    Args:
        value: Input string to validate
        input_type: Type of validation (text, alphanumeric, path, email)
        
    Returns:
        True if valid
        
    Raises:
        InputValidationError: If validation fails
    """
    # Check for SQL injection patterns
    if InputValidator.detect_sql_injection(value):
        raise InputValidationError(f"SQL injection pattern detected in input")
    
    # Check for XSS patterns
    if InputValidator.detect_xss(value):
        raise InputValidationError(f"XSS pattern detected in input")
    
    # Check for path traversal
    if input_type == "path" and InputValidator.detect_path_traversal(value):
        raise InputValidationError(f"Path traversal pattern detected: {value}")
    
    # Check for command injection
    if InputValidator.detect_command_injection(value):
        raise InputValidationError(f"Command injection pattern detected")
    
    return True


# Convenience functions for direct detection
def detect_sql_injection(value: str) -> bool:
    """Check if input contains SQL injection patterns."""
    return InputValidator.detect_sql_injection(value)


def detect_xss(value: str) -> bool:
    """Check if input contains XSS patterns."""
    return InputValidator.detect_xss(value)


def detect_path_traversal(value: str) -> bool:
    """Check if input contains path traversal patterns."""
    return InputValidator.detect_path_traversal(value)


def detect_command_injection(value: str) -> bool:
    """Check if input contains command injection patterns."""
    return InputValidator.detect_command_injection(value)


class InputValidator:
    """
    Comprehensive input validation for TITAN 3.0 API.
    
    Security Features:
    - Type validation
    - Range checking
    - String length limits
    - Pattern matching
    - SQL injection prevention
    - XSS prevention
    - Path traversal prevention
    """
    
    # Maximum lengths to prevent DoS
    MAX_SYMBOL_LENGTH = 20
    MAX_STRING_LENGTH = 1000
    MAX_JSON_DEPTH = 5
    
    # Allowed market types
    ALLOWED_MARKETS = {'CRYPTO', 'STOCKS', 'FOREX', 'COMMODITIES', 'INDICES'}
    
    # Allowed timeframes
    ALLOWED_TIMEFRAMES = {
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '12h',
        '1d', '3d', '1w', '1M'
    }
    
    # Symbol pattern (alphanumeric with optional underscores)
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9_]+$')
    
    # Dangerous SQL keywords (comprehensive patterns)
    SQL_INJECTION_PATTERNS = [
        r';\s*DROP\s+',
        r';\s*DELETE\s+',
        r';\s*UPDATE\s+.*\s+SET\s+',
        r';\s*INSERT\s+',
        r'--\s*$',
        r'/\*.*\*/',
        r'\bUNION\b.*\bSELECT\b',
        r'\bOR\b\s+\d+\s*=\s*\d+',
        r"'\s*OR\s+'",
        r"'\s*OR\s+\d+",
        r'\bxp_\w+\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r';\s*TRUNCATE\b',
        r';\s*ALTER\b',
        r';\s*CREATE\b',
        r'\bSELECT\b\s+\*\s+\bFROM\b',
        r'\bSELECT\b\s+\w+\s+\bFROM\b',
    ]
    
    # Simple dangerous SQL characters for quick checks
    SQL_DANGEROUS_CHARS = ['--', ';', '/*', '*/']
    
    # XSS dangerous patterns
    XSS_PATTERNS = [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe',
        r'<object',
        r'<embed',
    ]
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """
        Validate trading symbol.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Cleaned symbol
            
        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol is required", "symbol")
        
        symbol = symbol.strip().upper()
        
        if len(symbol) > cls.MAX_SYMBOL_LENGTH:
            raise ValidationError(
                f"Symbol too long (max {cls.MAX_SYMBOL_LENGTH} chars)",
                "symbol"
            )
        
        if not cls.SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                "Symbol must contain only uppercase letters, numbers, and underscores",
                "symbol"
            )
        
        return symbol
    
    @classmethod
    def validate_market(cls, market: str) -> str:
        """
        Validate market type.
        
        Args:
            market: Market type to validate
            
        Returns:
            Validated market type
            
        Raises:
            ValidationError: If market is invalid
        """
        if not market or not isinstance(market, str):
            raise ValidationError("Market is required", "market")
        
        market = market.strip().upper()
        
        if market not in cls.ALLOWED_MARKETS:
            raise ValidationError(
                f"Invalid market. Allowed: {', '.join(cls.ALLOWED_MARKETS)}",
                "market"
            )
        
        return market
    
    @classmethod
    def validate_timeframe(cls, timeframe: str) -> str:
        """
        Validate timeframe.
        
        Args:
            timeframe: Timeframe to validate
            
        Returns:
            Validated timeframe
            
        Raises:
            ValidationError: If timeframe is invalid
        """
        if not timeframe or not isinstance(timeframe, str):
            raise ValidationError("Timeframe is required", "timeframe")
        
        timeframe = timeframe.strip().lower()
        
        if timeframe not in cls.ALLOWED_TIMEFRAMES:
            raise ValidationError(
                f"Invalid timeframe. Allowed: {', '.join(cls.ALLOWED_TIMEFRAMES)}",
                "timeframe"
            )
        
        return timeframe
    
    @classmethod
    def validate_string(cls, value: str, field_name: str = "field", 
                       max_length: int = MAX_STRING_LENGTH,
                       min_length: int = 0,
                       allow_empty: bool = False) -> str:
        """
        Validate generic string input.
        
        Args:
            value: String to validate
            field_name: Name of the field for error messages
            max_length: Maximum allowed length
            min_length: Minimum required length
            allow_empty: Whether empty strings are allowed
            
        Returns:
            Validated string
            
        Raises:
            ValidationError: If string is invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field_name)
        
        value = value.strip()
        
        if not value and not allow_empty:
            raise ValidationError(f"{field_name} is required", field_name)
        
        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field_name
            )
        
        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {max_length}",
                field_name
            )
        
        return value
    
    @classmethod
    def validate_number(cls, value: Any, field_name: str = "field",
                       min_value: float = None, max_value: float = None,
                       allow_zero: bool = True) -> Union[int, float]:
        """
        Validate numeric input.
        
        Args:
            value: Number to validate
            field_name: Name of the field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_zero: Whether zero is allowed
            
        Returns:
            Validated number
            
        Raises:
            ValidationError: If number is invalid
        """
        if value is None:
            raise ValidationError(f"{field_name} is required", field_name)
        
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be a number", field_name)
        
        if not allow_zero and num == 0:
            raise ValidationError(f"{field_name} cannot be zero", field_name)
        
        if min_value is not None and num < min_value:
            raise ValidationError(
                f"{field_name} must be at least {min_value}",
                field_name
            )
        
        if max_value is not None and num > max_value:
            raise ValidationError(
                f"{field_name} must be at most {max_value}",
                field_name
            )
        
        # Return int if it's a whole number
        return int(num) if num.is_integer() else num
    
    @classmethod
    def validate_boolean(cls, value: Any, field_name: str = "field") -> bool:
        """
        Validate boolean input.
        
        Args:
            value: Value to validate
            field_name: Name of the field for error messages
            
        Returns:
            Boolean value
            
        Raises:
            ValidationError: If value is not boolean
        """
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes'):
                return True
            elif value.lower() in ('false', '0', 'no'):
                return False
        
        raise ValidationError(f"{field_name} must be a boolean", field_name)
    
    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """
        Sanitize string to prevent XSS and injection attacks.
        
        Args:
            value: String to sanitize
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove potential XSS patterns
        for pattern in cls.XSS_PATTERNS:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)
        
        return value.strip()
    
    @classmethod
    def detect_sql_injection(cls, value: str) -> bool:
        """
        Detect potential SQL injection attempts.
        
        Args:
            value: String to check
            
        Returns:
            True if SQL injection detected
        """
        if not isinstance(value, str):
            return False
        
        value_upper = value.upper()
        
        # Check for specific dangerous patterns first
        import re
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {pattern}")
                return True
        
        # Quick check for comment markers and statement terminators
        for dangerous in cls.SQL_DANGEROUS_CHARS:
            if dangerous in value:
                logger.warning(f"Potential SQL injection detected: {dangerous}")
                return True
        
        return False
    
    @classmethod
    def detect_xss(cls, value: str) -> bool:
        """
        Detect potential XSS (Cross-Site Scripting) attempts.
        
        Args:
            value: String to check
            
        Returns:
            True if XSS detected
        """
        if not isinstance(value, str):
            return False
        
        # Common XSS patterns
        xss_patterns = [
            r'<script[^>]*>',
            r'</script>',
            r'javascript:',
            r'on\w+\s*=',  # onclick=, onerror=, onload=, etc.
            r'<iframe',
            r'<object',
            r'<embed',
            r'<svg[^>]*on',
            r'<img[^>]*on',
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {pattern}")
                return True
        
        return False
    
    @classmethod
    def detect_path_traversal(cls, path: str) -> bool:
        """
        Detect path traversal attempts.
        
        Args:
            path: File path to check
            
        Returns:
            True if path traversal detected
        """
        if not isinstance(path, str):
            return False
        
        dangerous_patterns = ['..', '~', '\\', '%2e%2e', '%252e']
        
        for pattern in dangerous_patterns:
            if pattern in path.lower():
                logger.warning(f"Potential path traversal detected: {pattern}")
                return True
        
        return False
    
    @classmethod
    def detect_command_injection(cls, value: str) -> bool:
        """
        Detect potential command injection attempts.

        Args:
            value: String to check

        Returns:
            True if command injection detected
        """
        if not isinstance(value, str):
            return False

        dangerous_patterns = [';', '|', '&', '`', '$(', '${', '||', '&&']
        
        for pattern in dangerous_patterns:
            if pattern in value:
                logger.warning(f"Potential command injection detected: {pattern}")
                return True

        return False

    @classmethod
    def validate_json_depth(cls, obj: Any, max_depth: int = MAX_JSON_DEPTH) -> bool:
        """
        Validate JSON object depth to prevent DoS.
        
        Args:
            obj: JSON object to validate
            max_depth: Maximum allowed depth
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If depth exceeds limit
        """
        def check_depth(item, current_depth):
            if current_depth > max_depth:
                return False
            
            if isinstance(item, dict):
                return all(check_depth(v, current_depth + 1) for v in item.values())
            elif isinstance(item, list):
                return all(check_depth(v, current_depth + 1) for v in item)
            
            return True
        
        if not check_depth(obj, 0):
            raise ValidationError(
                f"JSON structure too deep (max {max_depth} levels)",
                "json"
            )
        
        return True
    
    @classmethod
    def validate_analysis_request(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete analysis request.
        
        Args:
            data: Request data dictionary
            
        Returns:
            Validated and sanitized data
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        
        cls.validate_json_depth(data)
        
        # Extract and validate fields
        symbol = cls.validate_symbol(data.get('symbol', 'BTCUSDT'))
        market = cls.validate_market(data.get('market', 'CRYPTO'))
        timeframe = cls.validate_timeframe(data.get('timeframe', '1h'))
        
        # Check for SQL injection in all string fields
        for key, value in data.items():
            if isinstance(value, str) and cls.detect_sql_injection(value):
                raise ValidationError(
                    f"Potentially dangerous content in field '{key}'",
                    key
                )
        
        return {
            'symbol': symbol,
            'market': market,
            'timeframe': timeframe,
            'original': data
        }
    
    @classmethod
    def validate_input(cls, data: Dict[str, Any]) -> bool:
        """
        Comprehensive input validation for API requests.
        
        Args:
            data: Dictionary of input values
            
        Returns:
            True if all inputs are valid
            
        Raises:
            ValidationError: If any input fails validation
        """
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary")
        
        for key, value in data.items():
            # Check for SQL injection in string values
            if isinstance(value, str):
                if cls.detect_sql_injection(value):
                    raise ValidationError(f"Potential SQL injection detected in field '{key}'", field=key)
                
                if cls.detect_xss(value):
                    raise ValidationError(f"Potential XSS detected in field '{key}'", field=key)
                
                if cls.detect_path_traversal(value):
                    raise ValidationError(f"Potential path traversal detected in field '{key}'", field=key)
        
        return True


# Convenience functions
def validate_symbol(symbol: str) -> str:
    return InputValidator.validate_symbol(symbol)


def validate_market(market: str) -> str:
    return InputValidator.validate_market(market)


def validate_timeframe(timeframe: str) -> str:
    return InputValidator.validate_timeframe(timeframe)


def validate_analysis_request(data: Dict[str, Any]) -> Dict[str, Any]:
    return InputValidator.validate_analysis_request(data)
