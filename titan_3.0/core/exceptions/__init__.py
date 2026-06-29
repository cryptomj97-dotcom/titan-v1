"""
TITAN 3.0 Exception Hierarchy
Custom exceptions for different error scenarios
"""
from typing import Optional, Dict, Any


class TITANException(Exception):
    """Base exception for all TITAN errors"""
    
    def __init__(self, message: str, code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': type(self).__name__,
            'code': self.code,
            'message': self.message,
            'context': self.context
        }


# Data Pipeline Exceptions
class DataException(TITANException):
    """Base exception for data-related errors"""
    pass


class DataSourceError(DataException):
    """Error fetching data from a source"""
    pass


class DataValidationError(DataException):
    """Data validation failed"""
    pass


class DataProcessingError(DataException):
    """Error during data processing"""
    pass


class MissingDataError(DataException):
    """Required data is missing"""
    pass


class StaleDataError(DataException):
    """Data is too old/stale"""
    pass


# Strategy Exceptions
class StrategyException(TITANException):
    """Base exception for strategy-related errors"""
    pass


class StrategyGenerationError(StrategyException):
    """Error generating trading strategies"""
    pass


class BacktestError(StrategyException):
    """Error during backtesting"""
    pass


class OptimizationError(StrategyException):
    """Error during strategy optimization"""
    pass


class InvalidSignalError(StrategyException):
    """Invalid trading signal generated"""
    pass


# ML Exceptions
class MLException(TITANException):
    """Base exception for ML-related errors"""
    pass


class ModelTrainingError(MLException):
    """Error during model training"""
    pass


class ModelPredictionError(MLException):
    """Error during model prediction"""
    pass


class FeatureEngineeringError(MLException):
    """Error in feature engineering"""
    pass


class RegimeDetectionError(MLException):
    """Error in regime detection"""
    pass


# Execution Exceptions
class ExecutionException(TITANException):
    """Base exception for execution-related errors"""
    pass


class OrderExecutionError(ExecutionException):
    """Error executing an order"""
    pass


class RiskLimitExceeded(ExecutionException):
    """Risk limit has been exceeded"""
    pass


class InsufficientCapital(ExecutionException):
    """Insufficient capital for trade"""
    pass


class BrokerConnectionError(ExecutionException):
    """Error connecting to broker"""
    pass


class SlippageError(ExecutionException):
    """Unexpected slippage occurred"""
    pass


# Configuration Exceptions
class ConfigException(TITANException):
    """Base exception for configuration errors"""
    pass


class ConfigNotFoundError(ConfigException):
    """Configuration file not found"""
    pass


class ConfigValidationError(ConfigException):
    """Configuration validation failed"""
    pass


class ConfigLoadError(ConfigException):
    """Error loading configuration"""
    pass


# API Exceptions
class APIException(TITANException):
    """Base exception for API errors"""
    pass


class RateLimitError(APIException):
    """API rate limit exceeded"""
    pass


class AuthenticationError(APIException):
    """Authentication failed"""
    pass


class NetworkError(APIException):
    """Network connectivity error"""
    pass


class TimeoutError(APIException):
    """Request timeout"""
    pass


# Alternative Data Exceptions
class AltDataException(TITANException):
    """Base exception for alternative data errors"""
    pass


class SentimentAnalysisError(AltDataException):
    """Error in sentiment analysis"""
    pass


class SatelliteDataError(AltDataException):
    """Error processing satellite data"""
    pass


class DataFusionError(AltDataException):
    """Error fusing multiple data sources"""
    pass


def handle_exception(exc: Exception, logger=None, context: Dict[str, Any] = None):
    """
    Handle exception with logging and structured output
    
    Args:
        exc: Exception to handle
        logger: Optional logger instance
        context: Additional context information
    
    Returns:
        Dictionary with error information
    """
    if isinstance(exc, TITANException):
        error_dict = exc.to_dict()
    else:
        error_dict = {
            'error_type': type(exc).__name__,
            'message': str(exc),
            'context': context or {}
        }
    
    if logger:
        logger.error(f"{error_dict['error_type']}: {error_dict['message']}", 
                    extra={'error_details': error_dict}, exc_info=True)
    
    return error_dict
