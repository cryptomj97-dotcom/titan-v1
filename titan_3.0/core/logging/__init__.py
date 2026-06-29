"""
TITAN 3.0 Logging Framework
Structured logging with multiple handlers and formatters
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName']:
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup a logger with console and optional file output
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_format: Use JSON formatting
        console_output: Enable console output
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TITANLogger:
    """Centralized logging manager for TITAN 3.0"""
    
    _loggers = {}
    _default_config = {
        'level': 'INFO',
        'json_format': False,
        'console_output': True,
        'log_dir': 'logs'
    }
    
    @classmethod
    def initialize(cls, **kwargs):
        """Initialize logging system with default configuration"""
        cls._default_config.update(kwargs)
        
        # Create log directory
        log_dir = Path(cls._default_config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_logger(cls, name: str, **kwargs) -> logging.Logger:
        """
        Get or create a logger with custom settings
        
        Args:
            name: Logger name (usually __name__)
            **kwargs: Override default config options
        
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Merge configs
        config = {**cls._default_config, **kwargs}
        
        # Determine log file path
        log_file = None
        if config.get('log_dir'):
            log_file = Path(config['log_dir']) / f"{name.replace('.', '_')}.log"
        
        # Create logger
        logger = setup_logger(
            name=name,
            level=config['level'],
            log_file=str(log_file) if log_file else None,
            json_format=config['json_format'],
            console_output=config['console_output']
        )
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def set_level(cls, name: str, level: str):
        """Set logging level for a specific logger"""
        if name in cls._loggers:
            cls._loggers[name].setLevel(getattr(logging, level.upper()))
    
    @classmethod
    def shutdown(cls):
        """Shutdown all loggers"""
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
        cls._loggers.clear()


# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return TITANLogger.get_logger(name)


def log_trade(logger: logging.Logger, trade_data: dict):
    """Log trade execution data"""
    logger.info("TRADE_EXECUTED", extra={'trade': trade_data})


def log_signal(logger: logging.Logger, signal_data: dict):
    """Log trading signal"""
    logger.info("SIGNAL_GENERATED", extra={'signal': signal_data})


def log_error(logger: logging.Logger, error: Exception, context: dict = None):
    """Log error with context"""
    extra = {'error_type': type(error).__name__, 'error_message': str(error)}
    if context:
        extra.update(context)
    logger.error(f"ERROR: {error}", extra=extra, exc_info=True)
