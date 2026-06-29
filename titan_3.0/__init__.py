"""
TITAN 3.0 - Main Package
Automated Trading System with Advanced Mathematics and Resilient Data Pipelines
"""

__version__ = '3.0.0'
__author__ = 'TITAN Team'

from .core import (
    TITANConfig,
    ConfigManager,
    get_config,
    TITANLogger,
    get_logger,
    TITANException,
    handle_exception
)

from .data import (
    DataSource,
    YahooFinanceDataSource,
    AlphaVantageDataSource,
    DataIngestionEngine,
    create_ingestion_engine
)

__all__ = [
    # Core
    'TITANConfig',
    'ConfigManager',
    'get_config',
    'TITANLogger',
    'get_logger',
    'TITANException',
    'handle_exception',
    
    # Data
    'DataSource',
    'YahooFinanceDataSource',
    'AlphaVantageDataSource',
    'DataIngestionEngine',
    'create_ingestion_engine',
]
