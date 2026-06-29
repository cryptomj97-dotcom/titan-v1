"""
TITAN 3.0 Data Ingestion Module
Resilient data pipeline with multi-source support, caching, and circuit breakers
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Set up absolute imports for package structure
import sys
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__" or "data.ingestion" in sys.modules:
    titan_root = Path(__file__).parent.parent
    if str(titan_root) not in sys.path:
        sys.path.insert(0, str(titan_root))

from core.logging import get_logger
from core.exceptions import (
    DataSourceError, 
    MissingDataError, 
    StaleDataError,
    RateLimitError,
    TimeoutError,
    DataValidationError
)
from core.config import DataSourceConfig

logger = get_logger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern for resilient API calls
    Prevents cascading failures when a data source is unavailable
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = 'closed'  # closed, open, half-open
        self.successful_requests = 0
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker entering half-open state")
                self.state = 'half-open'
                self.successful_requests = 0
            else:
                raise DataSourceError(
                    "Circuit breaker is open",
                    code="CIRCUIT_OPEN",
                    context={'source': kwargs.get('source', 'unknown')}
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker entering half-open state")
                self.state = 'half-open'
                self.successful_requests = 0
            else:
                raise DataSourceError(
                    "Circuit breaker is open",
                    code="CIRCUIT_OPEN",
                    context={'source': kwargs.get('source', 'unknown')}
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        self.failures = 0
        if self.state == 'half-open':
            self.successful_requests += 1
            if self.successful_requests >= self.half_open_requests:
                logger.info("Circuit breaker closed after successful recovery")
                self.state = 'closed'
        else:
            self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            logger.warning(f"Circuit breaker opened after {self.failures} failures")
            self.state = 'open'


class CacheManager:
    """
    Simple in-memory cache with TTL support
    Can be extended to use Redis for distributed caching
    """
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() - entry['timestamp'] > self.ttl:
            del self._cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any):
        """Set value in cache with timestamp"""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def invalidate(self, key: str):
        """Invalidate a specific cache key"""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
    
    def _evict_oldest(self):
        """Evict oldest cache entry"""
        if not self._cache:
            return
        
        oldest_key = min(self._cache.keys(), 
                        key=lambda k: self._cache[k]['timestamp'])
        del self._cache[oldest_key]


class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.retry_attempts * 2,
            recovery_timeout=60
        )
        self.cache = CacheManager(ttl=config.cache_ttl) if config.cache_enabled else None
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def fetch_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """Fetch historical price data"""
        pass
    
    @abstractmethod
    def fetch_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote"""
        pass
    
    def _get_cache_key(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> str:
        """Generate cache key"""
        return f"{symbol}_{start.isoformat()}_{end.isoformat()}_{timeframe}"
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean data"""
        if df.empty:
            raise MissingDataError(f"No data returned for {symbol}")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing columns: {missing_cols}")
        
        # Check for stale data
        latest_date = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index).max()
        if datetime.now() - latest_date > timedelta(days=7):
            self.logger.warning(f"Data for {symbol} may be stale (latest: {latest_date})")
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        # Sort by date
        df = df.sort_index()
        
        return df


class YahooFinanceDataSource(DataSource):
    """Yahoo Finance data source implementation"""
    
    def __init__(self, config: DataSourceConfig = None):
        super().__init__(config or DataSourceConfig(provider='yahoo'))
    
    def fetch_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """Fetch historical price data from Yahoo Finance"""
        
        # Check cache
        if self.cache:
            cache_key = self._get_cache_key(symbol, start_date, end_date, timeframe)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Cache hit for {symbol}")
                return cached_data
        
        try:
            import yfinance as yf
            
            def _fetch():
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=timeframe)
                return df
            
            df = self.circuit_breaker.call(_fetch, source='yahoo')
            df = self._validate_data(df, symbol)
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, df)
            
            self.logger.info(f"Fetched {len(df)} records for {symbol} from Yahoo Finance")
            return df
            
        except ImportError:
            self.logger.error("yfinance not installed. Run: pip install yfinance")
            raise DataSourceError("yfinance not available")
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise DataSourceError(str(e), context={'symbol': symbol, 'source': 'yahoo'})
    
    def fetch_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote from Yahoo Finance"""
        try:
            import yfinance as yf
            
            def _fetch():
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                return {
                    'symbol': symbol,
                    'price': info.last_price,
                    'bid': info.bid,
                    'ask': info.ask,
                    'volume': info.last_volume,
                    'timestamp': datetime.now()
                }
            
            quote = self.circuit_breaker.call(_fetch, source='yahoo')
            return quote
            
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}")
            raise DataSourceError(str(e), context={'symbol': symbol, 'source': 'yahoo'})


class AlphaVantageDataSource(DataSource):
    """Alpha Vantage data source implementation"""
    
    def __init__(self, config: DataSourceConfig = None):
        super().__init__(config or DataSourceConfig(provider='alphavantage'))
        self.api_key = config.api_key if config else None
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """Fetch historical price data from Alpha Vantage"""
        
        if not self.api_key:
            raise DataSourceError("API key required for Alpha Vantage")
        
        # Check cache
        if self.cache:
            cache_key = self._get_cache_key(symbol, start_date, end_date, timeframe)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            import requests
            
            # Map timeframe to Alpha Vantage function
            function_map = {
                '1m': 'TIME_SERIES_INTRADAY',
                '5m': 'TIME_SERIES_INTRADAY',
                '15m': 'TIME_SERIES_INTRADAY',
                '30m': 'TIME_SERIES_INTRADAY',
                '60m': 'TIME_SERIES_INTRADAY',
                '1d': 'TIME_SERIES_DAILY',
                '1w': 'TIME_SERIES_WEEKLY',
                '1M': 'TIME_SERIES_MONTHLY'
            }
            
            function = function_map.get(timeframe, 'TIME_SERIES_DAILY')
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'datatype': 'json',
                'outputsize': 'full'
            }
            
            if timeframe in ['1m', '5m', '15m', '30m', '60m']:
                params['interval'] = timeframe
            
            def _fetch():
                response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
                
                if response.status_code == 429:
                    raise RateLimitError("Alpha Vantage rate limit exceeded")
                
                response.raise_for_status()
                data = response.json()
                
                # Extract time series data
                key_prefix = function.replace('TIME_SERIES_', '').lower() + '_'
                for key in data:
                    if 'Time Series' in key or 'time series' in key.lower():
                        time_series = data[key]
                        break
                else:
                    if 'Note' in data:
                        raise RateLimitError(data['Note'])
                    raise DataSourceError(f"Unexpected response format: {data.keys()}")
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                return df
            
            df = self.circuit_breaker.call(_fetch, source='alphavantage')
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            df = self._validate_data(df, symbol)
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, df)
            
            self.logger.info(f"Fetched {len(df)} records for {symbol} from Alpha Vantage")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise DataSourceError(str(e), context={'symbol': symbol, 'source': 'alphavantage'})
    
    def fetch_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote from Alpha Vantage"""
        if not self.api_key:
            raise DataSourceError("API key required for Alpha Vantage")
        
        try:
            import requests
            
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' not in data:
                raise DataSourceError(f"Unexpected response: {data}")
            
            quote = data['Global Quote']
            return {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%'),
                'volume': int(quote.get('06. volume', 0)),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}")
            raise DataSourceError(str(e), context={'symbol': symbol, 'source': 'alphavantage'})


class DataIngestionEngine:
    """
    Main data ingestion engine that coordinates multiple data sources
    Implements fallback logic and parallel fetching
    """
    
    def __init__(self, config: Dict[str, DataSourceConfig] = None):
        self.sources: Dict[str, DataSource] = {}
        self.primary_source: Optional[str] = None
        self.logger = get_logger(__name__)
        
        if config:
            self.register_sources(config)
    
    def register_source(self, name: str, source: DataSource, primary: bool = False):
        """Register a data source"""
        self.sources[name] = source
        if primary or not self.primary_source:
            self.primary_source = name
        self.logger.info(f"Registered data source: {name}")
    
    def register_sources(self, configs: Dict[str, DataSourceConfig]):
        """Register multiple data sources from config"""
        source_classes = {
            'yahoo': YahooFinanceDataSource,
            'alphavantage': AlphaVantageDataSource
        }
        
        for name, config in configs.items():
            source_class = source_classes.get(config.provider)
            if source_class:
                source = source_class(config)
                self.register_source(name, source, primary=(name == 'primary'))
            else:
                self.logger.warning(f"Unknown provider: {config.provider}")
    
    def fetch_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        use_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Fetch price data with automatic fallback to alternative sources
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            use_fallback: Try alternative sources on failure
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.sources:
            raise DataSourceError("No data sources registered")
        
        # Try primary source first
        sources_to_try = [self.primary_source] if self.primary_source else []
        
        # Add remaining sources
        sources_to_try.extend([s for s in self.sources.keys() if s != self.primary_source])
        
        last_error = None
        for source_name in sources_to_try:
            try:
                self.logger.info(f"Fetching {symbol} from {source_name}")
                source = self.sources[source_name]
                df = source.fetch_price_data(symbol, start_date, end_date, timeframe)
                df.attrs['source'] = source_name
                return df
            except Exception as e:
                last_error = e
                self.logger.warning(f"Failed to fetch from {source_name}: {e}")
                
                if not use_fallback:
                    raise
        
        raise DataSourceError(
            f"All data sources failed for {symbol}",
            context={'last_error': str(last_error), 'tried_sources': sources_to_try}
        )
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            parallel: Fetch in parallel
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if parallel:
            return self._fetch_parallel(symbols, start_date, end_date, timeframe)
        else:
            return self._fetch_sequential(symbols, start_date, end_date, timeframe)
    
    def _fetch_sequential(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data sequentially"""
        results = {}
        for symbol in symbols:
            try:
                df = self.fetch_price_data(symbol, start_date, end_date, timeframe)
                results[symbol] = df
            except Exception as e:
                self.logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = None
        return results
    
    async def _fetch_parallel(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data in parallel using asyncio"""
        
        async def fetch_symbol(symbol: str):
            loop = asyncio.get_event_loop()
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda: self.fetch_price_data(symbol, start_date, end_date, timeframe)
                )
                return symbol, df
            except Exception as e:
                self.logger.error(f"Failed to fetch {symbol}: {e}")
                return symbol, None
        
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results_list = await asyncio.gather(*tasks)
        
        return dict(results_list)


# Factory function
def create_ingestion_engine(config_path: Optional[str] = None) -> DataIngestionEngine:
    """Create data ingestion engine from config"""
    from ..core.config import get_config
    
    titan_config = get_config()
    engine = DataIngestionEngine(titan_config.data_sources)
    
    # Set default sources if none configured
    if not engine.sources:
        yahoo_config = DataSourceConfig(provider='yahoo')
        engine.register_source('yahoo', YahooFinanceDataSource(yahoo_config), primary=True)
    
    return engine
