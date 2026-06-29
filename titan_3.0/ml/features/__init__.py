"""
TITAN 3.0 Feature Engineering Module
Technical indicators, statistical features, and advanced mathematical transformations
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from scipy import stats

# Set up absolute imports for package structure
import sys
from pathlib import Path

# Add parent directory to path if running as script
titan_root = Path(__file__).parent.parent
if str(titan_root) not in sys.path:
    sys.path.insert(0, str(titan_root))

from core.logging import get_logger
from core.exceptions import FeatureEngineeringError

logger = get_logger(__name__)


class TechnicalIndicators:
    """Collection of technical analysis indicators"""
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(
        prices: pd.Series,
        fast_window: int = 12,
        slow_window: int = 26,
        signal_window: int = 9
    ) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence"""
        ema_fast = prices.ewm(span=fast_window, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_window, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': (upper - lower) / middle,
            'percent_b': (prices - lower) / (upper - lower)
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Average Directional Index"""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = pd.Series(tr).rolling(window=window).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=window).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=window).mean() / atr
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3
    ) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_window).mean()
        
        return {'k': k, 'd': d}
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap


class StatisticalFeatures:
    """Statistical feature calculations"""
    
    @staticmethod
    def returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate returns"""
        return prices.pct_change(periods=periods)
    
    @staticmethod
    def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate log returns"""
        return np.log(prices / prices.shift(periods))
    
    @staticmethod
    def rolling_mean(
        series: pd.Series,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """Rolling mean"""
        return series.rolling(window=window, min_periods=min_periods).mean()
    
    @staticmethod
    def rolling_std(
        series: pd.Series,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """Rolling standard deviation"""
        return series.rolling(window=window, min_periods=min_periods).std()
    
    @staticmethod
    def rolling_skew(
        series: pd.Series,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """Rolling skewness"""
        return series.rolling(window=window, min_periods=min_periods).skew()
    
    @staticmethod
    def rolling_kurtosis(
        series: pd.Series,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """Rolling kurtosis"""
        return series.rolling(window=window, min_periods=min_periods).kurt()
    
    @staticmethod
    def zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
        """Z-score normalization"""
        if window:
            mean = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
        else:
            mean = series.mean()
            std = series.std()
        
        return (series - mean) / std
    
    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst Exponent to determine time series memory
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        lags = range(2, max_lag)
        
        try:
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            
            # Linear fit on log-log plot
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            h = reg[0]
            
            return h
        except Exception as e:
            logger.warning(f"Failed to calculate Hurst exponent: {e}")
            return np.nan
    
    @staticmethod
    def shannon_entropy(returns: pd.Series, bins: int = 10) -> float:
        """Calculate Shannon entropy of returns distribution"""
        hist, _ = np.histogram(returns.dropna(), bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        prob = hist / hist.sum()
        
        entropy = -np.sum(prob * np.log2(prob))
        return entropy


class AdvancedMathFeatures:
    """Advanced mathematical features for regime detection and pattern recognition"""
    
    @staticmethod
    def fourier_transform(series: pd.Series, n_components: int = 10) -> np.ndarray:
        """Extract dominant Fourier components"""
        fft = np.fft.fft(series.dropna().values)
        frequencies = np.fft.fftfreq(len(series.dropna()))
        
        # Get top n_components by magnitude
        indices = np.argsort(np.abs(fft))[-n_components:]
        
        return fft[indices]
    
    @staticmethod
    def wavelet_denoise(series: pd.Series, level: int = 1) -> pd.Series:
        """Wavelet-based denoising using PyWavelets"""
        try:
            import pywt
            
            values = series.dropna().values
            coeffs = pywt.wavedec(values, 'db4', level=level)
            
            # Threshold detail coefficients
            threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(values)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            
            denoised = pywt.waverec(coeffs, 'db4')
            
            return pd.Series(denoised[:len(series)], index=series.index)
        except ImportError:
            logger.warning("PyWavelets not installed. Install with: pip install PyWavelets")
            return series
    
    @staticmethod
    def correlation_matrix(returns_df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix"""
        return returns_df.corr(method=method)
    
    @staticmethod
    def covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix"""
        return returns_df.cov()
    
    @staticmethod
    def eigenvalues(cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate eigenvalues of covariance matrix"""
        eigenvals, _ = np.linalg.eigh(cov_matrix.values)
        return np.sort(eigenvals)[::-1]
    
    @staticmethod
    def pca_components(returns_df: pd.DataFrame, n_components: int) -> np.ndarray:
        """Principal Component Analysis"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(returns_df.dropna())
        
        return components


class FeatureEngine:
    """
    Main feature engineering engine
    Combines multiple feature types into a unified feature set
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.technicals = TechnicalIndicators()
        self.stats = StatisticalFeatures()
        self.advanced = AdvancedMathFeatures()
        self.logger = get_logger(f"{__name__}.FeatureEngine")
    
    def create_features(
        self,
        ohlcv: pd.DataFrame,
        feature_sets: List[str] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data
        
        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
            feature_sets: List of feature sets to include
                         Options: ['technical', 'statistical', 'advanced', 'all']
        
        Returns:
            DataFrame with all features
        """
        if feature_sets is None:
            feature_sets = ['technical', 'statistical']
        
        features = pd.DataFrame(index=ohlcv.index)
        
        if 'all' in feature_sets:
            feature_sets = ['technical', 'statistical', 'advanced']
        
        # Technical indicators
        if 'technical' in feature_sets:
            features = self._add_technical_features(features, ohlcv)
        
        # Statistical features
        if 'statistical' in feature_sets:
            features = self._add_statistical_features(features, ohlcv)
        
        # Advanced mathematical features
        if 'advanced' in feature_sets:
            features = self._add_advanced_features(features, ohlcv)
        
        self.logger.info(f"Created {len(features.columns)} features")
        return features
    
    def _add_technical_features(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Add technical indicator features"""
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volume = ohlcv['volume']
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'sma_{window}'] = self.technicals.sma(close, window)
            features[f'ema_{window}'] = self.technicals.ema(close, window)
        
        # RSI
        for window in [7, 14, 21]:
            features[f'rsi_{window}'] = self.technicals.rsi(close, window)
        
        # MACD
        macd = self.technicals.macd(close)
        features['macd'] = macd['macd']
        features['macd_signal'] = macd['signal']
        features['macd_hist'] = macd['histogram']
        
        # Bollinger Bands
        bb = self.technicals.bollinger_bands(close)
        features['bb_upper'] = bb['upper']
        features['bb_lower'] = bb['lower']
        features['bb_bandwidth'] = bb['bandwidth']
        features['bb_percent'] = bb['percent_b']
        
        # ATR
        features['atr'] = self.technicals.atr(high, low, close)
        
        # ADX
        features['adx'] = self.technicals.adx(high, low, close)
        
        # Stochastic
        stoch = self.technicals.stochastic(high, low, close)
        features['stoch_k'] = stoch['k']
        features['stoch_d'] = stoch['d']
        
        # OBV
        features['obv'] = self.technicals.obv(close, volume)
        
        # VWAP
        features['vwap'] = self.technicals.vwap(high, low, close, volume)
        
        return features
    
    def _add_statistical_features(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Add statistical features"""
        close = ohlcv['close']
        returns = self.stats.returns(close)
        
        # Returns
        features['returns_1d'] = returns
        features['returns_5d'] = self.stats.returns(close, 5)
        features['log_returns'] = self.stats.log_returns(close)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'roll_mean_{window}'] = self.stats.rolling_mean(returns, window)
            features[f'roll_std_{window}'] = self.stats.rolling_std(returns, window)
            features[f'roll_skew_{window}'] = self.stats.rolling_skew(returns, window)
        
        # Z-scores
        features['zscore_20'] = self.stats.zscore(close, 20)
        features['zscore_returns_20'] = self.stats.zscore(returns, 20)
        
        return features
    
    def _add_advanced_features(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Add advanced mathematical features"""
        close = ohlcv['close']
        returns = self.stats.returns(close)
        
        # Hurst exponent (rolling)
        hurst_values = []
        window = 60
        for i in range(window, len(returns)):
            h = self.advanced.hurst_exponent(returns.iloc[i-window:i])
            hurst_values.append(h)
        
        features['hurst'] = pd.Series(hurst_values, index=returns.index[window:])
        
        # Shannon entropy (rolling)
        entropy_values = []
        for i in range(window, len(returns)):
            e = self.advanced.shannon_entropy(returns.iloc[i-window:i])
            entropy_values.append(e)
        
        features['entropy'] = pd.Series(entropy_values, index=returns.index[window:])
        
        return features
    
    def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: str = 'correlation',
        top_n: int = 50
    ) -> List[str]:
        """
        Select most relevant features
        
        Args:
            features: Feature DataFrame
            target: Target variable
            method: Selection method ('correlation', 'mutual_info', 'recursive')
            top_n: Number of features to select
        
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
        
        # Drop NaN rows
        mask = ~(features.isna().any(axis=1) | target.isna())
        X = features[mask]
        y = target[mask]
        
        if method == 'correlation':
            correlations = X.corrwith(target[mask]).abs().sort_values(ascending=False)
            selected = correlations.head(top_n).index.tolist()
        
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=top_n)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()
        
        elif method == 'f_test':
            selector = SelectKBest(score_func=f_regression, k=top_n)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()
        
        else:
            raise FeatureEngineeringError(f"Unknown selection method: {method}")
        
        self.logger.info(f"Selected {len(selected)} features using {method}")
        return selected


# Factory function
def create_feature_engine(config: Dict = None) -> FeatureEngine:
    """Create feature engine instance"""
    return FeatureEngine(config)
