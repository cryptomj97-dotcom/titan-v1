"""
TITAN 3.0 Regime Detection Module
Topological Data Analysis (TDA) and other regime detection methods
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Set up absolute imports for package structure
import sys
from pathlib import Path

# Add parent directory to path if running as script
titan_root = Path(__file__).parent.parent
if str(titan_root) not in sys.path:
    sys.path.insert(0, str(titan_root))

from core.logging import get_logger
from core.exceptions import RegimeDetectionError

logger = get_logger(__name__)


class RegimeType(Enum):
    """Market regime types"""
    BULL_LOW_VOL = "bull_low_volatility"
    BULL_HIGH_VOL = "bull_high_volatility"
    BEAR_LOW_VOL = "bear_low_volatility"
    BEAR_HIGH_VOL = "bear_high_volatility"
    SIDEWAYS = "sideways"
    TRANSITION = "transition"


class TDABasedRegimeDetector:
    """
    Topological Data Analysis based regime detection
    Uses persistent homology to identify market structure changes
    """
    
    def __init__(self, n_regimes: int = 3, window_size: int = 60):
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.logger = get_logger(f"{__name__}.TDABasedRegimeDetector")
    
    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Detect market regimes using TDA-inspired features
        
        Args:
            returns: Return series
        
        Returns:
            Series with regime labels
        """
        try:
            # Calculate TDA-inspired features
            features = self._calculate_persistence_features(returns)
            
            # Cluster into regimes
            regimes = self._cluster_regimes(features, returns)
            
            self.logger.info(f"Detected {len(regimes.unique())} regimes using TDA")
            return regimes
            
        except Exception as e:
            self.logger.error(f"TDA regime detection failed: {e}")
            raise RegimeDetectionError(str(e))
    
    def _calculate_persistence_features(
        self,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate persistence diagram inspired features
        Simplified TDA approach without requiring specialized libraries
        """
        features = pd.DataFrame(index=returns.index)
        
        # Rolling window calculations
        for i in range(len(returns)):
            if i < self.window_size:
                continue
            
            window_returns = returns.iloc[i-self.window_size:i]
            
            # Feature 1: Persistence lifetime (range of returns)
            features.loc[returns.index[i], 'persistence_range'] = (
                window_returns.max() - window_returns.min()
            )
            
            # Feature 2: Number of significant extrema
            features.loc[returns.index[i], 'n_extrema'] = self._count_extrema(window_returns)
            
            # Feature 3: Topological complexity (zero crossings)
            features.loc[returns.index[i], 'zero_crossings'] = self._count_zero_crossings(window_returns)
            
            # Feature 4: Variance of local maxima
            features.loc[returns.index[i], 'max_variance'] = self._local_extrema_variance(window_returns, 'max')
            
            # Feature 5: Variance of local minima
            features.loc[returns.index[i], 'min_variance'] = self._local_extrema_variance(window_returns, 'min')
        
        return features.dropna()
    
    def _count_extrema(self, series: pd.Series) -> int:
        """Count significant local extrema"""
        diff = series.diff()
        sign_changes = (diff * diff.shift(1)) < 0
        return sign_changes.sum()
    
    def _count_zero_crossings(self, series: pd.Series) -> int:
        """Count zero crossings in the series"""
        sign_changes = (series * series.shift(1)) < 0
        return sign_changes.sum()
    
    def _local_extrema_variance(
        self,
        series: pd.Series,
        extremum_type: str = 'max'
    ) -> float:
        """Calculate variance of local extrema values"""
        if extremum_type == 'max':
            mask = (series > series.shift(1)) & (series > series.shift(-1))
        else:
            mask = (series < series.shift(1)) & (series < series.shift(-1))
        
        extrema_values = series[mask]
        
        if len(extrema_values) < 2:
            return 0.0
        
        return extrema_values.var()
    
    def _cluster_regimes(
        self,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> pd.Series:
        """Cluster features into regimes using K-Means"""
        from sklearn.cluster import KMeans
        
        # Prepare data
        X = features.dropna()
        
        if len(X) < self.n_regimes:
            self.logger.warning("Insufficient data for clustering")
            return pd.Series(index=returns.index, dtype=float)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Align with original index
        regimes = pd.Series(index=X.index, data=cluster_labels)
        
        # Reindex to match original returns index
        regimes = regimes.reindex(returns.index, method='ffill')
        
        return regimes


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detection
    Identifies latent market states
    """
    
    def __init__(self, n_regimes: int = 3, n_iterations: int = 100):
        self.n_regimes = n_regimes
        self.n_iterations = n_iterations
        self.model = None
        self.logger = get_logger(f"{__name__}.HMMRegimeDetector")
    
    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """
        Fit HMM and predict regimes
        
        Args:
            returns: Return series
        
        Returns:
            Series with regime labels
        """
        try:
            from hmmlearn import hmm
            
            # Prepare data (reshape for HMM)
            X = returns.dropna().values.reshape(-1, 1)
            
            if len(X) < self.n_regimes * 10:
                self.logger.warning("Insufficient data for HMM training")
                return pd.Series(index=returns.index, dtype=float)
            
            # Fit Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=self.n_iterations,
                random_state=42
            )
            
            self.model.fit(X)
            
            # Predict regimes
            hidden_states = self.model.predict(X)
            
            # Create regime series
            regimes = pd.Series(index=returns.dropna().index, data=hidden_states)
            regimes = regimes.reindex(returns.index, method='ffill')
            
            self.logger.info(f"HMM detected {len(np.unique(hidden_states))} regimes")
            return regimes
            
        except ImportError:
            self.logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
            return self._fallback_detection(returns)
        except Exception as e:
            self.logger.error(f"HMM regime detection failed: {e}")
            return self._fallback_detection(returns)
    
    def _fallback_detection(self, returns: pd.Series) -> pd.Series:
        """Fallback regime detection using simple heuristics"""
        self.logger.info("Using fallback regime detection")
        
        # Simple volatility-based regimes
        rolling_vol = returns.rolling(window=self.n_regimes * 10).std()
        
        regimes = pd.Series(index=returns.index, data=0)
        regimes[rolling_vol < rolling_vol.quantile(0.33)] = 0  # Low vol
        regimes[(rolling_vol >= rolling_vol.quantile(0.33)) & 
                (rolling_vol < rolling_vol.quantile(0.67))] = 1  # Medium vol
        regimes[rolling_vol >= rolling_vol.quantile(0.67)] = 2  # High vol
        
        return regimes
    
    def get_regime_probabilities(self, returns: pd.Series) -> pd.DataFrame:
        """Get probability distribution over regimes"""
        if self.model is None:
            raise RegimeDetectionError("Model not fitted. Call fit_predict first.")
        
        X = returns.dropna().values.reshape(-1, 1)
        probs = self.model.predict_proba(X)
        
        prob_df = pd.DataFrame(
            probs,
            index=returns.dropna().index,
            columns=[f'regime_{i}_prob' for i in range(self.n_regimes)]
        )
        
        return prob_df.reindex(returns.index, method='ffill')


class VolatilityRegimeDetector:
    """
    Simple volatility-based regime detection
    Fast and interpretable baseline
    """
    
    def __init__(
        self,
        lookback: int = 20,
        low_vol_threshold: float = 0.33,
        high_vol_threshold: float = 0.67
    ):
        self.lookback = lookback
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.logger = get_logger(f"{__name__}.VolatilityRegimeDetector")
    
    def detect_regimes(self, returns: pd.Series, prices: pd.Series = None) -> pd.Series:
        """
        Detect regimes based on volatility and trend
        
        Args:
            returns: Return series
            prices: Optional price series for trend detection
        
        Returns:
            Series with regime labels (0=low_vol, 1=medium_vol, 2=high_vol)
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.lookback).std()
        
        # Calculate volatility percentiles
        vol_quantiles = rolling_vol.quantile([
            self.low_vol_threshold,
            self.high_vol_threshold
        ])
        
        # Assign volatility regimes
        regimes = pd.Series(index=returns.index, data=1)  # Default medium
        regimes[rolling_vol < vol_quantiles[self.low_vol_threshold]] = 0
        regimes[rolling_vol > vol_quantiles[self.high_vol_threshold]] = 2
        
        # If prices provided, add trend dimension
        if prices is not None:
            sma = prices.rolling(window=self.lookback).mean()
            trend = pd.Series(index=prices.index, data=0)
            trend[prices > sma] = 1  # Bullish
            trend[prices < sma] = -1  # Bearish
            
            # Combine volatility and trend
            regime_names = []
            for i in range(len(regimes)):
                vol_state = regimes.iloc[i]
                trend_state = trend.iloc[i] if i < len(trend) else 0
                
                if trend_state == 1:
                    name = RegimeType.BULL_HIGH_VOL if vol_state == 2 else \
                           RegimeType.BULL_LOW_VOL if vol_state == 0 else \
                           RegimeType.BULL_LOW_VOL
                elif trend_state == -1:
                    name = RegimeType.BEAR_HIGH_VOL if vol_state == 2 else \
                           RegimeType.BEAR_LOW_VOL if vol_state == 0 else \
                           RegimeType.BEAR_LOW_VOL
                else:
                    name = RegimeType.SIDEWAYS
                
                regime_names.append(name.value)
            
            regimes = pd.Series(index=returns.index, data=regime_names)
        
        self.logger.info(f"Detected {len(regimes.unique())} volatility regimes")
        return regimes


class EnsembleRegimeDetector:
    """
    Ensemble regime detector combining multiple methods
    """
    
    def __init__(self, methods: List[str] = None, n_regimes: int = 3):
        self.methods = methods or ['tda', 'volatility']
        self.n_regimes = n_regimes
        self.detectors = {}
        self.logger = get_logger(f"{__name__}.EnsembleRegimeDetector")
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize regime detectors"""
        if 'tda' in self.methods:
            self.detectors['tda'] = TDABasedRegimeDetector(n_regimes=self.n_regimes)
        
        if 'hmm' in self.methods:
            self.detectors['hmm'] = HMMRegimeDetector(n_regimes=self.n_regimes)
        
        if 'volatility' in self.methods:
            self.detectors['volatility'] = VolatilityRegimeDetector()
    
    def detect_regimes(
        self,
        returns: pd.Series,
        prices: pd.Series = None,
        voting: str = 'majority'
    ) -> pd.Series:
        """
        Detect regimes using ensemble of methods
        
        Args:
            returns: Return series
            prices: Optional price series
            voting: Voting method ('majority', 'weighted', 'average')
        
        Returns:
            Consensus regime series
        """
        if len(self.detectors) == 0:
            raise RegimeDetectionError("No detectors initialized")
        
        predictions = {}
        
        # Get predictions from each detector
        for name, detector in self.detectors.items():
            try:
                if name == 'volatility' and prices is not None:
                    pred = detector.detect_regimes(returns, prices)
                else:
                    pred = detector.detect_regimes(returns)
                
                predictions[name] = pred
                self.logger.debug(f"{name} detector produced {len(pred.unique())} regimes")
            except Exception as e:
                self.logger.warning(f"{name} detector failed: {e}")
        
        if len(predictions) == 0:
            raise RegimeDetectionError("All regime detectors failed")
        
        # Combine predictions
        if voting == 'majority':
            consensus = self._majority_vote(predictions)
        elif voting == 'weighted':
            consensus = self._weighted_vote(predictions)
        else:
            consensus = self._average_vote(predictions)
        
        self.logger.info(f"Ensemble detected {len(consensus.unique())} consensus regimes")
        return consensus
    
    def _majority_vote(self, predictions: Dict[str, pd.Series]) -> pd.Series:
        """Majority voting across detectors"""
        # Align all predictions
        df = pd.DataFrame(predictions)
        
        # Take mode (most common value) at each timestamp
        consensus = df.mode(axis=1).iloc[:, 0]
        
        return consensus
    
    def _weighted_vote(self, predictions: Dict[str, pd.Series]) -> pd.Series:
        """Weighted voting (simplified - equal weights)"""
        # For now, same as majority vote
        # Could be extended with detector performance weights
        return self._majority_vote(predictions)
    
    def _average_vote(self, predictions: Dict[str, pd.Series]) -> pd.Series:
        """Average numeric regime labels"""
        df = pd.DataFrame(predictions)
        consensus = df.mean(axis=1).round().astype(int)
        return consensus


# Factory function
def create_regime_detector(
    method: str = 'ensemble',
    n_regimes: int = 3,
    **kwargs
) -> object:
    """
    Create regime detector instance
    
    Args:
        method: Detection method ('tda', 'hmm', 'volatility', 'ensemble')
        n_regimes: Number of regimes to detect
        **kwargs: Additional arguments for specific detectors
    
    Returns:
        Regime detector instance
    """
    if method == 'tda':
        return TDABasedRegimeDetector(n_regimes=n_regimes, **kwargs)
    elif method == 'hmm':
        return HMMRegimeDetector(n_regimes=n_regimes, **kwargs)
    elif method == 'volatility':
        return VolatilityRegimeDetector(**kwargs)
    elif method == 'ensemble':
        return EnsembleRegimeDetector(n_regimes=n_regimes, **kwargs)
    else:
        raise RegimeDetectionError(f"Unknown regime detection method: {method}")
