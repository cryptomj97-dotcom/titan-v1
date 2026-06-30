"""
Advanced Quantitative Algorithms Toolkit for TITAN 3.0

Implements stochastic calculus, time series analysis, signal processing,
and statistical methods for trading and price prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from scipy import stats
from scipy.signal import wavelets, hilbert
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler


class FractionalDifferentiation:
    """
    Fractional differentiation for memory preservation while achieving stationarity.
    Based on Marcos Lopez de Prado's work.
    """
    
    def __init__(self, d: float = 0.5):
        """
        Initialize fractional differencing operator.
        
        Args:
            d: Differentiation order (0 < d < 1)
               d=0: Original series
               d=1: Standard first difference
               0<d<1: Fractional difference preserving memory
        """
        self.d = d
        self.weights = self._compute_weights()
    
    def _compute_weights(self, n: int = 1000) -> np.ndarray:
        """Compute weights for fractional differentiation."""
        weights = np.zeros(n)
        weights[0] = 1.0
        
        for k in range(1, n):
            weights[k] = -weights[k-1] * (self.d - k + 1) / k
        
        return weights
    
    def fit_transform(self, prices: pd.Series) -> pd.Series:
        """
        Apply fractional differentiation to price series.
        
        Args:
            prices: Price series
            
        Returns:
            Fractionally differentiated series
        """
        if len(prices) < len(self.weights):
            # Truncate weights if series is too short
            weights = self.weights[:len(prices)]
        else:
            weights = self.weights
        
        # Convolve prices with weights
        result = np.convolve(prices.values, weights, mode='full')[:len(prices)]
        
        return pd.Series(result, index=prices.index, name=f'frac_diff_{self.d}')


class RegimeDetector:
    """
    Hidden Markov Model for regime detection.
    Identifies market states: Bull, Bear, Sideways, High Volatility.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
        """
        self.n_regimes = n_regimes
        self.model = None
        self.regime_labels = ['Low Vol', 'Normal', 'High Vol']
    
    def fit(self, returns: np.ndarray) -> 'RegimeDetector':
        """
        Fit HMM model to returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Self for chaining
        """
        from hmmlearn import hmm
        
        # Reshape for hmmlearn
        returns_2d = returns.reshape(-1, 1)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='diag',
            n_iter=100,
            random_state=42
        )
        
        self.model.fit(returns_2d)
        
        return self
    
    def predict(self, returns: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Array of regime labels (0, 1, 2, ...)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        returns_2d = returns.reshape(-1, 1)
        return self.model.predict(returns_2d)
    
    def get_regime_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over regimes.
        
        Args:
            returns: Array of returns
            
        Returns:
            Probability matrix [n_samples, n_regimes]
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        returns_2d = returns.reshape(-1, 1)
        return self.model.predict_proba(returns_2d)


class WaveletDecomposer:
    """
    Wavelet transform for multi-resolution analysis and denoising.
    """
    
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        """
        Initialize wavelet decomposer.
        
        Args:
            wavelet: Wavelet type ('db4', 'sym8', 'coif3', etc.)
            level: Decomposition level
        """
        self.wavelet = wavelet
        self.level = level
    
    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Decompose signal into wavelet coefficients.
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (detail coefficients list, approximation coefficients)
        """
        import pywt
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # coeffs[0] is approximation, coeffs[1:] are details
        approx = coeffs[0]
        details = coeffs[1:]
        
        return details, approx
    
    def reconstruct(self, details: List[np.ndarray], approx: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal from wavelet coefficients.
        
        Args:
            details: List of detail coefficients
            approx: Approximation coefficients
            
        Returns:
            Reconstructed signal
        """
        import pywt
        
        coeffs = [approx] + details
        return pywt.waverec(coeffs, self.wavelet)
    
    def denoise(self, signal: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            signal: Noisy signal
            threshold: Threshold for coefficient shrinkage (default: universal threshold)
            
        Returns:
            Denoised signal
        """
        import pywt
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Universal threshold if not specified
        if threshold is None:
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Soft thresholding on detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation unchanged
        for detail in coeffs[1:]:
            denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
        
        return pywt.waverec(denoised_coeffs, self.wavelet)


class CointegrationTracker:
    """
    Track cointegrated pairs for statistical arbitrage.
    """
    
    def __init__(self, confidence: float = 0.95):
        """
        Initialize cointegration tracker.
        
        Args:
            confidence: Confidence level for cointegration test
        """
        self.confidence = confidence
        self.pairs = []
    
    def test_pair(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Test two series for cointegration.
        
        Args:
            series1: First price series
            series2: Second price series
            
        Returns:
            Dictionary with test results
        """
        # Ensure same length
        min_len = min(len(series1), len(series2))
        s1 = series1.iloc[:min_len]
        s2 = series2.iloc[:min_len]
        
        # Engle-Granger cointegration test
        score, pvalue, _ = coint(s1, s2)
        
        is_cointegrated = pvalue < (1 - self.confidence)
        
        # Calculate hedge ratio
        if is_cointegrated:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(s2.values.reshape(-1, 1), s1.values)
            hedge_ratio = model.coef_[0]
            
            # Calculate spread
            spread = s1 - hedge_ratio * s2
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Current z-score
            current_z = (spread.iloc[-1] - spread_mean) / spread_std
        else:
            hedge_ratio = None
            spread_mean = None
            spread_std = None
            current_z = None
        
        return {
            'is_cointegrated': is_cointegrated,
            'pvalue': pvalue,
            'score': score,
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'current_z_score': current_z,
            'confidence': self.confidence
        }
    
    def find_pairs(self, prices: pd.DataFrame, min_correlation: float = 0.7) -> List[Dict]:
        """
        Find all cointegrated pairs in a universe of assets.
        
        Args:
            prices: DataFrame with multiple asset price series as columns
            min_correlation: Minimum correlation threshold to test
            
        Returns:
            List of cointegration test results for qualifying pairs
        """
        results = []
        assets = prices.columns.tolist()
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                # Check correlation first (faster filter)
                corr = prices[asset1].corr(prices[asset2])
                
                if abs(corr) >= min_correlation:
                    result = self.test_pair(prices[asset1], prices[asset2])
                    result['asset1'] = asset1
                    result['asset2'] = asset2
                    result['correlation'] = corr
                    
                    if result['is_cointegrated']:
                        results.append(result)
        
        # Sort by p-value (most significant first)
        results.sort(key=lambda x: x['pvalue'])
        
        return results


class HawkesProcess:
    """
    Hawkes process for modeling order book event arrivals.
    Self-exciting point process for trade flow modeling.
    """
    
    def __init__(self, baseline: float = 0.1, excitation: float = 0.5, decay: float = 0.3):
        """
        Initialize Hawkes process.
        
        Args:
            baseline: Baseline intensity (mu)
            excitation: Excitation parameter (alpha)
            decay: Decay rate (beta)
        """
        self.baseline = baseline
        self.excitation = excitation
        self.decay = decay
        self.events = []
        self.intensity = baseline
    
    def simulate(self, T: float, n_paths: int = 1) -> List[np.ndarray]:
        """
        Simulate Hawkes process paths.
        
        Args:
            T: Time horizon
            n_paths: Number of paths to simulate
            
        Returns:
            List of event time arrays
        """
        paths = []
        
        for _ in range(n_paths):
            events = []
            t = 0.0
            intensity = self.baseline
            
            while t < T:
                # Sample next event time
                u = np.random.exponential(1.0 / intensity)
                t += u
                
                if t < T:
                    events.append(t)
                    # Update intensity with excitation
                    intensity = self.baseline + self.excitation * np.sum(
                        np.exp(-self.decay * (t - np.array(events[:-1] or [0])))
                    )
            
            paths.append(np.array(events))
        
        return paths
    
    def get_intensity(self, t: float) -> float:
        """
        Get current intensity at time t.
        
        Args:
            t: Current time
            
        Returns:
            Intensity value
        """
        if not self.events:
            return self.baseline
        
        events_array = np.array(self.events)
        events_before_t = events_array[events_array < t]
        
        if len(events_before_t) == 0:
            return self.baseline
        
        excitation_sum = np.sum(np.exp(-self.decay * (t - events_before_t)))
        return self.baseline + self.excitation * excitation_sum


def calculate_rough_volatility(
    prices: pd.Series, 
    hurst: float = 0.1,
    window: int = 21
) -> pd.Series:
    """
    Calculate rough volatility using fractional Brownian motion.
    
    Args:
        prices: Price series
        hurst: Hurst exponent (typically ~0.1 for rough volatility)
        window: Rolling window for volatility estimation
        
    Returns:
        Rough volatility series
    """
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Realized volatility
    realized_vol = log_returns.rolling(window).std() * np.sqrt(252)
    
    # Fractional integration for roughness
    frac_diff = FractionalDifferentiation(d=hurst)
    rough_vol = frac_diff.fit_transform(realized_vol.dropna())
    
    return rough_vol


def spectral_analysis(
    signal: np.ndarray,
    sampling_rate: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform spectral analysis on signal.
    
    Args:
        signal: Input signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (frequencies, power spectral density, phase)
    """
    from scipy import signal as scipy_signal
    
    # FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1.0/sampling_rate)
    
    # Power spectral density
    psd = np.abs(fft_result) ** 2 / len(signal)
    
    # Phase
    phase = np.angle(fft_result)
    
    # Return only positive frequencies
    positive_mask = freqs >= 0
    return freqs[positive_mask], psd[positive_mask], phase[positive_mask]


def kalman_filter(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    process_noise: np.ndarray,
    measurement_noise: np.ndarray,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Kalman filter for state estimation.
    
    Args:
        observations: Observation sequence
        transition_matrix: State transition matrix (F)
        observation_matrix: Observation matrix (H)
        process_noise: Process noise covariance (Q)
        measurement_noise: Measurement noise covariance (R)
        initial_state: Initial state estimate
        initial_covariance: Initial covariance estimate
        
    Returns:
        Tuple of (filtered states, filtered covariances)
    """
    n_obs = len(observations)
    n_states = len(initial_state)
    
    states = np.zeros((n_obs, n_states))
    covariances = np.zeros((n_obs, n_states, n_states))
    
    state = initial_state.copy()
    covariance = initial_covariance.copy()
    
    for t in range(n_obs):
        # Prediction step
        state_pred = transition_matrix @ state
        covariance_pred = transition_matrix @ covariance @ transition_matrix.T + process_noise
        
        # Update step
        innovation = observations[t] - observation_matrix @ state_pred
        innovation_covariance = observation_matrix @ covariance_pred @ observation_matrix.T + measurement_noise
        kalman_gain = covariance_pred @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        
        state = state_pred + kalman_gain @ innovation
        covariance = (np.eye(n_states) - kalman_gain @ observation_matrix) @ covariance_pred
        
        states[t] = state
        covariances[t] = covariance
    
    return states, covariances


def vpín_indicator(
    volumes: np.ndarray,
    prices: np.ndarray,
    bucket_size: int = 1000
) -> np.ndarray:
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
    
    Args:
        volumes: Trade volumes
        prices: Trade prices
        bucket_size: Volume per bucket
        
    Returns:
        VPIN values
    """
    n_buckets = int(np.sum(volumes) / bucket_size)
    vpins = np.zeros(n_buckets)
    
    volume_clock = 0
    buy_volume = 0
    sell_volume = 0
    bucket_idx = 0
    
    for i in range(len(volumes)):
        # Classify trade using tick rule
        if i > 0:
            if prices[i] > prices[i-1]:
                buy_volume += volumes[i]
            elif prices[i] < prices[i-1]:
                sell_volume += volumes[i]
            # else: no change, skip
        
        volume_clock += volumes[i]
        
        if volume_clock >= bucket_size:
            # Calculate VPIN for this bucket
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                vpins[bucket_idx] = abs(buy_volume - sell_volume) / total_volume
            
            # Reset for next bucket
            buy_volume = 0
            sell_volume = 0
            volume_clock = 0
            bucket_idx += 1
    
    return vpins[:bucket_idx] if bucket_idx > 0 else np.array([0.0])
