"""
4. The Hunter Daemon - Real-Time Pattern Scanner
Continuously scans market data for statistical arbitrage, patterns, and anomalies.
"""

import time
import threading
import queue
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Import from multi_agent_debate to avoid circular dependency
@dataclass
class MarketData:
    timestamp: float
    price: float
    volume: float
    volatility: float
    bid: float
    ask: float
    symbol: str = "BTCUSD"

@dataclass 
class TradingSignal:
    signal_id: str
    symbol: str
    direction: str
    strength: float
    confidence: float
    timestamp: float
    features: Dict[str, float]

class SignalPool:
    """Thread-safe signal pool (duplicate to avoid circular imports)"""
    def __init__(self, max_age_seconds: float = 30.0):
        self.signals: Dict[str, TradingSignal] = {}
        self.max_age_seconds = max_age_seconds
        self.lock = threading.Lock()
        
    def add_signal(self, signal: TradingSignal) -> None:
        with self.lock:
            self.signals[signal.signal_id] = signal
            
    def get_signals(self, symbol: Optional[str] = None) -> List[TradingSignal]:
        with self.lock:
            current_time = time.time()
            valid_signals = [
                s for s in self.signals.values()
                if current_time - s.timestamp <= self.max_age_seconds
                and (symbol is None or s.symbol == symbol)
            ]
            self._cleanup_expired(current_time)
            return valid_signals
    
    def _cleanup_expired(self, current_time: float) -> None:
        expired_ids = [
            sid for sid, sig in self.signals.items()
            if current_time - sig.timestamp > self.max_age_seconds
        ]
        for sid in expired_ids:
            del self.signals[sid]

class HunterDaemon:
    """
    Real-time pattern scanning daemon
    Continuously monitors market data for hidden patterns and opportunities
    """
    
    def __init__(self, signal_pool: SignalPool):
        self.signal_pool = signal_pool
        self.running = False
        self.daemon_thread: Optional[threading.Thread] = None
        self.market_data_queue = queue.Queue(maxsize=1000)
        self.scan_interval = 0.01  # 10ms between scans
        self.stats = {
            'scans_performed': 0,
            'signals_generated': 0,
            'arbitrage_found': 0,
            'patterns_detected': 0,
            'anomalies_found': 0
        }
        self.lock = threading.Lock()
        
    def start(self) -> None:
        """Start the hunter daemon"""
        if self.running:
            logger.warning("Hunter daemon already running")
            return
            
        self.running = True
        self.daemon_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.daemon_thread.start()
        logger.info("Hunter Daemon started - scanning for patterns")
        
    def stop(self) -> None:
        """Stop the hunter daemon"""
        self.running = False
        if self.daemon_thread:
            self.daemon_thread.join(timeout=2.0)
        logger.info("Hunter Daemon stopped")
        
    def add_market_data(self, data: MarketData) -> None:
        """Add market data to scanning queue"""
        try:
            self.market_data_queue.put_nowait(data)
        except queue.Full:
            logger.debug("Market data queue full, dropping oldest data")
            
    def _scan_loop(self) -> None:
        """Main scanning loop"""
        while self.running:
            try:
                # Process available market data
                market_data_batch = []
                while not self.market_data_queue.empty() and len(market_data_batch) < 10:
                    try:
                        data = self.market_data_queue.get_nowait()
                        market_data_batch.append(data)
                    except queue.Empty:
                        break
                
                if market_data_batch:
                    latest_data = market_data_batch[-1]
                    
                    # Run all scanners
                    arb_signals = self._scan_for_arbitrage(latest_data)
                    pattern_signals = self._scan_technical_patterns(latest_data)
                    anomaly_signals = self._scan_statistical_anomalies(latest_data)
                    
                    # Add all signals to pool
                    all_signals = arb_signals + pattern_signals + anomaly_signals
                    for signal in all_signals:
                        self.signal_pool.add_signal(signal)
                    
                    # Update stats
                    with self.lock:
                        self.stats['scans_performed'] += 1
                        self.stats['signals_generated'] += len(all_signals)
                        self.stats['arbitrage_found'] += len(arb_signals)
                        self.stats['patterns_detected'] += len(pattern_signals)
                        self.stats['anomalies_found'] += len(anomaly_signals)
                
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Hunter daemon error: {e}")
                time.sleep(0.1)  # Longer sleep on error
    
    def _scan_for_arbitrage(self, market_data: MarketData) -> List[TradingSignal]:
        """Scan for statistical arbitrage opportunities"""
        signals = []
        
        # Check for bid-ask arbitrage (theoretical)
        if market_data.bid > market_data.ask * 1.0005:  # Tiny spread for HFT
            signal = TradingSignal(
                signal_id=f"arb_{int(time.time()*1000)}",
                symbol=market_data.symbol,
                direction="BUY",
                strength=0.95,
                confidence=0.85,
                timestamp=time.time(),
                features={
                    'type': 'bid_ask_arbitrage',
                    'spread': market_data.bid - market_data.ask,
                    'spread_pct': (market_data.bid / market_data.ask - 1) * 100
                }
            )
            signals.append(signal)
        
        return signals
        
    def _scan_technical_patterns(self, market_data: MarketData) -> List[TradingSignal]:
        """Scan for technical chart patterns"""
        signals = []
        
        # Detect high volatility contraction (potential breakout)
        if market_data.volatility < 0.005:  # Very low volatility
            signal = TradingSignal(
                signal_id=f"tech_{int(time.time()*1000)}",
                symbol=market_data.symbol,
                direction="HOLD",
                strength=0.6,
                confidence=0.7,
                timestamp=time.time(),
                features={
                    'type': 'volatility_contraction',
                    'volatility': market_data.volatility,
                    'pattern': 'coiling'
                }
            )
            signals.append(signal)
        
        # Detect momentum continuation
        elif market_data.volatility > 0.03:
            direction = "BUY" if np.random.random() > 0.5 else "SELL"
            signal = TradingSignal(
                signal_id=f"tech_{int(time.time()*1000)}",
                symbol=market_data.symbol,
                direction=direction,
                strength=0.7,
                confidence=0.65,
                timestamp=time.time(),
                features={
                    'type': 'momentum',
                    'volatility': market_data.volatility,
                    'pattern': 'trend_continuation'
                }
            )
            signals.append(signal)
        
        return signals
        
    def _scan_statistical_anomalies(self, market_data: MarketData) -> List[TradingSignal]:
        """Scan for statistical anomalies and outliers"""
        signals = []
        
        # Detect unusual volume spikes
        if market_data.volume > 1500:  # High volume threshold
            signal = TradingSignal(
                signal_id=f"stat_{int(time.time()*1000)}",
                symbol=market_data.symbol,
                direction="HOLD",
                strength=0.8,
                confidence=0.75,
                timestamp=time.time(),
                features={
                    'type': 'volume_spike',
                    'volume': market_data.volume,
                    'z_score': (market_data.volume - 500) / 300  # Approximate
                }
            )
            signals.append(signal)
        
        # Detect price gaps
        mid_price = (market_data.bid + market_data.ask) / 2
        if abs(mid_price - 100) > 3:  # Significant deviation from base
            direction = "SELL" if mid_price > 103 else "BUY"
            signal = TradingSignal(
                signal_id=f"stat_{int(time.time()*1000)}",
                symbol=market_data.symbol,
                direction=direction,
                strength=0.75,
                confidence=0.7,
                timestamp=time.time(),
                features={
                    'type': 'price_gap',
                    'deviation': mid_price - 100,
                    'mean_reversion_prob': 0.8
                }
            )
            signals.append(signal)
        
        return signals
    
    def get_stats(self) -> Dict:
        """Get daemon statistics"""
        with self.lock:
            return self.stats.copy()
    
    def clear_stats(self) -> None:
        """Reset statistics"""
        with self.lock:
            self.stats = {
                'scans_performed': 0,
                'signals_generated': 0,
                'arbitrage_found': 0,
                'patterns_detected': 0,
                'anomalies_found': 0
            }
