"""
TITAN Hunter Daemon - Real-Time Pattern Scanner
Continuously scans market data for statistical arbitrage, patterns, and anomalies.
"""
import threading
import time
import queue
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    timestamp: float
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    order_book_depth: Dict[str, float]

@dataclass
class DetectedPattern:
    pattern_id: str
    pattern_type: str  # STAT_ARB, MEAN_REV, MOMENTUM, ANOMALY
    symbol: str
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    timestamp: float
    metadata: Dict[str, Any]

class SignalPool:
    """Thread-safe pool of detected signals"""
    
    def __init__(self, max_size: int = 1000):
        self.signals: Dict[str, DetectedPattern] = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def add(self, signal: DetectedPattern):
        with self.lock:
            if len(self.signals) >= self.max_size:
                # Remove oldest signal
                oldest_id = min(self.signals.keys(), key=lambda k: self.signals[k].timestamp)
                del self.signals[oldest_id]
            self.signals[signal.pattern_id] = signal
            
    def get_all(self) -> List[DetectedPattern]:
        with self.lock:
            return list(self.signals.values())
            
    def clear_expired(self, max_age_seconds: float = 60.0):
        with self.lock:
            current_time = time.time()
            expired = [
                pid for pid, sig in self.signals.items()
                if current_time - sig.timestamp > max_age_seconds
            ]
            for pid in expired:
                del self.signals[pid]

class HunterDaemon:
    """Real-time pattern scanning daemon"""
    
    def __init__(self, signal_pool: SignalPool):
        self.signal_pool = signal_pool
        self.running = False
        self.daemon_thread: Optional[threading.Thread] = None
        self.data_queue = queue.Queue(maxsize=5000)
        self.price_history = deque(maxlen=200)  # Keep last 200 ticks per symbol
        self.scan_interval = 0.01  # 10ms between scans
        
    def start(self):
        """Start the scanning daemon"""
        self.running = True
        self.daemon_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.daemon_thread.start()
        logger.info("Hunter Daemon started - scanning for patterns")
        
    def stop(self):
        """Stop the scanning daemon"""
        self.running = False
        if self.daemon_thread:
            self.daemon_thread.join(timeout=2.0)
        logger.info("Hunter Daemon stopped")
        
    def feed_data(self, tick: MarketTick):
        """Feed new market data to the scanner"""
        try:
            self.data_queue.put_nowait(tick)
        except queue.Full:
            logger.warning("Data queue full, dropping tick")
            
    def _scan_loop(self):
        """Main scanning loop"""
        while self.running:
            try:
                # Process available data
                ticks_to_process = []
                while not self.data_queue.empty():
                    try:
                        tick = self.data_queue.get_nowait()
                        ticks_to_process.append(tick)
                        self.price_history.append(tick)
                    except queue.Empty:
                        break
                        
                if ticks_to_process:
                    # Run all scanners
                    patterns = []
                    patterns.extend(self._scan_statistical_arbitrage(ticks_to_process))
                    patterns.extend(self._scan_mean_reversion(ticks_to_process))
                    patterns.extend(self._scan_momentum_breakouts(ticks_to_process))
                    patterns.extend(self._scan_volume_anomalies(ticks_to_process))
                    
                    # Add detected patterns to pool
                    for pattern in patterns:
                        self.signal_pool.add(pattern)
                        
                # Clear old signals
                self.signal_pool.clear_expired()
                
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Hunter daemon error: {e}")
                time.sleep(0.1)
                
    def _scan_statistical_arbitrage(self, ticks: List[MarketTick]) -> List[DetectedPattern]:
        """Detect statistical arbitrage opportunities"""
        patterns = []
        
        # Simple example: detect bid-ask spread anomalies
        for tick in ticks:
            spread = tick.ask - tick.bid
            mid_price = (tick.ask + tick.bid) / 2
            
            if spread < 0:  # Impossible spread - arbitrage opportunity
                pattern = DetectedPattern(
                    pattern_id=f"arb_{int(time.time()*1000)}",
                    pattern_type="STAT_ARB",
                    symbol=tick.symbol,
                    confidence=0.95,
                    entry_price=mid_price,
                    target_price=mid_price * 1.001,
                    stop_loss=mid_price * 0.999,
                    timestamp=time.time(),
                    metadata={"spread": spread, "type": "negative_spread"}
                )
                patterns.append(pattern)
                logger.info(f"Stat Arb detected: {tick.symbol} spread={spread:.4f}")
                
        return patterns
        
    def _scan_mean_reversion(self, ticks: List[MarketTick]) -> List[DetectedPattern]:
        """Detect mean reversion opportunities using Z-scores"""
        patterns = []
        
        if len(self.price_history) < 50:
            return patterns
            
        prices = [t.price for t in self.price_history]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            return patterns
            
        latest_tick = ticks[-1] if ticks else self.price_history[-1]
        z_score = (latest_tick.price - mean_price) / std_price
        
        if abs(z_score) > 2.5:  # Significant deviation
            direction = "BUY" if z_score < 0 else "SELL"
            target = mean_price
            stop = latest_tick.price * (1.02 if z_score < 0 else 0.98)
            
            pattern = DetectedPattern(
                pattern_id=f"mr_{int(time.time()*1000)}",
                pattern_type="MEAN_REV",
                symbol=latest_tick.symbol,
                confidence=min(0.9, abs(z_score) / 4),
                entry_price=latest_tick.price,
                target_price=target,
                stop_loss=stop,
                timestamp=time.time(),
                metadata={"z_score": z_score, "mean": mean_price}
            )
            patterns.append(pattern)
            
        return patterns
        
    def _scan_momentum_breakouts(self, ticks: List[MarketTick]) -> List[DetectedPattern]:
        """Detect momentum breakouts"""
        patterns = []
        
        if len(self.price_history) < 20:
            return patterns
            
        prices = [t.price for t in self.price_history[-20:]]
        recent_high = max(prices)
        recent_low = min(prices)
        
        latest_tick = ticks[-1] if ticks else self.price_history[-1]
        
        if latest_tick.price > recent_high * 1.002:  # Breakout above
            pattern = DetectedPattern(
                pattern_id=f"mom_{int(time.time()*1000)}",
                pattern_type="MOMENTUM",
                symbol=latest_tick.symbol,
                confidence=0.75,
                entry_price=latest_tick.price,
                target_price=latest_tick.price * 1.01,
                stop_loss=recent_high,
                timestamp=time.time(),
                metadata={"breakout_type": "above", "resistance": recent_high}
            )
            patterns.append(pattern)
            
        elif latest_tick.price < recent_low * 0.998:  # Breakout below
            pattern = DetectedPattern(
                pattern_id=f"mom_{int(time.time()*1000)}",
                pattern_type="MOMENTUM",
                symbol=latest_tick.symbol,
                confidence=0.75,
                entry_price=latest_tick.price,
                target_price=latest_tick.price * 0.99,
                stop_loss=recent_low,
                timestamp=time.time(),
                metadata={"breakout_type": "below", "support": recent_low}
            )
            patterns.append(pattern)
            
        return patterns
        
    def _scan_volume_anomalies(self, ticks: List[MarketTick]) -> List[DetectedPattern]:
        """Detect unusual volume spikes"""
        patterns = []
        
        if len(self.price_history) < 50:
            return patterns
            
        volumes = [t.volume for t in self.price_history]
        avg_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        latest_tick = ticks[-1] if ticks else self.price_history[-1]
        
        if std_volume > 0:
            volume_z = (latest_tick.volume - avg_volume) / std_volume
            
            if volume_z > 3.0:  # Extreme volume spike
                pattern = DetectedPattern(
                    pattern_id=f"vol_{int(time.time()*1000)}",
                    pattern_type="ANOMALY",
                    symbol=latest_tick.symbol,
                    confidence=min(0.9, volume_z / 5),
                    entry_price=latest_tick.price,
                    target_price=latest_tick.price,  # Informational only
                    stop_loss=0,
                    timestamp=time.time(),
                    metadata={"volume_z_score": volume_z, "volume": latest_tick.volume}
                )
                patterns.append(pattern)
                logger.info(f"Volume anomaly: {latest_tick.symbol} vol_z={volume_z:.2f}")
                
        return patterns
