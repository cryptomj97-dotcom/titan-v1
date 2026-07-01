"""
Meta-Learning Loop for Continuous Adaptation
Monitors performance, detects concept drift, and triggers retraining.
"""
import time
import logging
from collections import deque
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.baseline_accuracy = 0.6
        
    def record_trade(self, pnl: float, confidence: float, outcome: str):
        """Record a trade result"""
        self.performance_history.append({
            'timestamp': time.time(),
            'pnl': pnl,
            'confidence': confidence,
            'outcome': outcome,  # WIN, LOSS, BREAKEVEN
            'accuracy': 1 if outcome == 'WIN' else 0
        })
        
    def get_recent_accuracy(self, last_n: int = 20) -> float:
        """Calculate recent accuracy"""
        if len(self.performance_history) < last_n:
            return 0.5  # Default until enough data
            
        recent = list(self.performance_history)[-last_n:]
        wins = sum(1 for t in recent if t['outcome'] == 'WIN')
        return wins / len(recent)
        
    def get_avg_pnl(self, last_n: int = 20) -> float:
        """Calculate average P&L"""
        if len(self.performance_history) < last_n:
            return 0.0
            
        recent = list(self.performance_history)[-last_n:]
        return np.mean([t['pnl'] for t in recent])

class ConceptDriftDetector:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.baseline_distribution = []
        
    def set_baseline(self, accuracies: List[float]):
        """Set baseline performance distribution"""
        self.baseline_distribution = accuracies
        logger.info(f"Baseline set: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        
    def detect_drift(self, recent_accuracies: List[float]) -> bool:
        """Detect if performance has significantly drifted from baseline"""
        if len(self.baseline_distribution) < 10 or len(recent_accuracies) < 10:
            return False
            
        baseline_mean = np.mean(self.baseline_distribution)
        recent_mean = np.mean(recent_accuracies)
        
        drift = abs(baseline_mean - recent_mean)
        
        if drift > self.threshold:
            logger.warning(f"Concept drift detected! Baseline: {baseline_mean:.3f}, Current: {recent_mean:.3f}")
            return True
            
        return False

class MetaLearningLoop:
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.drift_detector = ConceptDriftDetector()
        self.retrain_interval_seconds = 3600  # 1 hour
        self.last_retrain_time = time.time()
        self.is_retraining = False
        self.retrain_trigger_count = 0
        
    def update(self, pnl: float, confidence: float, outcome: str):
        """Update with new trade result"""
        self.monitor.record_trade(pnl, confidence, outcome)
        
    def should_retrain(self) -> bool:
        """Check if retraining is needed"""
        if self.is_retraining:
            return False
            
        # Check time-based retraining
        if time.time() - self.last_retrain_time > self.retrain_interval_seconds:
            logger.info("Scheduled retraining triggered")
            return True
            
        # Check performance-based retraining
        recent_accuracy = self.monitor.get_recent_accuracy()
        if recent_accuracy < 0.45:  # Below 45% accuracy
            logger.warning(f"Poor performance detected: {recent_accuracy:.2f}")
            return True
            
        # Check concept drift
        if len(self.monitor.performance_history) >= 30:
            all_accuracies = [t['accuracy'] for t in self.monitor.performance_history]
            recent = all_accuracies[-20:]
            baseline = all_accuracies[:-20]
            
            if self.drift_detector.detect_drift(recent):
                return True
                
        return False
        
    def trigger_retraining(self, model_updater_callback=None):
        """Execute retraining process"""
        self.is_retraining = True
        self.retrain_trigger_count += 1
        
        logger.info(f"=== Starting Meta-Learning Retraining #{self.retrain_trigger_count} ===")
        
        try:
            # Step 1: Collect recent data
            logger.info("Collecting recent market data...")
            time.sleep(0.1)  # Simulate data collection
            
            # Step 2: Analyze failure patterns
            recent_trades = list(self.monitor.performance_history)[-20:]
            losses = [t for t in recent_trades if t['outcome'] == 'LOSS']
            if losses:
                avg_loss_confidence = np.mean([t['confidence'] for t in losses])
                logger.warning(f"Pattern: Losses occurred with avg confidence {avg_loss_confidence:.2f}")
                
            # Step 3: Retrain models (callback to actual training system)
            if model_updater_callback:
                logger.info("Executing model update...")
                model_updater_callback()
            else:
                logger.info("No model updater callback provided - simulating retrain")
                time.sleep(0.2)
                
            # Step 4: Validate new model
            logger.info("Validating updated model...")
            time.sleep(0.1)
            
            # Reset baseline after successful retrain
            all_accuracies = [t['accuracy'] for t in self.monitor.performance_history]
            self.drift_detector.set_baseline(all_accuracies[-30:])
            
            logger.info("=== Retraining Complete ===")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            
        finally:
            self.is_retraining = False
            self.last_retrain_time = time.time()
            
    def get_status_report(self) -> Dict[str, Any]:
        """Generate status report"""
        return {
            'total_trades': len(self.monitor.performance_history),
            'recent_accuracy': self.monitor.get_recent_accuracy(),
            'avg_pnl': self.monitor.get_avg_pnl(),
            'last_retrain': self.last_retrain_time,
            'retrain_count': self.retrain_trigger_count,
            'is_retraining': self.is_retraining,
            'next_scheduled_retrain': self.last_retrain_time + self.retrain_interval_seconds
        }

# Test block
if __name__ == "__main__":
    loop = MetaLearningLoop()
    
    # Simulate some trades
    import random
    for i in range(30):
        pnl = random.uniform(-1, 1.5)
        outcome = "WIN" if pnl > 0 else "LOSS"
        conf = random.uniform(0.5, 0.9)
        loop.update(pnl, conf, outcome)
        
    print("Status:", loop.get_status_report())
    print("Should retrain?", loop.should_retrain())
