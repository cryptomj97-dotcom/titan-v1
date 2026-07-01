"""
7. Infinite Meta-Learning Loop
Continuous self-improvement system with concept drift detection and auto-retraining.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetaLearningLoop:
    """
    Infinite Meta-Learning System
    Continuously monitors performance, detects concept drift, and triggers retraining
    """
    
    def __init__(
        self, 
        performance_threshold: float = 0.6,
        retrain_interval: float = 3600,
        window_size: int = 100
    ):
        self.performance_threshold = performance_threshold
        self.retrain_interval = retrain_interval  # seconds
        self.window_size = window_size
        
        self.performance_history: deque = deque(maxlen=window_size)
        self.last_retrain = time.time()
        self.models: Dict[str, Any] = {}
        self.model_versions: Dict[str, int] = {}
        
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.retrain_callback: Optional[Callable] = None
        self.deploy_callback: Optional[Callable] = None
        
        # Metrics
        self.stats = {
            'retrains_triggered': 0,
            'drift_events_detected': 0,
            'performance_degradations': 0,
            'model_updates': 0
        }
        self.lock = threading.Lock()
        
    def start(self) -> None:
        """Start meta-learning monitoring"""
        if self.running:
            logger.warning("Meta-learning loop already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Meta-Learning Loop started")
        
    def stop(self) -> None:
        """Stop meta-learning monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Meta-Learning Loop stopped")
        
    def set_callbacks(
        self, 
        retrain_fn: Optional[Callable] = None,
        deploy_fn: Optional[Callable] = None
    ) -> None:
        """Set callback functions for retraining and deployment"""
        self.retrain_callback = retrain_fn
        self.deploy_callback = deploy_fn
        
    def record_performance(self, metrics: Dict[str, float]) -> None:
        """Record new performance metrics"""
        self.performance_history.append({
            'timestamp': time.time(),
            **metrics
        })
        
    def _monitor_loop(self) -> None:
        """Continuous monitoring loop"""
        while self.running:
            try:
                should_retrain = False
                reason = []
                
                # Check performance degradation
                if len(self.performance_history) >= 20:
                    recent_perf = list(self.performance_history)[-10:]
                    older_perf = list(self.performance_history)[-20:-10]
                    
                    recent_avg = np.mean([p.get('accuracy', 0) for p in recent_perf])
                    older_avg = np.mean([p.get('accuracy', 0) for p in older_perf])
                    
                    # Detect significant performance drop
                    if recent_avg < older_avg * 0.85:  # 15% drop
                        should_retrain = True
                        reason.append(f"Performance degradation: {older_avg:.3f} -> {recent_avg:.3f}")
                        with self.lock:
                            self.stats['performance_degradations'] += 1
                    
                    # Check absolute threshold
                    if recent_avg < self.performance_threshold:
                        should_retrain = True
                        reason.append(f"Below threshold: {recent_avg:.3f} < {self.performance_threshold}")
                
                # Check scheduled retraining
                if time.time() - self.last_retrain > self.retrain_interval:
                    should_retrain = True
                    reason.append("Scheduled retraining interval reached")
                
                # Trigger retraining if needed
                if should_retrain and self.retrain_callback:
                    logger.info(f"Retraining triggered: {'; '.join(reason)}")
                    with self.lock:
                        self.stats['retrains_triggered'] += 1
                    
                    # Execute retraining
                    new_model = self.retrain_callback()
                    
                    if new_model:
                        self.last_retrain = time.time()
                        
                        # Deploy new model
                        if self.deploy_callback:
                            self.deploy_callback(new_model)
                            with self.lock:
                                self.stats['model_updates'] += 1
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Meta-learning monitor error: {e}")
                time.sleep(30)
    
    def detect_concept_drift(self, current_data: np.ndarray, reference_data: np.ndarray) -> bool:
        """
        Detect concept drift using statistical tests
        Returns True if drift is detected
        """
        if len(current_data) < 10 or len(reference_data) < 10:
            return False
        
        # Simple Kolmogorov-Smirnov-like test
        current_mean = np.mean(current_data)
        ref_mean = np.mean(reference_data)
        current_std = np.std(current_data)
        ref_std = np.std(reference_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((current_std**2 + ref_std**2) / 2)
        if pooled_std == 0:
            return False
            
        cohens_d = abs(current_mean - ref_mean) / pooled_std
        
        # Drift detected if effect size > 0.8 (large effect)
        drift_detected = cohens_d > 0.8
        
        if drift_detected:
            with self.lock:
                self.stats['drift_events_detected'] += 1
            logger.info(f"Concept drift detected: Cohen's d = {cohens_d:.3f}")
        
        return drift_detected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            if len(self.performance_history) > 0:
                recent = list(self.performance_history)[-10:]
                stats['current_accuracy'] = np.mean([p.get('accuracy', 0) for p in recent])
                stats['performance_trend'] = (
                    "improving" if len(recent) > 1 and recent[-1].get('accuracy', 0) > recent[0].get('accuracy', 0)
                    else "degrading" if len(recent) > 1
                    else "stable"
                )
            else:
                stats['current_accuracy'] = 0.0
                stats['performance_trend'] = "insufficient_data"
            
            stats['time_since_last_retrain'] = time.time() - self.last_retrain
            stats['next_scheduled_retrain'] = self.last_retrain + self.retrain_interval
            
            return stats
    
    def force_retrain(self) -> bool:
        """Manually trigger retraining"""
        if self.retrain_callback:
            logger.info("Manual retraining triggered")
            new_model = self.retrain_callback()
            if new_model and self.deploy_callback:
                self.deploy_callback(new_model)
                self.last_retrain = time.time()
                with self.lock:
                    self.stats['retrains_triggered'] += 1
                    self.stats['model_updates'] += 1
                return True
        return False
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        with self.lock:
            self.stats = {
                'retrains_triggered': 0,
                'drift_events_detected': 0,
                'performance_degradations': 0,
                'model_updates': 0
            }
