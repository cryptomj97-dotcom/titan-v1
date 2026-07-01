"""
Unified Cognitive Quantitative Organism - Main Orchestrator
Integrates all cognitive components into a single autonomous system.
"""

import time
import threading
from typing import Dict, Any, List, Optional
import logging
import numpy as np

from .multi_agent_debate import (
    SignalPool, MarketData, TradingSignal, 
    AgentCouncil, ConsensusResult
)
from .generative_world_models import GenerativeWorldModel
from .neuro_symbolic_engine import NeuroSymbolicEngine
from .hunter_daemon import HunterDaemon
from .hft_executor import HFTExecutor
from .adaptive_lens_ui import AdaptiveLensUI
from .meta_learning_loop import MetaLearningLoop

logger = logging.getLogger(__name__)

class UnifiedCognitiveQuantitativeOrganism:
    """
    The complete autonomous cognitive trading system
    Integrates multi-agent debate, generative models, neuro-symbolic reasoning,
    real-time scanning, HFT execution, adaptive UI, and meta-learning
    """
    
    def __init__(self):
        # Initialize all components
        self.signal_pool = SignalPool()
        self.council = AgentCouncil()
        self.simulator = GenerativeWorldModel()
        self.neuro_engine = NeuroSymbolicEngine()
        self.hunter_daemon = HunterDaemon(self.signal_pool)
        self.executor = HFTExecutor()
        self.ui_adapter = AdaptiveLensUI()
        self.meta_learner = MetaLearningLoop()
        
        # State tracking
        self.running = False
        self.main_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.daily_pnl: List[float] = []
        self._last_consensus: Optional[ConsensusResult] = None
        
        # Register default user
        self.ui_adapter.register_user("default", "beginner")
        
    def start(self) -> None:
        """Start the unified organism"""
        if self.running:
            logger.warning("Organism already running")
            return
            
        self.running = True
        self.hunter_daemon.start()
        self.executor.start()
        self.meta_learner.start()
        
        # Set meta-learning callbacks
        self.meta_learner.set_callbacks(
            retrain_fn=self._retrain_models,
            deploy_fn=self._deploy_new_model
        )
        
        # Start main processing loop
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
        logger.info("Unified Cognitive Quantitative Organism STARTED")
        
    def stop(self) -> None:
        """Stop the unified organism"""
        self.running = False
        self.hunter_daemon.stop()
        self.executor.stop()
        self.meta_learner.stop()
        
        if self.main_thread:
            self.main_thread.join(timeout=5.0)
            
        logger.info("Unified Cognitive Quantitative Organism STOPPED")
        
    def _main_loop(self) -> None:
        """Main processing loop coordinating all components"""
        cycle_count = 0
        
        while self.running:
            try:
                # Generate synthetic market data for demonstration
                market_data = MarketData(
                    timestamp=time.time(),
                    price=100 + np.random.normal(0, 0.5),
                    volume=np.random.uniform(100, 1000),
                    volatility=np.random.uniform(0.01, 0.03),
                    bid=100 + np.random.normal(-0.01, 0.1),
                    ask=100 + np.random.normal(0.01, 0.1),
                    symbol="BTCUSD"
                )
                
                # Feed data to hunter daemon
                self.hunter_daemon.add_market_data(market_data)
                
                # Get current signals from pool
                signals = self.signal_pool.get_signals()
                
                # Conduct multi-agent debate
                consensus = self.council.conduct_debate(market_data, signals)
                self._last_consensus = consensus
                
                # Apply neuro-symbolic constraints
                constrained_decision = self.neuro_engine.apply_constraints({
                    'action': consensus.final_action,
                    'confidence': consensus.weighted_confidence,
                    'supporting_agents': consensus.supporting_agents,
                    'reasoning': '; '.join(consensus.debate_log[-3:]) if consensus.debate_log else ''
                }, market_data)
                
                # Execute if confidence is sufficient
                if constrained_decision.get('confidence', 0) > 0.6:
                    execution_result = self.executor.execute_order(
                        action=constrained_decision['action'],
                        market_data=market_data,
                        confidence=constrained_decision['confidence']
                    )
                    
                    # Track execution
                    if execution_result.status == 'EXECUTED':
                        self.trade_history.append({
                            'timestamp': time.time(),
                            'action': execution_result.executed_price,
                            'price': execution_result.executed_price,
                            'confidence': execution_result.latency_ms,
                            'pnl': np.random.uniform(-1, 1)  # Simulated P&L
                        })
                        
                        # Record performance for meta-learning
                        if len(self.trade_history) % 10 == 0:
                            avg_pnl = np.mean([t['pnl'] for t in self.trade_history[-10:]])
                            self.meta_learner.record_performance({
                                'accuracy': abs(avg_pnl) > 0.1,
                                'avg_pnl': avg_pnl,
                                'win_rate': len([t for t in self.trade_history[-10:] if t['pnl'] > 0]) / 10
                            })
                
                # Periodically generate synthetic scenarios for stress testing
                if cycle_count % 100 == 0 and not self.simulator.synthetic_paths:
                    logger.info("Generating synthetic market scenarios...")
                    self.simulator.generate_market_scenarios(num_scenarios=100)
                
                cycle_count += 1
                time.sleep(0.05)  # 50ms between cycles
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(1.0)
    
    def _retrain_models(self) -> Dict[str, Any]:
        """Callback for meta-learning retraining"""
        logger.info("Initiating model retraining...")
        # In production, this would retrain TFT, GNN, etc.
        time.sleep(0.1)  # Simulate training
        return {'model_version': int(time.time())}
    
    def _deploy_new_model(self, new_model: Dict[str, Any]) -> None:
        """Callback for deploying new model"""
        logger.info(f"Deploying new model version: {new_model}")
        # In production, this would swap model weights
    
    def get_state(self) -> Dict[str, Any]:
        """Get current organism state"""
        hunter_stats = self.hunter_daemon.get_stats()
        executor_stats = self.executor.get_stats()
        meta_stats = self.meta_learner.get_stats()
        
        return {
            'consensus_action': self._last_consensus.final_action if self._last_consensus else 'HOLD',
            'consensus_confidence': self._last_consensus.weighted_confidence if self._last_consensus else 0.0,
            'agent_decisions': {
                agent.name: agent.analyze.__doc__
                for agent in self.council.agents
            } if self._last_consensus else {},
            'debate_log': self._last_consensus.debate_log[-5:] if self._last_consensus else [],
            'market_data': {},
            'signals_count': len(self.signal_pool.get_signals()),
            'recent_executions': [
                {
                    'order_id': e.order_id,
                    'action': e.status,
                    'price': e.executed_price,
                    'latency_ms': e.latency_ms
                }
                for e in self.executor.get_recent_executions(5)
            ],
            'performance': {
                'total_trades': len(self.trade_history),
                'avg_pnl': np.mean([t['pnl'] for t in self.trade_history]) if self.trade_history else 0.0,
                'win_rate': len([t for t in self.trade_history if t['pnl'] > 0]) / len(self.trade_history) if self.trade_history else 0.0
            },
            'scenario_analysis': {
                'scenarios_generated': len(self.simulator.synthetic_paths),
                'model_trained': self.simulator.model_trained
            },
            'risk_metrics': self.neuro_engine.get_rule_status(),
            'hunter_stats': hunter_stats,
            'meta_learning': meta_stats,
            'applied_rules': [],
            'running': self.running
        }
    
    def get_dashboard(self, user_id: str = "default") -> Dict[str, Any]:
        """Get dashboard data adapted for user type"""
        state = self.get_state()
        return self.ui_adapter.generate_dashboard_data(user_id, state)
    
    def register_user(self, user_id: str, level: str = "beginner") -> None:
        """Register a new user"""
        self.ui_adapter.register_user(user_id, level)

def main():
    """Run the Unified Cognitive Quantitative Organism"""
    organism = UnifiedCognitiveQuantitativeOrganism()
    
    try:
        organism.start()
        print("=" * 60)
        print("TITAN: Unified Cognitive Quantitative Organism")
        print("=" * 60)
        print("\nSystem is running... Press Ctrl+C to stop\n")
        
        while True:
            time.sleep(2)
            state = organism.get_state()
            dashboard = organism.get_dashboard("default")
            
            print(f"\rCycle: Action={state['consensus_action']:6s} | "
                  f"Conf={state['consensus_confidence']:.2f} | "
                  f"Signals={state['signals_count']:3d} | "
                  f"Trades={state['performance']['total_trades']:4d} | "
                  f"Mode={dashboard['mode']:8s}", end='')
                  
    except KeyboardInterrupt:
        print("\n\nStopping organism...")
        organism.stop()
        print("Organism stopped successfully.")

if __name__ == "__main__":
    main()
