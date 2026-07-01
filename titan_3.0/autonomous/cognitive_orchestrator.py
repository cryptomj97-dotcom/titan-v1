"""
TITAN Unified Cognitive Orchestrator
Integrates Multi-Agent, Generative Models, Neuro-Symbolic, and Hunter Daemon into a single autonomous system.
Supports Dual-Mode UI (Oracle for normal users, Architect for pros).
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

# Use absolute imports for standalone execution
try:
    from .multi_agent_system import AgentCouncil, MarketState, ConsensusResult
    from .generative_world_models import GenerativeSimulator
    from .neuro_symbolic_engine import NeuroSymbolicEngine, NeuralPrediction
    from .hunter_daemon import HunterDaemon, SignalPool, MarketTick
except ImportError:
    from multi_agent_system import AgentCouncil, MarketState, ConsensusResult
    from generative_world_models import GenerativeSimulator
    from neuro_symbolic_engine import NeuroSymbolicEngine, NeuralPrediction
    from hunter_daemon import HunterDaemon, SignalPool, MarketTick

logger = logging.getLogger(__name__)

@dataclass
class CognitiveDecision:
    action: str
    confidence: float
    reasoning: str
    user_mode: str  # 'oracle' or 'architect'
    agent_debate: Optional[ConsensusResult]
    stress_test_results: Optional[Dict]
    rules_applied: List[str]
    timestamp: float

class AdaptiveLensUI:
    """Dual-mode interface adapter"""
    
    @staticmethod
    def format_for_user(decision: CognitiveDecision, user_type: str = 'oracle') -> Dict[str, Any]:
        if user_type == 'oracle':
            # Simple, high-confidence signals only
            if decision.confidence < 0.6:
                return {
                    'action': 'HOLD',
                    'message': 'No high-confidence opportunities detected',
                    'confidence': decision.confidence
                }
            return {
                'action': decision.action,
                'confidence': f"{decision.confidence:.0%}",
                'simple_reasoning': decision.reasoning.split('|')[0],  # First part only
                'timestamp': decision.timestamp
            }
        else:  # architect mode
            return {
                'action': decision.action,
                'confidence': decision.confidence,
                'full_reasoning': decision.reasoning,
                'agent_debate': vars(decision.agent_debate) if decision.agent_debate else None,
                'stress_test_results': decision.stress_test_results,
                'rules_applied': decision.rules_applied,
                'timestamp': decision.timestamp
            }

class CognitiveOrchestrator:
    """Main orchestrator for the Unified Cognitive System"""
    
    def __init__(self):
        self.signal_pool = SignalPool()
        self.agent_council = AgentCouncil()
        self.generative_sim = GenerativeSimulator()
        self.neuro_engine = NeuroSymbolicEngine()
        self.hunter_daemon = HunterDaemon(self.signal_pool)
        self.ui_adapter = AdaptiveLensUI()
        
        self.running = False
        self.market_history: List[MarketState] = []
        self.decision_log: List[CognitiveDecision] = []
        
    def start(self):
        """Start all cognitive components"""
        self.running = True
        self.hunter_daemon.start()
        logger.info("Cognitive Orchestrator started")
        
    def stop(self):
        """Stop all cognitive components"""
        self.running = False
        self.hunter_daemon.stop()
        logger.info("Cognitive Orchestrator stopped")
        
    def process_market_tick(self, tick: MarketTick) -> Optional[CognitiveDecision]:
        """Process a single market tick through the entire cognitive pipeline"""
        
        # Feed to Hunter Daemon for pattern scanning
        self.hunter_daemon.feed_data(tick)
        
        # Update market state history
        market_state = MarketState(
            timestamp=tick.timestamp,
            price=tick.price,
            volume=tick.volume,
            volatility=0.02,  # Would calculate from history in production
            bid=tick.bid,
            ask=tick.ask,
            order_book_imbalance=0.0
        )
        self.market_history.append(market_state)
        if len(self.market_history) > 200:
            self.market_history = self.market_history[-200:]
            
        # Step 1: Multi-Agent Debate
        agent_consensus = self.agent_council.conduct_debate(market_state, self.market_history)
        
        # Step 2: Generative Stress Testing (if confidence is high enough)
        stress_results = None
        if agent_consensus.weighted_confidence > 0.5:
            # Generate quick stress test scenarios
            scenarios = self.generative_sim.gan.generate_scenarios(
                tick.price, num_scenarios=100, volatility_regime='normal'
            )
            # Simplified strategy function for demo
            def dummy_strategy(scenario): return 0.01
            stress_results = self.generative_sim.gan.stress_test_strategy(dummy_strategy, scenarios)
            
            # Reduce confidence if stress test shows poor results
            if stress_results['max_drawdown'] < -0.10:  # More than 10% drawdown
                agent_consensus.weighted_confidence *= 0.7
                
        # Step 3: Neuro-Symbolic Reasoning
        neural_pred = NeuralPrediction(
            action=agent_consensus.final_action,
            confidence=agent_consensus.weighted_confidence,
            pattern_detected="Multi-agent consensus",
            raw_score=agent_consensus.weighted_confidence
        )
        
        market_context = {
            'volatility': market_state.volatility,
            'current_drawdown': 0.0,
            'proposed_position_size': 0.05,
            'news_detected': False,
            'daily_volume': tick.volume * 10000
        }
        
        final_decision = self.neuro_engine.apply_constraints(neural_pred, market_context)
        
        # Build cognitive decision
        cognitive_decision = CognitiveDecision(
            action=final_decision['action'],
            confidence=final_decision['confidence'],
            reasoning=final_decision['reasoning'],
            user_mode='auto',
            agent_debate=agent_consensus,
            stress_test_results=stress_results,
            rules_applied=final_decision.get('rules_applied', []),
            timestamp=time.time()
        )
        
        self.decision_log.append(cognitive_decision)
        if len(self.decision_log) > 1000:
            self.decision_log = self.decision_log[-500:]
            
        return cognitive_decision
        
    def get_oracle_signal(self, tick: MarketTick) -> Dict[str, Any]:
        """Get simplified signal for normal users"""
        decision = self.process_market_tick(tick)
        if not decision:
            return {'action': 'HOLD', 'message': 'Processing error'}
            
        return self.ui_adapter.format_for_user(decision, user_type='oracle')
        
    def get_architect_analysis(self, tick: MarketTick) -> Dict[str, Any]:
        """Get detailed analysis for pro users"""
        decision = self.process_market_tick(tick)
        if not decision:
            return {'error': 'Processing error'}
            
        # Add hunter daemon patterns
        patterns = self.signal_pool.get_all()
        
        return {
            **self.ui_adapter.format_for_user(decision, user_type='architect'),
            'active_patterns': [vars(p) for p in patterns[:10]],  # Top 10 patterns
            'pattern_count': len(patterns)
        }

def run_demo():
    """Run a demonstration of the cognitive system"""
    print("Starting TITAN Cognitive Orchestrator Demo...")
    
    orchestrator = CognitiveOrchestrator()
    orchestrator.start()
    
    try:
        # Simulate market ticks
        base_price = 100.0
        for i in range(50):
            # Random walk
            price_change = (0.5 - 0.5) * 0.1
            new_price = base_price + price_change
            base_price = new_price
            
            tick = MarketTick(
                timestamp=time.time(),
                symbol="BTCUSD",
                price=new_price,
                volume=100 + (i % 50),
                bid=new_price - 0.01,
                ask=new_price + 0.01,
                order_book_depth={}
            )
            
            # Process tick
            decision = orchestrator.process_market_tick(tick)
            
            if i % 10 == 0:  # Print every 10th decision
                print(f"\n--- Tick {i} ---")
                print(f"Price: ${new_price:.2f}")
                print(f"Action: {decision.action}")
                print(f"Confidence: {decision.confidence:.2f}")
                print(f"Reasoning: {decision.reasoning[:80]}...")
                
                # Get Oracle view
                oracle_view = orchestrator.get_oracle_signal(tick)
                print(f"\nOracle View: {json.dumps(oracle_view, indent=2)}")
                
                # Get Architect view
                architect_view = orchestrator.get_architect_analysis(tick)
                print(f"\nArchitect View (summary):")
                print(f"  Action: {architect_view['action']}")
                print(f"  Confidence: {architect_view['confidence']:.2f}")
                print(f"  Active Patterns: {architect_view['pattern_count']}")
                
            time.sleep(0.05)  # 50ms between ticks
            
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    finally:
        orchestrator.stop()
        print("Demo completed")

if __name__ == "__main__":
    run_demo()
