"""
TITAN Multi-Agent Collaborative Intelligence System
"The Council" - Specialized agents debating to reach consensus
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    TREND = "trend_follower"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "stat_arb"
    MACRO = "macro_analyst"
    RISK = "risk_manager"

@dataclass
class AgentVote:
    agent_id: str
    role: AgentRole
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    expected_return: float
    risk_score: float

class SpecializedAgent:
    """Base class for specialized trading agents"""
    
    def __init__(self, role: AgentRole, model=None):
        self.role = role
        self.model = model
        self.performance_history = []
        self.weight = 1.0
        
    def analyze(self, market_data: Dict) -> AgentVote:
        raise NotImplementedError
    
    def update_weight(self, reward: float):
        """Update agent weight based on performance"""
        self.performance_history.append(reward)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        recent_perf = np.mean(self.performance_history[-10:])
        self.weight = max(0.1, min(2.0, 1.0 + recent_perf))

class TrendAgent(SpecializedAgent):
    """Identifies and follows market trends using TFT outputs"""
    
    def analyze(self, market_data: Dict) -> AgentVote:
        # Simulated trend analysis logic
        price = market_data.get('price', 0)
        ma_short = market_data.get('ma_short', 0)
        ma_long = market_data.get('ma_long', 0)
        
        if ma_short > ma_long * 1.02:
            return AgentVote(
                agent_id="trend_01",
                role=AgentRole.TREND,
                action="BUY",
                confidence=0.75,
                reasoning="Strong upward trend detected",
                expected_return=0.02,
                risk_score=0.4
            )
        elif ma_short < ma_long * 0.98:
            return AgentVote(
                agent_id="trend_01",
                role=AgentRole.TREND,
                action="SELL",
                confidence=0.75,
                reasoning="Strong downward trend detected",
                expected_return=-0.02,
                risk_score=0.4
            )
        
        return AgentVote(
            agent_id="trend_01",
            role=AgentRole.TREND,
            action="HOLD",
            confidence=0.6,
            reasoning="No clear trend",
            expected_return=0.0,
            risk_score=0.2
        )

class MeanReversionAgent(SpecializedAgent):
    """Bets on prices reverting to statistical mean"""
    
    def analyze(self, market_data: Dict) -> AgentVote:
        z_score = market_data.get('z_score', 0)
        
        if z_score < -2.0:
            return AgentVote(
                agent_id="mr_01",
                role=AgentRole.MEAN_REVERSION,
                action="BUY",
                confidence=min(0.9, abs(z_score)/3),
                reasoning=f"Price significantly below mean (Z={z_score:.2f})",
                expected_return=abs(z_score) * 0.01,
                risk_score=0.3
            )
        elif z_score > 2.0:
            return AgentVote(
                agent_id="mr_01",
                role=AgentRole.MEAN_REVERSION,
                action="SELL",
                confidence=min(0.9, abs(z_score)/3),
                reasoning=f"Price significantly above mean (Z={z_score:.2f})",
                expected_return=-abs(z_score) * 0.01,
                risk_score=0.3
            )
        
        return AgentVote(
            agent_id="mr_01",
            role=AgentRole.MEAN_REVERSION,
            action="HOLD",
            confidence=0.5,
            reasoning="Price near mean",
            expected_return=0.0,
            risk_score=0.1
        )

class ArbitrageAgent(SpecializedAgent):
    """Finds statistical arbitrage opportunities"""
    
    def analyze(self, market_data: Dict) -> AgentVote:
        spread_z = market_data.get('spread_z_score', 0)
        cointegration_p = market_data.get('cointegration_p', 1.0)
        
        if cointegration_p < 0.05 and abs(spread_z) > 2.5:
            action = "BUY" if spread_z < 0 else "SELL"
            return AgentVote(
                agent_id="arb_01",
                role=AgentRole.ARBITRAGE,
                action=action,
                confidence=0.85,
                reasoning=f"Cointegrated pair divergence (Z={spread_z:.2f})",
                expected_return=abs(spread_z) * 0.005,
                risk_score=0.2
            )
        
        return AgentVote(
            agent_id="arb_01",
            role=AgentRole.ARBITRAGE,
            action="HOLD",
            confidence=0.5,
            reasoning="No arb opportunity",
            expected_return=0.0,
            risk_score=0.0
        )

class RiskAgent(SpecializedAgent):
    """Monitors and manages portfolio risk"""
    
    def analyze(self, market_data: Dict) -> AgentVote:
        var = market_data.get('var_95', 0)
        max_drawdown = market_data.get('current_drawdown', 0)
        volatility = market_data.get('volatility', 0)
        
        if var > 0.05 or max_drawdown > 0.15 or volatility > 0.5:
            return AgentVote(
                agent_id="risk_01",
                role=AgentRole.RISK,
                action="SELL",
                confidence=0.95,
                reasoning="Risk limits exceeded",
                expected_return=0.0,
                risk_score=0.9
            )
        
        return AgentVote(
            agent_id="risk_01",
            role=AgentRole.RISK,
            action="HOLD",
            confidence=0.8,
            reasoning="Risk within limits",
            expected_return=0.0,
            risk_score=0.2
        )

class MultiAgentDebateSystem:
    """Orchestrates debate between specialized agents"""
    
    def __init__(self):
        self.agents: Dict[str, SpecializedAgent] = {
            'trend_01': TrendAgent(AgentRole.TREND),
            'mr_01': MeanReversionAgent(AgentRole.MEAN_REVERSION),
            'arb_01': ArbitrageAgent(AgentRole.ARBITRAGE),
            'risk_01': RiskAgent(AgentRole.RISK)
        }
        self.debate_history = []
        
    def conduct_debate(self, market_data: Dict) -> Dict[str, Any]:
        """Run debate and reach consensus"""
        votes: List[AgentVote] = []
        
        for agent_id, agent in self.agents.items():
            try:
                vote = agent.analyze(market_data)
                votes.append(vote)
            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {e}")
        
        consensus = self._aggregate_votes(votes)
        self.debate_history.append({
            'votes': votes,
            'consensus': consensus,
            'timestamp': market_data.get('timestamp', None)
        })
        
        return consensus
    
    def _aggregate_votes(self, votes: List[AgentVote]) -> Dict[str, Any]:
        """Weighted voting system with meta-critic evaluation"""
        if not votes:
            return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'No votes'}
        
        buy_weight = sum(v.confidence * self.agents[v.agent_id.split('_')[0] + '_01'].weight 
                        for v in votes if v.action == 'BUY')
        sell_weight = sum(v.confidence * self.agents[v.agent_id.split('_')[0] + '_01'].weight 
                         for v in votes if v.action == 'SELL')
        hold_weight = sum(v.confidence * self.agents[v.agent_id.split('_')[0] + '_01'].weight 
                         for v in votes if v.action == 'HOLD')
        
        weights = {'BUY': buy_weight, 'SELL': sell_weight, 'HOLD': hold_weight}
        max_action = max(weights, key=weights.get)
        total_weight = sum(weights.values())
        
        consensus_confidence = weights[max_action] / total_weight if total_weight > 0 else 0
        
        reasoning = f"Consensus: {max_action} ({len([v for v in votes if v.action == max_action])}/{len(votes)} agents)"
        
        return {
            'action': max_action,
            'confidence': round(consensus_confidence, 3),
            'reasoning': reasoning,
            'votes_breakdown': {k: round(v, 3) for k, v in weights.items()},
            'individual_votes': [
                {
                    'agent': v.agent_id,
                    'role': v.role.value,
                    'action': v.action,
                    'confidence': v.confidence
                } for v in votes
            ]
        }
    
    def update_agents(self, rewards: Dict[str, float]):
        """Update agent weights based on actual outcomes"""
        for agent_id, reward in rewards.items():
            if agent_id in self.agents:
                self.agents[agent_id].update_weight(reward)

if __name__ == "__main__":
    # Test multi-agent system
    debate_system = MultiAgentDebateSystem()
    
    test_data = {
        'price': 100,
        'ma_short': 102,
        'ma_long': 98,
        'z_score': -2.5,
        'spread_z_score': 3.0,
        'cointegration_p': 0.03,
        'var_95': 0.03,
        'current_drawdown': 0.05,
        'volatility': 0.2,
        'timestamp': '2024-01-01'
    }
    
    result = debate_system.conduct_debate(test_data)
    print("Consensus Decision:", result)
    assert result['action'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= result['confidence'] <= 1
    print("Multi-Agent Debate System: OK")
