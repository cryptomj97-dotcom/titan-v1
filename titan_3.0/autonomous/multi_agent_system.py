"""
TITAN Multi-Agent Collaborative Intelligence Module
Implements specialized agents (Trend, Mean-Reversion, Risk) with weighted voting and debate logic.
"""
import time
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    timestamp: float
    price: float
    volume: float
    volatility: float
    bid: float
    ask: float
    order_book_imbalance: float = 0.0

@dataclass
class AgentDecision:
    agent_name: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    target_price: float = 0.0
    stop_loss: float = 0.0
    timestamp: float = 0.0

@dataclass
class ConsensusResult:
    final_action: str
    weighted_confidence: float
    supporting_agents: List[str]
    dissenting_agents: List[str]
    reasoning_summary: str
    timestamp: float

class BaseAgent:
    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        
    def analyze(self, state: MarketState, history: List[MarketState]) -> AgentDecision:
        raise NotImplementedError

class TrendFollowingAgent(BaseAgent):
    def __init__(self):
        super().__init__("TrendFollower", "momentum")
        
    def analyze(self, state: MarketState, history: List[MarketState]) -> AgentDecision:
        if len(history) < 5:
            return AgentDecision(self.name, "HOLD", 0.1, "Insufficient data", timestamp=time.time())
            
        prices = [h.price for h in history[-10:]]
        sma_short = np.mean(prices[-3:])
        sma_long = np.mean(prices[-7:])
        
        momentum = (prices[-1] - prices[-5]) / prices[-5]
        
        if sma_short > sma_long * 1.002 and momentum > 0.005:
            return AgentDecision(
                self.name, "BUY", min(0.95, 0.5 + abs(momentum)*10), 
                f"Strong upward trend detected. Momentum: {momentum:.4f}",
                target_price=state.price * 1.02,
                stop_loss=state.price * 0.99,
                timestamp=time.time()
            )
        elif sma_short < sma_long * 0.998 and momentum < -0.005:
            return AgentDecision(
                self.name, "SELL", min(0.95, 0.5 + abs(momentum)*10),
                f"Strong downward trend detected. Momentum: {momentum:.4f}",
                target_price=state.price * 0.98,
                stop_loss=state.price * 1.01,
                timestamp=time.time()
            )
            
        return AgentDecision(self.name, "HOLD", 0.3, "No clear trend", timestamp=time.time())

class MeanReversionAgent(BaseAgent):
    def __init__(self):
        super().__init__("MeanReverter", "statistical_arbitrage")
        
    def analyze(self, state: MarketState, history: List[MarketState]) -> AgentDecision:
        if len(history) < 20:
            return AgentDecision(self.name, "HOLD", 0.1, "Insufficient data", timestamp=time.time())
            
        prices = [h.price for h in history[-20:]]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            return AgentDecision(self.name, "HOLD", 0.1, "Zero volatility", timestamp=time.time())
            
        z_score = (state.price - mean_price) / std_price
        
        if z_score < -2.0:
            confidence = min(0.9, abs(z_score) / 5)
            return AgentDecision(
                self.name, "BUY", confidence,
                f"Price significantly below mean. Z-Score: {z_score:.2f}",
                target_price=mean_price,
                stop_loss=state.price * 0.98,
                timestamp=time.time()
            )
        elif z_score > 2.0:
            confidence = min(0.9, abs(z_score) / 5)
            return AgentDecision(
                self.name, "SELL", confidence,
                f"Price significantly above mean. Z-Score: {z_score:.2f}",
                target_price=mean_price,
                stop_loss=state.price * 1.02,
                timestamp=time.time()
            )
            
        return AgentDecision(self.name, "HOLD", 0.2, f"Normal distribution. Z-Score: {z_score:.2f}", timestamp=time.time())

class RiskManagementAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskManager", "capital_preservation")
        
    def analyze(self, state: MarketState, history: List[MarketState]) -> AgentDecision:
        # Veto power based on volatility and market conditions
        if state.volatility > 0.05:  # Extreme volatility
            return AgentDecision(
                self.name, "HOLD", 0.99,
                f"CRITICAL: Excessive volatility ({state.volatility:.2f}). Halting trades.",
                timestamp=time.time()
            )
            
        if state.order_book_imbalance > 0.8 or state.order_book_imbalance < -0.8:
            return AgentDecision(
                self.name, "HOLD", 0.95,
                f"WARNING: Severe order book imbalance ({state.order_book_imbalance:.2f})",
                timestamp=time.time()
            )
            
        return AgentDecision(self.name, "HOLD", 0.1, "Risk parameters normal", timestamp=time.time())

class AgentCouncil:
    def __init__(self):
        self.agents = [
            TrendFollowingAgent(),
            MeanReversionAgent(),
            RiskManagementAgent()
        ]
        self.weights = {
            "TrendFollower": 0.4,
            "MeanReverter": 0.4,
            "RiskManager": 1.0  # Risk manager has veto power
        }
        
    def conduct_debate(self, state: MarketState, history: List[MarketState]) -> ConsensusResult:
        decisions = []
        for agent in self.agents:
            try:
                decision = agent.analyze(state, history)
                decisions.append(decision)
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                
        # Check for Risk Manager Veto
        risk_decision = next((d for d in decisions if d.agent_name == "RiskManager"), None)
        if risk_decision and risk_decision.confidence > 0.9 and risk_decision.action == "HOLD":
            return ConsensusResult(
                final_action="HOLD",
                weighted_confidence=risk_decision.confidence,
                supporting_agents=["RiskManager"],
                dissenting_agents=[a.name for a in self.agents if a.name != "RiskManager"],
                reasoning_summary=risk_decision.reasoning,
                timestamp=time.time()
            )
            
        # Weighted Voting
        vote_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        supporters = {"BUY": [], "SELL": [], "HOLD": []}
        
        for d in decisions:
            if d.agent_name == "RiskManager":
                continue
            weight = self.weights.get(d.agent_name, 0.5)
            score = d.confidence * weight
            vote_scores[d.action] += score
            supporters[d.action].append(d.agent_name)
            
        # Determine Winner
        winner = max(vote_scores, key=vote_scores.get)
        total_score = sum(vote_scores.values())
        final_confidence = vote_scores[winner] / total_score if total_score > 0 else 0
        
        return ConsensusResult(
            final_action=winner,
            weighted_confidence=final_confidence,
            supporting_agents=supporters[winner],
            dissenting_agents=[a for a in ["TrendFollower", "MeanReverter"] if a not in supporters[winner]],
            reasoning_summary=f"Consensus reached via weighted voting. {winner} selected with {final_confidence:.2f} confidence.",
            timestamp=time.time()
        )
