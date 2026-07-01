"""
1. Multi-Agent Debate & Consensus System
Implements specialized agents, weighted voting, and meta-critic evaluation.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real-time market data structure"""
    timestamp: float
    price: float
    volume: float
    volatility: float
    bid: float
    ask: float
    symbol: str = "BTCUSD"

@dataclass
class TradingSignal:
    """Trading signal from pattern scanner"""
    signal_id: str
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: float
    features: Dict[str, float]

@dataclass
class AgentDecision:
    """Decision from individual agent"""
    agent_name: str
    action: str
    confidence: float
    reasoning: str
    timestamp: float

@dataclass
class ConsensusResult:
    """Final consensus from agent council"""
    final_action: str
    weighted_confidence: float
    supporting_agents: List[str]
    dissenting_agents: List[str]
    timestamp: float
    debate_log: List[str]

class SignalPool:
    """Thread-safe pool for real-time trading signals"""
    
    def __init__(self, max_age_seconds: float = 30.0):
        self.signals: Dict[str, TradingSignal] = {}
        self.max_age_seconds = max_age_seconds
        self.lock = threading.Lock()
        
    def add_signal(self, signal: TradingSignal) -> None:
        """Add a new signal to the pool"""
        with self.lock:
            self.signals[signal.signal_id] = signal
            
    def get_signals(self, symbol: Optional[str] = None) -> List[TradingSignal]:
        """Get all valid signals, optionally filtered by symbol"""
        with self.lock:
            current_time = time.time()
            valid_signals = []
            
            for signal in self.signals.values():
                # Check if signal is still fresh
                if current_time - signal.timestamp <= self.max_age_seconds:
                    if symbol is None or signal.symbol == symbol:
                        valid_signals.append(signal)
                else:
                    # Mark for removal
                    pass
                    
            # Clean up expired signals
            self._cleanup_expired(current_time)
            return valid_signals
            
    def _cleanup_expired(self, current_time: float) -> None:
        """Remove expired signals"""
        expired_ids = [
            sid for sid, sig in self.signals.items() 
            if current_time - sig.timestamp > self.max_age_seconds
        ]
        for sid in expired_ids:
            del self.signals[sid]
    
    def clear(self) -> None:
        """Clear all signals"""
        with self.lock:
            self.signals.clear()

class Agent:
    """Base class for trading agents"""
    
    def __init__(self, name: str, expertise: str, weight: float = 1.0):
        self.name = name
        self.expertise = expertise
        self.weight = weight  # Voting weight
        
    def analyze(self, market_data: MarketData, signals: List[TradingSignal]) -> AgentDecision:
        """Analyze market conditions and return decision"""
        raise NotImplementedError("Subclasses must implement analyze()")

class TrendFollowingAgent(Agent):
    """Agent specializing in trend detection and momentum trading"""
    
    def __init__(self):
        super().__init__("TrendFollower", "trend_analysis", weight=1.2)
        self.lookback_periods = 20
        
    def analyze(self, market_data: MarketData, signals: List[TradingSignal]) -> AgentDecision:
        """Detect trend direction based on recent signals and price action"""
        recent_signals = [
            s for s in signals 
            if time.time() - s.timestamp < 10.0 and s.symbol == market_data.symbol
        ]
        
        if not recent_signals:
            return AgentDecision(
                agent_name=self.name,
                action="HOLD",
                confidence=0.3,
                reasoning="Insufficient recent signals for trend analysis",
                timestamp=time.time()
            )
        
        buy_signals = sum(s.strength * s.confidence for s in recent_signals if s.direction == "BUY")
        sell_signals = sum(s.strength * s.confidence for s in recent_signals if s.direction == "SELL")
        
        total_signal_strength = buy_signals + sell_signals
        
        if total_signal_strength < 0.1:
            return AgentDecision(
                agent_name=self.name,
                action="HOLD",
                confidence=0.4,
                reasoning="Weak signal strength",
                timestamp=time.time()
            )
        
        # Determine trend direction
        if buy_signals > sell_signals * 1.3:
            action = "BUY"
            confidence = min(0.95, 0.5 + (buy_signals - sell_signals) * 0.15)
            reasoning = f"Strong upward trend detected (BUY:{buy_signals:.2f} vs SELL:{sell_signals:.2f})"
        elif sell_signals > buy_signals * 1.3:
            action = "SELL"
            confidence = min(0.95, 0.5 + (sell_signals - buy_signals) * 0.15)
            reasoning = f"Strong downward trend detected (SELL:{sell_signals:.2f} vs BUY:{buy_signals:.2f})"
        else:
            action = "HOLD"
            confidence = 0.4
            reasoning = f"Conflicting trends (BUY:{buy_signals:.2f} vs SELL:{sell_signals:.2f})"
            
        return AgentDecision(
            agent_name=self.name,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=time.time()
        )

class MeanReversionAgent(Agent):
    """Agent specializing in mean reversion and statistical arbitrage"""
    
    def __init__(self):
        super().__init__("MeanReversion", "statistical_arbitrage", weight=1.1)
        self.z_score_threshold = 2.0
        
    def analyze(self, market_data: MarketData, signals: List[TradingSignal]) -> AgentDecision:
        """Detect mean reversion opportunities based on price deviation"""
        # Calculate implied z-score from bid-ask spread and volatility
        mid_price = (market_data.bid + market_data.ask) / 2
        spread_ratio = abs(market_data.price - mid_price) / mid_price
        
        if market_data.volatility == 0:
            return AgentDecision(
                agent_name=self.name,
                action="HOLD",
                confidence=0.2,
                reasoning="Zero volatility - no mean reversion opportunity",
                timestamp=time.time()
            )
        
        z_score = spread_ratio / market_data.volatility
        
        if z_score > self.z_score_threshold:
            # Price significantly above mean - expect reversion down
            if market_data.price > mid_price:
                action = "SELL"
                confidence = min(0.9, 0.5 + (z_score - self.z_score_threshold) * 0.1)
                reasoning = f"Price {z_score:.2f}σ above mean - strong reversion signal"
            else:
                action = "BUY"
                confidence = min(0.9, 0.5 + (z_score - self.z_score_threshold) * 0.1)
                reasoning = f"Price {z_score:.2f}σ below mean - strong reversion signal"
        elif z_score < -self.z_score_threshold:
            # Extreme opposite case
            action = "HOLD"
            confidence = 0.3
            reasoning = f"Extreme z-score {z_score:.2f} - waiting for confirmation"
        else:
            action = "HOLD"
            confidence = 0.3
            reasoning = f"Z-score {z_score:.2f} within normal range"
            
        return AgentDecision(
            agent_name=self.name,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=time.time()
        )

class RiskManagementAgent(Agent):
    """Agent specializing in risk assessment and veto power"""
    
    def __init__(self):
        super().__init__("RiskManager", "risk_control", weight=2.0)  # Higher weight for veto
        self.volatility_threshold = 0.05
        self.max_spread_ratio = 0.02
        
    def analyze(self, market_data: MarketData, signals: List[TradingSignal]) -> AgentDecision:
        """Assess market risk and potentially veto trades"""
        risks = []
        
        # Check volatility
        if market_data.volatility > self.volatility_threshold:
            risks.append(f"Excessive volatility: {market_data.volatility:.4f}")
        
        # Check bid-ask spread
        spread_ratio = (market_data.ask - market_data.bid) / market_data.price
        if spread_ratio > self.max_spread_ratio:
            risks.append(f"Wide spread: {spread_ratio:.4f}")
        
        # Check for extreme price movements
        if len(signals) > 10:
            recent_buy = sum(1 for s in signals[-10:] if s.direction == "BUY")
            recent_sell = sum(1 for s in signals[-10:] if s.direction == "SELL")
            if abs(recent_buy - recent_sell) > 7:
                risks.append("Extreme signal imbalance detected")
        
        if risks:
            return AgentDecision(
                agent_name=self.name,
                action="HOLD",
                confidence=min(0.95, 0.7 + len(risks) * 0.08),
                reasoning=f"RISK VETO: {'; '.join(risks)}",
                timestamp=time.time()
            )
        
        return AgentDecision(
            agent_name=self.name,
            action="HOLD",  # Risk manager doesn't initiate, only vetoes
            confidence=0.6,
            reasoning="Market conditions within acceptable risk parameters",
            timestamp=time.time()
        )

class AgentCouncil:
    """Multi-Agent Debate System with weighted voting and meta-critic"""
    
    def __init__(self):
        self.agents = [
            TrendFollowingAgent(),
            MeanReversionAgent(),
            RiskManagementAgent()
        ]
        self.meta_critic_threshold = 0.85
        self.debate_history: List[ConsensusResult] = []
        self.lock = threading.Lock()
        
    def conduct_debate(self, market_data: MarketData, signals: List[TradingSignal]) -> ConsensusResult:
        """Conduct debate among agents and reach consensus"""
        decisions = []
        debate_log = []
        
        # Each agent analyzes independently
        for agent in self.agents:
            try:
                decision = agent.analyze(market_data, signals)
                decisions.append(decision)
                debate_log.append(f"{agent.name}: {decision.action} (conf: {decision.confidence:.2f}) - {decision.reasoning}")
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                debate_log.append(f"{agent.name}: ERROR - {str(e)}")
        
        # Check for Risk Manager veto first
        risk_decisions = [d for d in decisions if d.agent_name == "RiskManager"]
        if risk_decisions and risk_decisions[0].confidence >= self.meta_critic_threshold:
            consensus = ConsensusResult(
                final_action="HOLD",
                weighted_confidence=risk_decisions[0].confidence,
                supporting_agents=["RiskManager"],
                dissenting_agents=[d.agent_name for d in decisions if d.agent_name != "RiskManager"],
                timestamp=time.time(),
                debate_log=debate_log
            )
            self.debate_history.append(consensus)
            return consensus
        
        # Aggregate non-risk decisions with weights
        trade_decisions = [d for d in decisions if d.agent_name != "RiskManager"]
        
        buy_weighted = sum(
            d.confidence * next((a.weight for a in self.agents if a.name == d.agent_name), 1.0)
            for d in trade_decisions if d.action == "BUY"
        )
        sell_weighted = sum(
            d.confidence * next((a.weight for a in self.agents if a.name == d.agent_name), 1.0)
            for d in trade_decisions if d.action == "SELL"
        )
        hold_weighted = sum(
            d.confidence * next((a.weight for a in self.agents if a.name == d.agent_name), 1.0)
            for d in trade_decisions if d.action == "HOLD"
        )
        
        total_weight = buy_weighted + sell_weighted + hold_weighted
        
        if total_weight == 0:
            consensus = ConsensusResult(
                final_action="HOLD",
                weighted_confidence=0.0,
                supporting_agents=[],
                dissenting_agents=[],
                timestamp=time.time(),
                debate_log=debate_log
            )
        else:
            # Determine winner
            actions = {"BUY": buy_weighted, "SELL": sell_weighted, "HOLD": hold_weighted}
            final_action = max(actions, key=actions.get)
            winning_weight = actions[final_action]
            
            supporting_agents = [
                d.agent_name for d in trade_decisions if d.action == final_action
            ]
            dissenting_agents = [
                d.agent_name for d in trade_decisions if d.action != final_action
            ]
            
            weighted_confidence = winning_weight / total_weight
            
            consensus = ConsensusResult(
                final_action=final_action,
                weighted_confidence=min(0.95, weighted_confidence),
                supporting_agents=supporting_agents,
                dissenting_agents=dissenting_agents,
                timestamp=time.time(),
                debate_log=debate_log
            )
        
        self.debate_history.append(consensus)
        
        # Keep only recent history
        if len(self.debate_history) > 100:
            self.debate_history = self.debate_history[-50:]
        
        return consensus
    
    def get_recent_debates(self, count: int = 10) -> List[ConsensusResult]:
        """Get recent debate results"""
        return self.debate_history[-count:]
