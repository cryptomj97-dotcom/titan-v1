"""
TITAN 3.0 - Phase 7: Adversarial Debate System (Backend Logic)
Modules:
- debate_engine.py: Simulates Bull/Bear/Judge agents to validate signals
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Argument:
    agent: str  # "BULL", "BEAR"
    points: List[str]
    confidence: float

class DebateEngine:
    """
    Simulates an adversarial debate between Bull and Bear agents.
    A Judge agent evaluates the arguments to produce a final consensus score.
    """
    
    def __init__(self):
        self.bull_factors = ['momentum', 'breakout', 'volume_spike', 'sentiment_positive', 'macro_favorable']
        self.bear_factors = ['resistance', 'divergence', 'volume_drop', 'sentiment_negative', 'macro_risk']
    
    def generate_bull_case(self, market_data: Dict[str, Any]) -> Argument:
        points = []
        confidence = 0.5
        
        # Analyze technicals
        if market_data.get('rsi', 50) < 70:
            points.append("RSI indicates room for upward movement before overbought.")
            confidence += 0.1
            
        if market_data.get('macd_hist', 0) > 0:
            points.append("MACD histogram is positive, signaling bullish momentum.")
            confidence += 0.15
            
        if market_data.get('price') > market_data.get('sma_50', 0):
            points.append("Price is trading above 50-day SMA, confirming uptrend.")
            confidence += 0.1
            
        if market_data.get('sentiment_score', 0) > 0.2:
            points.append("Alternative data sentiment is positive.")
            confidence += 0.1
            
        return Argument(agent="BULL", points=points, confidence=min(confidence, 0.95))

    def generate_bear_case(self, market_data: Dict[str, Any]) -> Argument:
        points = []
        confidence = 0.5
        
        # Analyze technicals
        if market_data.get('rsi', 50) > 70:
            points.append("RSI is overbought, suggesting a potential pullback.")
            confidence += 0.15
            
        if market_data.get('macd_hist', 0) < 0:
            points.append("MACD histogram is negative, signaling bearish momentum.")
            confidence += 0.15
            
        if market_data.get('price') < market_data.get('sma_200', float('inf')):
            points.append("Price is below 200-day SMA, indicating long-term weakness.")
            confidence += 0.1
            
        if market_data.get('volatility_regime') == 'HIGH':
            points.append("High volatility regime increases downside risk.")
            confidence += 0.1
            
        return Argument(agent="BEAR", points=points, confidence=min(confidence, 0.95))

    def judge_verdict(self, bull: Argument, bear: Argument, current_signal: str) -> Dict[str, Any]:
        """
        Weighs arguments from both sides against the current proposed signal.
        Returns a final decision and confidence.
        """
        bull_score = len(bull.points) * bull.confidence
        bear_score = len(bear.points) * bear.confidence
        
        total_score = bull_score + bear_score
        if total_score == 0:
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": "Insufficient data"}
        
        net_score = (bull_score - bear_score) / total_score
        
        reasoning = []
        if bull_score > bear_score:
            reasoning.extend(bull.points[:2]) # Top 2 points
        else:
            reasoning.extend(bear.points[:2])
            
        # Decision Logic
        if abs(net_score) < 0.1:
            decision = "HOLD"
        elif net_score > 0.2:
            decision = "BUY"
        elif net_score < -0.2:
            decision = "SELL"
        else:
            decision = "HOLD" # Weak signal
            
        # Override if contradicts strong signal without confidence
        if current_signal == "BUY" and decision == "SELL":
            reasoning.append("Debate contradicts initial signal strongly. Caution advised.")
            
        return {
            "decision": decision,
            "confidence": round(abs(net_score), 3),
            "bull_score": round(bull_score, 3),
            "bear_score": round(bear_score, 3),
            "reasoning": reasoning,
            "verdict_summary": f"Bull ({bull_score:.2f}) vs Bear ({bear_score:.2f})"
        }

    def run_debate(self, market_data: Dict[str, Any], initial_signal: str) -> Dict[str, Any]:
        bull_case = self.generate_bull_case(market_data)
        bear_case = self.generate_bear_case(market_data)
        verdict = self.judge_verdict(bull_case, bear_case, initial_signal)
        
        return {
            "bull_case": bull_case,
            "bear_case": bear_case,
            "verdict": verdict
        }
