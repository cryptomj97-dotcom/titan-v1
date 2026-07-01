"""
TITAN Neuro-Symbolic Reasoning Engine
Combines neural network pattern recognition with symbolic economic rules.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class NeuralPrediction:
    action: str
    confidence: float
    pattern_detected: str
    raw_score: float
    
@dataclass
class SymbolicRule:
    rule_id: str
    condition: str
    constraint_type: str  # VETO, MODIFY, CONFIRM
    parameters: Dict[str, float]
    active: bool = True

class EconomicRuleLibrary:
    """Library of hard-coded economic and risk management rules"""
    
    def __init__(self):
        self.rules = [
            SymbolicRule(
                rule_id="RISK_001",
                condition="volatility > threshold",
                constraint_type="VETO",
                parameters={"threshold": 0.05},
                active=True
            ),
            SymbolicRule(
                rule_id="RISK_002", 
                condition="drawdown > max_allowed",
                constraint_type="VETO",
                parameters={"max_allowed": 0.10},
                active=True
            ),
            SymbolicRule(
                rule_id="SIZE_001",
                condition="position_size > limit",
                constraint_type="MODIFY",
                parameters={"limit": 0.10},  # Max 10% of portfolio
                active=True
            ),
            SymbolicRule(
                rule_id="TREND_001",
                condition="news_event_detected",
                constraint_type="MODIFY",
                parameters={"confidence_reduction": 0.5},
                active=True
            ),
            SymbolicRule(
                rule_id="LIQ_001",
                condition="liquidity < minimum",
                constraint_type="VETO",
                parameters={"minimum": 1000000},  # $1M daily volume
                active=True
            )
        ]
        
    def get_active_rules(self) -> List[SymbolicRule]:
        return [r for r in self.rules if r.active]

class NeuroSymbolicEngine:
    """Fuses neural predictions with symbolic reasoning"""
    
    def __init__(self):
        self.rule_library = EconomicRuleLibrary()
        
    def evaluate_conditions(self, rule: SymbolicRule, market_context: Dict[str, Any]) -> bool:
        """Evaluate if a symbolic rule's conditions are met"""
        try:
            if rule.rule_id == "RISK_001":
                return market_context.get('volatility', 0) > rule.parameters['threshold']
            elif rule.rule_id == "RISK_002":
                return market_context.get('current_drawdown', 0) > rule.parameters['max_allowed']
            elif rule.rule_id == "SIZE_001":
                return market_context.get('proposed_position_size', 0) > rule.parameters['limit']
            elif rule.rule_id == "TREND_001":
                return market_context.get('news_detected', False)
            elif rule.rule_id == "LIQ_001":
                return market_context.get('daily_volume', 0) < rule.parameters['minimum']
            return False
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return False
            
    def apply_constraints(self, neural_pred: NeuralPrediction, 
                         market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply symbolic rules to neural network output"""
        
        final_decision = {
            'action': neural_pred.action,
            'confidence': neural_pred.confidence,
            'reasoning': f"Neural pattern: {neural_pred.pattern_detected}",
            'rules_applied': [],
            'vetoed': False,
            'timestamp': time.time()
        }
        
        active_rules = self.rule_library.get_active_rules()
        
        for rule in active_rules:
            if self.evaluate_conditions(rule, market_context):
                if rule.constraint_type == "VETO":
                    final_decision['action'] = 'HOLD'
                    final_decision['confidence'] = 0.99
                    final_decision['reasoning'] += f" | VETOED by {rule.rule_id}: {rule.condition}"
                    final_decision['rules_applied'].append(rule.rule_id)
                    final_decision['vetoed'] = True
                    logger.warning(f"Trade vetoed by rule {rule.rule_id}")
                    
                elif rule.constraint_type == "MODIFY":
                    if rule.rule_id == "SIZE_001":
                        final_decision['position_size'] = rule.parameters['limit']
                        final_decision['reasoning'] += f" | Position size capped by {rule.rule_id}"
                        final_decision['rules_applied'].append(rule.rule_id)
                        
                    elif rule.rule_id == "TREND_001":
                        reduction = rule.parameters['confidence_reduction']
                        final_decision['confidence'] *= (1 - reduction)
                        final_decision['reasoning'] += f" | Confidence reduced due to news event ({rule.rule_id})"
                        final_decision['rules_applied'].append(rule.rule_id)
                        
        if not final_decision['vetoed']:
            final_decision['reasoning'] += " | All symbolic constraints satisfied"
            
        return final_decision
        
    def explain_decision(self, decision: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the decision"""
        explanation = f"Action: {decision['action']} (Confidence: {decision['confidence']:.2f})\n"
        explanation += f"Reasoning: {decision['reasoning']}\n"
        
        if decision['rules_applied']:
            explanation += f"Rules Applied: {', '.join(decision['rules_applied'])}\n"
            
        if decision.get('vetoed'):
            explanation += "⚠️ TRADE BLOCKED by risk management rules\n"
            
        return explanation
