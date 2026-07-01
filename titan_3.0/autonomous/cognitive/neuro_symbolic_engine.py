"""
3. Neuro-Symbolic Fusion Engine
Combines neural network pattern recognition with symbolic economic rules
for explainable and constrained decision making.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SymbolicRule:
    """Economic or risk management rule"""
    rule_id: str
    description: str
    condition: str  # Human-readable condition
    action: str  # Action to take if condition met
    priority: int  # Higher priority rules execute first
    enabled: bool = True

class NeuroSymbolicEngine:
    """
    Neuro-Symbolic Reasoning Engine
    Fuses neural network outputs with hard symbolic constraints
    """
    
    def __init__(self):
        self.symbolic_rules = self._initialize_symbolic_rules()
        self.economic_laws = self._initialize_economic_laws()
        self.rule_violations_log: List[Dict] = []
        
    def _initialize_symbolic_rules(self) -> Dict[str, SymbolicRule]:
        """Initialize core symbolic rules"""
        return {
            'no_trading_during_extreme_volatility': SymbolicRule(
                rule_id='RULE_001',
                description='Prevent trading during extreme volatility events',
                condition='volatility > 0.05',
                action='HOLD',
                priority=10
            ),
            'max_position_size': SymbolicRule(
                rule_id='RULE_002',
                description='Limit maximum position size per trade',
                condition='position_size > 0.1',
                action='REDUCE_POSITION',
                priority=9
            ),
            'stop_loss_enforcement': SymbolicRule(
                rule_id='RULE_003',
                description='Enforce stop loss on all positions',
                condition='loss > 0.05',
                action='CLOSE_POSITION',
                priority=10
            ),
            'no_trading_wide_spread': SymbolicRule(
                rule_id='RULE_004',
                description='Avoid trading when bid-ask spread is too wide',
                condition='spread_ratio > 0.02',
                action='HOLD',
                priority=8
            ),
            'leverage_limit': SymbolicRule(
                rule_id='RULE_005',
                description='Enforce maximum leverage ratio',
                condition='leverage > 2.0',
                action='REDUCE_LEVERAGE',
                priority=9
            )
        }
    
    def _initialize_economic_laws(self) -> List[Dict]:
        """Initialize fundamental economic laws that cannot be violated"""
        return [
            {
                'law_id': 'LAW_001',
                'description': 'No negative prices',
                'check': lambda price: price > 0
            },
            {
                'law_id': 'LAW_002',
                'description': 'Bid must be less than ask',
                'check': lambda bid, ask: bid < ask
            },
            {
                'law_id': 'LAW_003',
                'description': 'Position size cannot exceed capital',
                'check': lambda position, capital: position <= capital
            }
        ]
    
    def apply_constraints(
        self, 
        neural_decision: Dict[str, Any], 
        market_context: Any
    ) -> Dict[str, Any]:
        """
        Apply symbolic rules and economic laws to neural network output
        
        Args:
            neural_decision: Raw decision from neural network
            market_context: Current market data object
            
        Returns:
            Constrained decision with explanations
        """
        constrained_decision = neural_decision.copy()
        violations = []
        applied_rules = []
        
        # Extract market context attributes
        volatility = getattr(market_context, 'volatility', 0.0)
        bid = getattr(market_context, 'bid', 0.0)
        ask = getattr(market_context, 'ask', 0.0)
        price = getattr(market_context, 'price', 0.0)
        
        # Check economic laws first (non-negotiable)
        for law in self.economic_laws:
            try:
                if law['law_id'] == 'LAW_001' and price <= 0:
                    constrained_decision['action'] = 'HOLD'
                    constrained_decision['confidence'] = 0.0
                    violations.append(f"ECONOMIC_LAW_VIOLATION: {law['description']}")
                elif law['law_id'] == 'LAW_002' and bid >= ask and bid > 0 and ask > 0:
                    constrained_decision['action'] = 'HOLD'
                    violations.append(f"ECONOMIC_LAW_VIOLATION: {law['description']}")
            except Exception as e:
                logger.warning(f"Error checking economic law {law['law_id']}: {e}")
        
        # Apply symbolic rules by priority
        sorted_rules = sorted(
            [r for r in self.symbolic_rules.values() if r.enabled],
            key=lambda x: x.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            rule_triggered = False
            
            # Evaluate rule conditions
            if rule.rule_id == 'RULE_001':  # Extreme volatility
                if volatility > 0.05:
                    rule_triggered = True
                    if constrained_decision.get('action') != 'HOLD':
                        constrained_decision['action'] = 'HOLD'
                        constrained_decision['confidence'] *= 0.3
                        applied_rules.append(rule.rule_id)
            
            elif rule.rule_id == 'RULE_002':  # Max position size
                proposed_size = constrained_decision.get('position_size', 0.05)
                if proposed_size > 0.1:
                    rule_triggered = True
                    constrained_decision['position_size'] = 0.1
                    applied_rules.append(rule.rule_id)
            
            elif rule.rule_id == 'RULE_003':  # Stop loss
                if 'stop_loss' in constrained_decision and 'entry_price' in constrained_decision:
                    entry = constrained_decision['entry_price']
                    stop = constrained_decision['stop_loss']
                    if constrained_decision.get('action') == 'BUY':
                        if stop > entry * 0.95:  # Stop loss too tight
                            constrained_decision['stop_loss'] = entry * 0.95
                            applied_rules.append(rule.rule_id)
                    elif constrained_decision.get('action') == 'SELL':
                        if stop < entry * 1.05:  # Stop loss too tight
                            constrained_decision['stop_loss'] = entry * 1.05
                            applied_rules.append(rule.rule_id)
            
            elif rule.rule_id == 'RULE_004':  # Wide spread
                if price > 0:
                    spread_ratio = (ask - bid) / price
                    if spread_ratio > 0.02:
                        rule_triggered = True
                        if constrained_decision.get('action') != 'HOLD':
                            constrained_decision['action'] = 'HOLD'
                            constrained_decision['confidence'] *= 0.5
                            applied_rules.append(rule.rule_id)
            
            elif rule.rule_id == 'RULE_005':  # Leverage limit
                proposed_leverage = constrained_decision.get('leverage', 1.0)
                if proposed_leverage > 2.0:
                    rule_triggered = True
                    constrained_decision['leverage'] = 2.0
                    applied_rules.append(rule.rule_id)
            
            if rule_triggered:
                violations.append(f"RULE_TRIGGERED: {rule.rule_id} - {rule.description}")
        
        # Add explanation to decision
        original_reasoning = constrained_decision.get('reasoning', '')
        if applied_rules:
            constrained_decision['reasoning'] = (
                f"{original_reasoning} | SYMBOLIC CONSTRAINTS APPLIED: {', '.join(applied_rules)}"
            )
        if violations:
            constrained_decision['violations'] = violations
            self.rule_violations_log.append({
                'timestamp': time.time(),
                'violations': violations,
                'original_decision': neural_decision,
                'constrained_decision': constrained_decision
            })
        
        # Keep only recent violations log
        if len(self.rule_violations_log) > 100:
            self.rule_violations_log = self.rule_violations_log[-50:]
        
        constrained_decision['symbolic_validated'] = True
        constrained_decision['applied_rules_count'] = len(applied_rules)
        
        return constrained_decision
    
    def validate_market_data(self, market_context: Any) -> List[str]:
        """Validate market data against economic laws"""
        violations = []
        
        price = getattr(market_context, 'price', 0.0)
        bid = getattr(market_context, 'bid', 0.0)
        ask = getattr(market_context, 'ask', 0.0)
        
        if price <= 0:
            violations.append("Invalid price: must be positive")
        
        if bid > 0 and ask > 0 and bid >= ask:
            violations.append("Invalid spread: bid must be less than ask")
        
        return violations
    
    def get_rule_status(self) -> Dict[str, Any]:
        """Get current status of all symbolic rules"""
        return {
            'total_rules': len(self.symbolic_rules),
            'enabled_rules': sum(1 for r in self.symbolic_rules.values() if r.enabled),
            'recent_violations': len(self.rule_violations_log),
            'rules': {
                rule_id: {
                    'description': rule.description,
                    'priority': rule.priority,
                    'enabled': rule.enabled
                }
                for rule_id, rule in self.symbolic_rules.items()
            }
        }
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a symbolic rule"""
        if rule_id in self.symbolic_rules:
            self.symbolic_rules[rule_id].enabled = True
            logger.info(f"Enabled rule: {rule_id}")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a symbolic rule"""
        if rule_id in self.symbolic_rules:
            self.symbolic_rules[rule_id].enabled = False
            logger.info(f"Disabled rule: {rule_id}")
            return True
        return False
