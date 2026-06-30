"""Risk management module with limits and kill switches."""

import numpy as np
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass
class PositionLimit:
    """Position limit configuration."""
    asset: str
    max_quantity: float
    max_notional: float
    max_pct_portfolio: float
    

@dataclass
class LossLimit:
    """Loss limit configuration."""
    daily_max_loss: float  # Absolute dollar amount
    daily_max_loss_pct: float  # Percentage of portfolio
    weekly_max_loss: float
    monthly_max_loss: float
    max_drawdown_pct: float  # Maximum peak-to-trough decline


@dataclass
class ConcentrationLimit:
    """Concentration limit configuration."""
    max_single_position_pct: float  # Max % in single asset
    max_sector_exposure_pct: float  # Max % in single sector
    max_correlation_sum: float  # Max sum of correlations


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    monthly_pnl: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    var_95: float  # 1-day 95% VaR
    cvar_95: float  # 1-day 95% CVaR
    leverage: float
    concentration_top3: float  # Top 3 positions as % of portfolio


class RiskManager:
    """
    Comprehensive risk management system.
    
    Monitors positions, P&L, drawdowns, and enforces limits
    with automatic kill switch functionality.
    """
    
    def __init__(self, 
                 initial_capital: float,
                 position_limits: List[PositionLimit],
                 loss_limits: LossLimit,
                 concentration_limits: ConcentrationLimit,
                 var_confidence: float = 0.95):
        """
        Initialize risk manager.
        
        Args:
            initial_capital: Starting capital
            position_limits: Per-asset position limits
            loss_limits: Drawdown and loss limits
            concentration_limits: Concentration limits
            var_confidence: Confidence level for VaR calculations
        """
        self.initial_capital = initial_capital
        self.position_limits = {limit.asset: limit for limit in position_limits}
        self.loss_limits = loss_limits
        self.concentration_limits = concentration_limits
        self.var_confidence = var_confidence
        
        # State tracking
        self.positions: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.current_prices: Dict[str, float] = {}
        self.portfolio_value = initial_capital
        self.peak_portfolio_value = initial_capital
        
        # P&L tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.trade_log: List[Dict] = []
        
        # Kill switch state
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
        self.kill_switch_timestamp: Optional[datetime] = None
        
        # Historical returns for VaR
        self.daily_returns: List[float] = []
        self.max_lookback_days = 252
        
        logger.info(f"Risk manager initialized with capital: ${initial_capital:,.2f}")
        
    def update_price(self, asset: str, price: float):
        """Update current price for an asset."""
        self.current_prices[asset] = price
        
        # Recalculate portfolio value
        self._recalculate_portfolio_value()
        
    def _recalculate_portfolio_value(self):
        """Recalculate total portfolio value."""
        cash = self.initial_capital + self.daily_pnl + self.weekly_pnl + self.monthly_pnl
        
        position_value = 0.0
        for asset, qty in self.positions.items():
            if asset in self.current_prices:
                position_value += qty * self.current_prices[asset]
                
        self.portfolio_value = cash + position_value
        
        # Update peak
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
            
    def add_position(self, asset: str, quantity: float, price: float) -> Dict:
        """
        Add or increase a position with risk checks.
        
        Args:
            asset: Asset identifier
            quantity: Quantity to add (positive for long, negative for short)
            price: Current price
            
        Returns:
            Risk check result
        """
        # Check kill switch first
        if self.kill_switch_active:
            return {
                'approved': False,
                'reason': 'kill_switch_active',
                'message': f"Trading halted: {self.kill_switch_reason}"
            }
        
        # Calculate new position
        current_qty = self.positions.get(asset, 0.0)
        new_qty = current_qty + quantity
        new_notional = abs(new_qty * price)
        
        # Check position limits
        if asset in self.position_limits:
            limit = self.position_limits[asset]
            
            if abs(new_qty) > limit.max_quantity:
                return {
                    'approved': False,
                    'reason': 'position_limit_quantity',
                    'message': f"Quantity {abs(new_qty):.2f} exceeds limit {limit.max_quantity:.2f}"
                }
                
            if new_notional > limit.max_notional:
                return {
                    'approved': False,
                    'reason': 'position_limit_notional',
                    'message': f"Notional ${new_notional:,.2f} exceeds limit ${limit.max_notional:,.2f}"
                }
        
        # Check concentration limits
        portfolio_pct = new_notional / self.portfolio_value if self.portfolio_value > 0 else 1.0
        
        if portfolio_pct > self.concentration_limits.max_single_position_pct:
            return {
                'approved': False,
                'reason': 'concentration_limit',
                'message': f"Position {portfolio_pct:.2%} exceeds max {self.concentration_limits.max_single_position_pct:.2%}"
            }
        
        # Approve and update
        self.positions[asset] = new_qty
        self.entry_prices[asset] = price
        self.current_prices[asset] = price
        
        logger.info(f"Position added: {asset} {quantity:+.2f} @ {price:.4f}, "
                   f"new qty: {new_qty:.2f}")
        
        return {
            'approved': True,
            'new_quantity': new_qty,
            'new_notional': new_notional,
            'portfolio_pct': portfolio_pct
        }
    
    def remove_position(self, asset: str, quantity: float, price: float) -> Dict:
        """Remove or reduce a position."""
        current_qty = self.positions.get(asset, 0.0)
        
        if quantity <= 0:
            return {'approved': False, 'reason': 'invalid_quantity'}
        
        # Reduce position
        new_qty = current_qty - quantity
        
        if new_qty < 0:
            # Can't reduce more than held
            quantity = current_qty
            new_qty = 0
        
        # Calculate P&L on reduction
        if asset in self.entry_prices:
            pnl = (price - self.entry_prices[asset]) * quantity
            self.daily_pnl += pnl
        
        self.positions[asset] = new_qty
        
        if new_qty == 0:
            del self.positions[asset]
            if asset in self.entry_prices:
                del self.entry_prices[asset]
        
        logger.info(f"Position reduced: {asset} {quantity:.2f} @ {price:.4f}, "
                   f"remaining: {new_qty:.2f}")
        
        return {
            'approved': True,
            'reduced_by': quantity,
            'remaining': new_qty
        }
    
    def check_risk_limits(self) -> Dict:
        """
        Check all risk limits and return status.
        
        Returns:
            Risk status with any violations
        """
        self._recalculate_portfolio_value()
        
        # Calculate drawdown
        current_dd = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        
        # Check loss limits
        violations = []
        risk_level = RiskLevel.NORMAL
        
        if self.daily_pnl < -self.loss_limits.daily_max_loss:
            violations.append(f"Daily loss ${abs(self.daily_pnl):,.2f} > limit ${self.loss_limits.daily_max_loss:,.2f}")
            risk_level = RiskLevel.HALT
            
        if current_dd > self.loss_limits.max_drawdown_pct:
            violations.append(f"Drawdown {current_dd:.2%} > limit {self.loss_limits.max_drawdown_pct:.2%}")
            risk_level = RiskLevel.HALT
            
        if self.daily_pnl / self.initial_capital < -self.loss_limits.daily_max_loss_pct:
            violations.append(f"Daily loss {self.daily_pnl/self.initial_capital:.2%} > limit {-self.loss_limits.daily_max_loss_pct:.2%}")
            risk_level = RiskLevel.HALT
        
        # Check concentration
        if self.positions:
            top_positions = sorted(
                [(asset, abs(qty * self.current_prices.get(asset, 0))) 
                 for asset, qty in self.positions.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            top3_concentration = sum(p[1] for p in top_positions) / self.portfolio_value
            
            if top3_concentration > 0.5:  # Example threshold
                violations.append(f"Top 3 concentration {top3_concentration:.2%} too high")
                if risk_level == RiskLevel.NORMAL:
                    risk_level = RiskLevel.WARNING
        
        # Trigger kill switch if needed
        if risk_level == RiskLevel.HALT and not self.kill_switch_active:
            self.activate_kill_switch(f"Risk limit breach: {'; '.join(violations)}")
        
        return {
            'risk_level': risk_level.value,
            'violations': violations,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_capital if self.initial_capital > 0 else 0,
            'current_drawdown': current_dd,
            'kill_switch_active': self.kill_switch_active
        }
    
    def activate_kill_switch(self, reason: str):
        """Activate the kill switch to halt all trading."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_timestamp = datetime.now()
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # In production, would send alerts here
        return {
            'status': 'halted',
            'reason': reason,
            'timestamp': self.kill_switch_timestamp.isoformat()
        }
    
    def deactivate_kill_switch(self, authorized_by: str) -> Dict:
        """Deactivate kill switch (requires authorization)."""
        if not self.kill_switch_active:
            return {'status': 'already_inactive'}
        
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_timestamp = None
        
        logger.warning(f"Kill switch deactivated by: {authorized_by}")
        
        return {
            'status': 'active',
            'deactivated_by': authorized_by,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical simulation.
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR as positive number (potential loss)
        """
        if len(self.daily_returns) < 30:
            # Not enough data, use parametric approximation
            if self.positions:
                # Simple volatility-based VaR
                portfolio_vol = 0.02  # Assume 2% daily vol
                z_score = 1.645 if confidence == 0.95 else 2.326
                return self.portfolio_value * portfolio_vol * z_score
            return 0.0
        
        # Historical simulation
        sorted_returns = sorted(self.daily_returns)
        index = int((1 - confidence) * len(sorted_returns))
        var_return = sorted_returns[max(0, index)]
        
        return abs(var_return) * self.portfolio_value
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics."""
        self._recalculate_portfolio_value()
        
        current_dd = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        
        # Calculate leverage
        gross_exposure = sum(abs(qty * self.current_prices.get(asset, 0)) 
                            for asset, qty in self.positions.items())
        leverage = gross_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Top 3 concentration
        if self.positions:
            position_values = sorted(
                [abs(qty * self.current_prices.get(asset, 0)) 
                 for asset, qty in self.positions.items()],
                reverse=True
            )
            top3_conc = sum(position_values[:3]) / self.portfolio_value if self.portfolio_value > 0 else 0
        else:
            top3_conc = 0
        
        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=self.portfolio_value,
            daily_pnl=self.daily_pnl,
            daily_pnl_pct=self.daily_pnl / self.initial_capital if self.initial_capital > 0 else 0,
            weekly_pnl=self.weekly_pnl,
            monthly_pnl=self.monthly_pnl,
            current_drawdown_pct=current_dd,
            max_drawdown_pct=current_dd,  # Would track historical max in production
            var_95=self.calculate_var(0.95),
            cvar_95=self.calculate_var(0.95) * 1.3,  # Simplified CVaR
            leverage=leverage,
            concentration_top3=top3_conc
        )
    
    def end_of_day_reset(self):
        """Reset daily counters at end of day."""
        # Store today's return
        if self.initial_capital > 0:
            daily_return = self.daily_pnl / self.initial_capital
            self.daily_returns.append(daily_return)
            
            # Keep only recent history
            if len(self.daily_returns) > self.max_lookback_days:
                self.daily_returns = self.daily_returns[-self.max_lookback_days:]
        
        # Reset daily P&L
        self.weekly_pnl += self.daily_pnl
        self.daily_pnl = 0.0
        
        logger.info(f"EOD reset: daily P&L ${self.daily_pnl:,.2f}, "
                   f"weekly P&L ${self.weekly_pnl:,.2f}")
    
    def export_state(self) -> Dict:
        """Export current risk state for monitoring."""
        metrics = self.get_risk_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            'metrics': {
                'portfolio_value': metrics.portfolio_value,
                'daily_pnl': metrics.daily_pnl,
                'daily_pnl_pct': metrics.daily_pnl_pct,
                'drawdown': metrics.current_drawdown_pct,
                'var_95': metrics.var_95,
                'leverage': metrics.leverage,
                'concentration_top3': metrics.concentration_top3
            },
            'positions': {
                asset: {
                    'quantity': qty,
                    'price': self.current_prices.get(asset, 0),
                    'notional': qty * self.current_prices.get(asset, 0)
                }
                for asset, qty in self.positions.items()
            }
        }
