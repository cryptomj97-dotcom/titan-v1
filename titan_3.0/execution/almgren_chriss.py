"""Almgren-Chriss optimal execution algorithm."""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AlmgrenChrissParams:
    """Parameters for Almgren-Chriss model."""
    sigma: float  # Volatility (daily)
    eta: float  # Temporary price impact coefficient
    gamma: float  # Permanent price impact coefficient
    risk_aversion: float  # Risk aversion parameter (lambda)
    trading_time: float  # Trading time in days (e.g., 1/252 for 1 day)


class AlmgrenChrissExecutor:
    """
    Almgren-Chriss optimal execution strategy.
    
    Minimizes implementation shortfall by balancing market impact
    against timing risk using the classic AC framework.
    """
    
    def __init__(self, params: AlmgrenChrissParams,
                 num_intervals: int = 10):
        """
        Initialize Almgren-Chriss executor.
        
        Args:
            params: Model parameters
            num_intervals: Number of trading intervals
        """
        self.params = params
        self.num_intervals = num_intervals
        self.remaining_quantity = 0.0
        self.executed_quantity = 0.0
        self.trades_executed = []
        self.current_interval = 0
        
    def compute_optimal_trajectory(self, total_quantity: float,
                                   initial_price: float) -> List[Dict]:
        """
        Compute optimal trading trajectory using AC model.
        
        The optimal strategy trades more aggressively early when
        risk aversion is high, and more evenly when market impact
        dominates.
        
        Args:
            total_quantity: Total quantity to execute
            initial_price: Starting price
            
        Returns:
            List of trade instructions with quantities per interval
        """
        Q = total_quantity
        S = initial_price
        N = self.num_intervals
        
        # Extract parameters
        sigma = self.params.sigma
        eta = self.params.eta
        gamma = self.params.gamma
        lam = self.params.risk_aversion
        tau = self.params.trading_time / N  # Time per interval
        
        # Calculate key coefficients
        # kappa determines the shape of the trading curve
        kappa = np.sqrt((lam * sigma**2 * tau) / (2 * eta))
        
        # Optimal trajectory: q_k = Q * sinh(kappa*(N-k)) / sinh(kappa*N)
        # For small kappa, this approaches linear; for large kappa, front-loaded
        
        trajectory = []
        remaining = Q
        
        for k in range(N):
            if kappa * N < 0.1:  # Linear approximation for small kappa
                q_k = Q / N
            else:
                # Full AC solution
                sinh_term = np.sinh(kappa * (N - k)) / np.sinh(kappa * N)
                q_k = Q * sinh_term
                
                # Adjust for remaining quantity in last interval
                if k == N - 1:
                    q_k = remaining
            
            # Ensure non-negative and don't exceed remaining
            q_k = max(0, min(q_k, remaining))
            
            # Calculate expected costs for this interval
            permanent_impact = gamma * q_k
            temporary_impact = eta * q_k / tau
            
            # Expected price at execution
            expected_price = S - permanent_impact * (k + 0.5) - temporary_impact
            
            trajectory.append({
                'interval': k,
                'quantity': q_k,
                'cumulative_qty': Q - remaining + q_k,
                'remaining_qty': remaining - q_k,
                'expected_price': expected_price,
                'permanent_impact': permanent_impact,
                'temporary_impact': temporary_impact,
                'time_offset': timedelta(days=self.params.trading_time * k / N)
            })
            
            remaining -= q_k
        
        self.remaining_quantity = total_quantity
        logger.info(f"AC trajectory computed: {N} intervals, "
                   f"total qty: {total_quantity}, kappa: {kappa:.4f}")
        
        return trajectory
    
    def execute_trade(self, trade_instruction: Dict, 
                     current_price: float,
                     realized_vol: Optional[float] = None) -> Dict:
        """
        Execute a single trade from the optimal trajectory.
        
        Args:
            trade_instruction: Trade details from trajectory
            current_price: Current market price
            realized_vol: Realized volatility (optional, for adaptive adjustment)
            
        Returns:
            Execution result with actual fill details
        """
        q_target = trade_instruction['quantity']
        
        if q_target <= 0 or self.remaining_quantity <= 0:
            return {
                'filled_qty': 0,
                'price': current_price,
                'status': 'skipped',
                'reason': 'no_quantity'
            }
        
        # Adaptive adjustment based on realized volatility
        if realized_vol is not None and realized_vol > self.params.sigma:
            # Trade faster if volatility is higher than expected
            adjustment_factor = min(1.5, realized_vol / self.params.sigma)
            q_execute = min(q_target * adjustment_factor, self.remaining_quantity)
        else:
            q_execute = min(q_target, self.remaining_quantity)
        
        # Calculate actual execution price with impact
        permanent_impact = self.params.gamma * q_execute
        temporary_impact = self.params.eta * q_execute / (self.params.trading_time / self.num_intervals)
        
        # Price slippage
        execution_price = current_price - permanent_impact - temporary_impact
        
        # Ensure price doesn't go negative
        execution_price = max(execution_price, current_price * 0.99)  # Max 1% discount
        
        fill_value = q_execute * execution_price
        
        self.executed_quantity += q_execute
        self.remaining_quantity -= q_execute
        self.current_interval += 1
        
        result = {
            'filled_qty': q_execute,
            'price': execution_price,
            'value': fill_value,
            'timestamp': datetime.now(),
            'interval': trade_instruction['interval'],
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact,
            'slippage_pct': (current_price - execution_price) / current_price * 100,
            'remaining_qty': self.remaining_quantity,
            'status': 'filled' if q_execute > 0 else 'partial'
        }
        
        self.trades_executed.append(result)
        
        logger.debug(f"AC trade executed: {q_execute} @ {execution_price:.4f}, "
                    f"slippage: {result['slippage_pct']:.4f}%")
        
        return result
    
    def get_implementation_shortfall(self, initial_price: float) -> Dict:
        """
        Calculate implementation shortfall metrics.
        
        Args:
            initial_price: Price at start of execution
            
        Returns:
            Dictionary with shortfall metrics
        """
        if self.executed_quantity == 0:
            return {'status': 'no_execution'}
        
        avg_execution_price = sum(
            t['filled_qty'] * t['price'] for t in self.trades_executed
        ) / self.executed_quantity
        
        # Implementation shortfall in basis points
        decision_price_value = self.executed_quantity * initial_price
        actual_cost = sum(t['value'] for t in self.trades_executed)
        
        shortfall_bps = (decision_price_value - actual_cost) / decision_price_value * 10000
        
        # Decompose shortfall
        total_permanent_impact = sum(t['permanent_impact'] * t['filled_qty'] 
                                     for t in self.trades_executed)
        total_temporary_impact = sum(t['temporary_impact'] * t['filled_qty'] 
                                     for t in self.trades_executed)
        
        return {
            'executed_quantity': self.executed_quantity,
            'average_price': avg_execution_price,
            'initial_price': initial_price,
            'shortfall_bps': shortfall_bps,
            'shortfall_value': decision_price_value - actual_cost,
            'permanent_impact_cost': total_permanent_impact,
            'temporary_impact_cost': total_temporary_impact,
            'trades_executed': len(self.trades_executed),
            'completion_pct': (1 - self.remaining_quantity / 
                              (self.executed_quantity + self.remaining_quantity)) * 100
                               if self.executed_quantity + self.remaining_quantity > 0 else 0
        }
    
    def reset(self):
        """Reset execution state."""
        self.remaining_quantity = 0.0
        self.executed_quantity = 0.0
        self.trades_executed = []
        self.current_interval = 0
        logger.info("Almgren-Chriss executor reset")


def estimate_ac_parameters(price_history: np.ndarray, 
                          volume_history: np.ndarray,
                          trading_time: float = 1/252) -> AlmgrenChrissParams:
    """
    Estimate Almgren-Chriss parameters from historical data.
    
    Args:
        price_history: Historical prices
        volume_history: Historical volumes
        trading_time: Trading horizon in years
        
    Returns:
        Estimated parameters
    """
    # Calculate daily volatility
    returns = np.diff(np.log(price_history))
    sigma = np.std(returns) * np.sqrt(252)  # Annualized
    
    # Estimate temporary impact (eta) from price-volume relationship
    # Simplified: eta ~ average price change / volume
    price_changes = np.abs(np.diff(price_history))
    avg_volume = np.mean(volume_history)
    eta = np.mean(price_changes / (avg_volume + 1e-8)) * 0.5  # Conservative estimate
    
    # Permanent impact (gamma) typically smaller than temporary
    gamma = eta * 0.3
    
    # Risk aversion parameter (typical values 0.1 to 10)
    risk_aversion = 1.0
    
    return AlmgrenChrissParams(
        sigma=sigma,
        eta=eta,
        gamma=gamma,
        risk_aversion=risk_aversion,
        trading_time=trading_time
    )
