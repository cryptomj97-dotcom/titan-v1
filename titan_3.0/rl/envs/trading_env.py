"""
TITAN 3.0 - Reinforcement Learning Trading Environment
Gymnasium-compatible environment for training RL agents on trading tasks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """RL action space for trading."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4


@dataclass
class TradingState:
    """Current state of the trading environment."""
    balance: float
    equity: float
    position: int  # -1 (short), 0 (none), 1 (long)
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    step: int
    total_trades: int


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for trading.
    
    State Space:
        - Price features (normalized)
        - Technical indicators
        - Market regime
        - Account state (balance, position, PnL)
    
    Action Space:
        - Discrete: HOLD, BUY, SELL, CLOSE_LONG, CLOSE_SHORT
    
    Reward Function:
        - Realized PnL from closed trades
        - Penalty for drawdowns
        - Bonus for Sharpe ratio improvement
        - Transaction cost penalty
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self,
                 df: pd.DataFrame,
                 features: Dict[str, pd.Series],
                 regimes: pd.Series,
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001,
                 max_steps: int = None,
                 reward_scaling: float = 1.0,
                 render_mode: str = None):
        
        super().__init__()
        
        self.df = df.copy()
        self.features = features
        self.regimes = regimes
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps or len(df)
        self.reward_scaling = reward_scaling
        self.render_mode = render_mode
        
        # Calculate dimensions
        n_features = len(self.features)
        n_regime_categories = 4  # bull, bear, sideways, volatile
        account_state_dim = 6  # balance, equity, position, entry_price, unrealized_pnl, realized_pnl
        
        self.observation_dim = n_features + n_regime_categories + account_state_dim
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.equity_curve = [initial_balance]
        
        # Normalization parameters
        self.feature_means = None
        self.feature_stds = None
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        # Get current price data
        current_idx = self.current_step
        
        # Normalize features
        feature_values = []
        for name, series in self.features.items():
            value = series.iloc[current_idx] if hasattr(series, 'iloc') else series[current_idx]
            feature_values.append(value)
        
        features_array = np.array(feature_values, dtype=np.float32)
        
        # Normalize features if we have statistics
        if self.feature_means is not None:
            features_array = (features_array - self.feature_means) / (self.feature_stds + 1e-8)
        
        # One-hot encode regime
        current_regime = self.regimes.iloc[current_idx] if hasattr(self.regimes, 'iloc') else self.regimes[current_idx]
        regime_map = {'bull': 0, 'bear': 1, 'sideways': 2, 'volatile': 3}
        regime_idx = regime_map.get(current_regime, 2)
        regime_onehot = np.zeros(4, dtype=np.float32)
        regime_onehot[regime_idx] = 1.0
        
        # Account state
        self.equity = self.balance + self.unrealized_pnl
        account_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.equity / self.initial_balance,   # Normalized equity
            float(self.position),                  # Position: -1, 0, 1
            self.entry_price / max(self.df['close'].max(), 1e-8),  # Normalized entry price
            self.unrealized_pnl / self.initial_balance,  # Normalized unrealized PnL
            self.realized_pnl / self.initial_balance     # Normalized realized PnL
        ], dtype=np.float32)
        
        # Concatenate all observations
        observation = np.concatenate([features_array, regime_onehot, account_state])
        
        return observation
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.equity_curve = [self.initial_balance]
        
        # Compute normalization statistics from full dataset
        feature_data = []
        for name, series in self.features.items():
            feature_data.append(series.values)
        
        feature_matrix = np.column_stack(feature_data)
        self.feature_means = np.nanmean(feature_matrix, axis=0)
        self.feature_stds = np.nanstd(feature_matrix, axis=0)
        
        observation = self._get_observation()
        
        info = {
            'balance': self.balance,
            'equity': self.equity,
            'step': self.current_step
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action from action space (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE_LONG, 4=CLOSE_SHORT)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.df['close'].iloc[self.current_step]
        
        # Execute action
        reward = 0.0
        
        if action == Action.BUY and self.position == 0:
            # Open long position
            self.position = 1
            self.entry_price = current_price * (1 + self.transaction_cost)
        
        elif action == Action.SELL and self.position == 0:
            # Open short position
            self.position = -1
            self.entry_price = current_price * (1 - self.transaction_cost)
        
        elif action == Action.CLOSE_LONG and self.position == 1:
            # Close long position
            pnl = (current_price - self.entry_price) * (self.balance / self.entry_price)
            pnl -= current_price * (self.balance / self.entry_price) * self.transaction_cost
            self.balance += pnl
            self.realized_pnl += pnl
            self.total_trades += 1
            reward = pnl / self.initial_balance
            self.position = 0
            self.entry_price = 0.0
        
        elif action == Action.CLOSE_SHORT and self.position == -1:
            # Close short position
            pnl = (self.entry_price - current_price) * (self.balance / self.entry_price)
            pnl -= current_price * (self.balance / self.entry_price) * self.transaction_cost
            self.balance += pnl
            self.realized_pnl += pnl
            self.total_trades += 1
            reward = pnl / self.initial_balance
            self.position = 0
            self.entry_price = 0.0
        
        # Update unrealized PnL
        if self.position == 1:
            self.unrealized_pnl = (current_price - self.entry_price) * (self.balance / self.entry_price)
        elif self.position == -1:
            self.unrealized_pnl = (self.entry_price - current_price) * (self.balance / self.entry_price)
        else:
            self.unrealized_pnl = 0.0
        
        # Update equity curve
        self.equity = self.balance + self.unrealized_pnl
        self.equity_curve.append(self.equity)
        
        # Calculate reward components
        # 1. Realized PnL reward (already added above for closed trades)
        # 2. Unrealized PnL change
        if len(self.equity_curve) > 1:
            equity_change = (self.equity - self.equity_curve[-2]) / self.initial_balance
            reward += equity_change * 0.1  # Small weight for unrealized changes
        
        # 3. Drawdown penalty
        if len(self.equity_curve) > 1:
            running_max = max(self.equity_curve[:-1])
            current_drawdown = (running_max - self.equity) / running_max if running_max > 0 else 0
            if current_drawdown > 0.05:  # Penalty only if drawdown > 5%
                reward -= current_drawdown * 0.5
        
        # 4. Transaction cost penalty for unnecessary trading
        if action == Action.HOLD and self.position != 0:
            reward -= 0.0001  # Small penalty for holding through costs
        
        # Scale reward
        reward *= self.reward_scaling
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if self.balance <= 0:
            terminated = True  # Bankruptcy
            reward -= 1.0  # Large penalty
        
        if self.current_step >= self.max_steps - 1:
            truncated = True
        
        # Get new observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'step': self.current_step,
            'current_price': current_price
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            logger.info(f"Step: {self.current_step}")
            logger.info(f"Balance: ${self.balance:,.2f}")
            logger.info(f"Equity: ${self.equity:,.2f}")
            logger.info(f"Position: {self.position}")
            logger.info(f"Realized PnL: ${self.realized_pnl:,.2f}")
            logger.info(f"Unrealized PnL: ${self.unrealized_pnl:,.2f}")
            logger.info(f"Total Trades: {self.total_trades}")
            logger.info("-" * 40)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from equity curve."""
        if len(self.equity_curve) < 2:
            return {}
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Total return
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Win rate
        if self.total_trades > 0:
            # Simplified: assume half are wins if positive total PnL
            win_rate = 0.5 + (self.realized_pnl / (self.total_trades * self.initial_balance)) * 0.5
            win_rate = max(0, min(1, win_rate))
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'final_balance': self.balance
        }
