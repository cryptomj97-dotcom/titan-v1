"""
TITAN 3.0 - Phase 5: Reinforcement Learning Engine
Modules:
- rl_environment.py: Custom Gym environment for trading
- rl_agent.py: PPO-based trading agent
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom Gym Environment for trading.
    State: [Price, Returns, Indicators..., Regime, Position, Cash]
    Action: [Buy, Hold, Sell]
    Reward: Portfolio value change + Risk penalty
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, regime_series: pd.Series, 
                 initial_capital: float = 100000.0, window_size: int = 20):
        super(TradingEnvironment, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.regime_series = regime_series.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.window_size = window_size
        
        # Define action and observation space
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space depends on number of features
        # We assume df has columns: open, high, low, close, volume + indicators
        n_features = len(df.columns)
        # Add regime (one-hot encoded approx), position, cash_ratio
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features + 3,), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = 0.0
        self.shares = 0
        self.net_worth = 0.0
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_capital
        self.shares = 0
        self.net_worth = self.initial_capital
        self.history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Get window of data
        end_idx = self.current_step
        start_idx = end_idx - self.window_size
        
        # Flatten recent window stats (mean, std, last) to keep obs size manageable
        window_data = self.df.iloc[start_idx:end_idx]
        
        # Simple approach: take last row + regime + position info
        current_row = self.df.iloc[self.current_step].values.astype(np.float32)
        regime_val = float(self.regime_series.iloc[self.current_step])
        pos_val = float(self.shares)
        cash_val = float(self.balance / self.initial_capital)
        
        obs = np.concatenate([current_row, [regime_val, pos_val, cash_val]])
        return obs

    def step(self, action: int):
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute action
        if action == 1:  # Buy
            max_shares = int(self.balance // current_price)
            if max_shares > 0:
                self.shares += max_shares
                self.balance -= max_shares * current_price
        elif action == 2:  # Sell
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        # Calculate net worth
        self.net_worth = self.balance + (self.shares * current_price)
        
        # Reward: Change in net worth normalized
        prev_net_worth = self.history[-1]['net_worth'] if self.history else self.initial_capital
        reward = (self.net_worth - prev_net_worth) / self.initial_capital
        
        # Penalty for holding during high volatility or wrong regime (simplified)
        # In full version, this would be more complex
        
        done = self.current_step >= len(self.df) - 1
        if done:
            reward += (self.net_worth / self.initial_capital)  # Terminal bonus
            
        self.current_step += 1
        
        info = {
            'net_worth': self.net_worth,
            'shares': self.shares,
            'balance': self.balance,
            'price': current_price
        }
        
        self.history.append(info)
        
        return self._get_observation(), reward, done, False, info

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_capital
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}")

class PPOAgent:
    """
    Simplified PPO Agent implementation using PyTorch.
    In production, use stable-baselines3.
    """
    def __init__(self, env: TradingEnvironment, learning_rate: float = 3e-4, gamma: float = 0.99):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.device = "cpu" # Default to CPU
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.torch = torch
            self.nn = nn
            self.optim = optim
            self._build_network()
        except ImportError:
            logger.warning("PyTorch not installed. RL Agent disabled.")
            self.policy = None

    def _build_network(self):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        
        # Actor-Critic Network
        self.policy = self.nn.Sequential(
            self.nn.Linear(obs_dim, 128),
            self.nn.ReLU(),
            self.nn.Linear(128, 64),
            self.nn.ReLU(),
            self.nn.Linear(64, act_dim)  # logits
        ).to(self.device)
        
        self.value_net = self.nn.Sequential(
            self.nn.Linear(obs_dim, 128),
            self.nn.ReLU(),
            self.nn.Linear(128, 64),
            self.nn.ReLU(),
            self.nn.Linear(64, 1)
        ).to(self.device)
        
        self.optimizer = self.optim.Adam(list(self.policy.parameters()) + list(self.value_net.parameters()), lr=self.lr)

    def select_action(self, state: np.ndarray, deterministic=False):
        if self.policy is None:
            return 1 # Hold if no model
        
        state_tensor = self.torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.policy(state_tensor)
            if deterministic:
                action = logits.argmax(dim=1).item()
            else:
                dist = self.torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
        return action

    def train_step(self, states, actions, rewards, next_states, dones):
        if self.policy is None:
            return 0.0
            
        states = self.torch.FloatTensor(states).to(self.device)
        actions = self.torch.LongTensor(actions).to(self.device)
        rewards = self.torch.FloatTensor(rewards).to(self.device)
        next_states = self.torch.FloatTensor(next_states).to(self.device)
        dones = self.torch.FloatTensor(dones).to(self.device)
        
        # Simple Policy Gradient update (simplified PPO for brevity)
        logits = self.policy(states)
        values = self.value_net(states).squeeze()
        
        log_probs = self.torch.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Advantage estimation (simplified)
        with self.torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            target_values = rewards + (1 - dones) * self.gamma * next_values
        
        advantages = target_values - values
        
        actor_loss = -(selected_log_probs * advantages.detach()).mean()
        critic_loss = ((values - target_values) ** 2).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
