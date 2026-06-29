"""
TITAN 3.0 - PPO Agent Implementation
Proximal Policy Optimization for trading strategy learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    # Architecture
    hidden_dim: int = 256
    n_layers: int = 2
    
    # Training hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # PPO specific
    n_epochs: int = 10
    batch_size: int = 64
    normalize_advantage: bool = True
    
    # Environment
    n_actions: int = 5


class ActorCriticNetwork(nn.Module):
    """Actor-Critic neural network for PPO."""
    
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        super().__init__()
        
        self.config = config
        
        # Shared layers
        layers = []
        prev_dim = obs_dim
        
        for _ in range(config.n_layers):
            layers.append(nn.Linear(prev_dim, config.hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = config.hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(config.hidden_dim, action_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(config.hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value."""
        shared = self.shared_net(obs)
        
        action_logits = self.actor(shared)
        value = self.critic(shared)
        
        return action_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Sample action from policy."""
        action_logits, _ = self.forward(obs)
        
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of given actions."""
        action_logits, values = self.forward(obs)
        
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


class RolloutBuffer:
    """Buffer to store rollout experiences for PPO."""
    
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def add(self, obs: np.ndarray, action: int, reward: float, 
            done: bool, log_prob: float, value: float):
        """Add transition to buffer."""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored experiences."""
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values)
        }
    
    def clear(self):
        """Clear the buffer."""
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def get_episode_rewards(self) -> List[float]:
        """Get list of episode rewards."""
        rewards = self.episode_rewards.copy()
        self.episode_rewards = []
        return rewards


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    
    Implements PPO-Clip algorithm with generalized advantage estimation.
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int = 5,
                 config: PPOConfig = None,
                 device: str = None):
        
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize network
        self.network = ActorCriticNetwork(obs_dim, action_dim, self.config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_rewards': []
        }
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, Dict]:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = self.network.get_action(obs_tensor, deterministic)
            
            # Get value for bootstrapping
            _, value = self.network.forward(obs_tensor)
            
        return action, {
            'log_prob': log_prob.item(),
            'value': value.item()
        }
    
    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                        done: bool, log_prob: float, value: float):
        """Store transition in rollout buffer."""
        self.buffer.add(obs, action, reward, done, log_prob, value)
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                   dones: np.ndarray, next_value: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        
        # Normalize advantages
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO algorithm."""
        data = self.buffer.get()
        
        if len(data['obs']) < self.config.batch_size:
            self.buffer.clear()
            return {}
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(data['obs']).to(self.device)
        actions_tensor = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(data['log_probs']).to(self.device)
        returns_tensor = torch.FloatTensor(data['returns']).to(self.device)
        
        # Calculate returns
        advantages = self.compute_gae(
            data['rewards'],
            data['values'],
            data['dones'],
            next_value=0.0
        )
        returns = advantages + data['values']
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Create mini-batches
        dataset_size = len(data['obs'])
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        # PPO epochs
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Evaluate current policy on batch
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                   1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_bonus = entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss -
                    self.config.entropy_coef * entropy_bonus
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_bonus.item())
        
        # Clear buffer
        self.buffer.clear()
        
        # Update scheduler
        self.scheduler.step()
        
        # Store statistics
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
        avg_value_loss = np.mean(value_losses) if value_losses else 0
        avg_entropy = np.mean(entropies) if entropies else 0
        
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy'].append(avg_entropy)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }
    
    def train(self, env, total_timesteps: int = 100000, 
              log_interval: int = 10, callback=None) -> Dict:
        """
        Train the agent in the environment.
        
        Args:
            env: Gymnasium environment
            total_timesteps: Total training timesteps
            log_interval: Log every N episodes
            callback: Optional callback function called each episode
        
        Returns:
            Training history
        """
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        
        obs, _ = env.reset()
        episode_rewards = []
        current_episode_reward = 0
        
        for timestep in range(total_timesteps):
            # Select action
            action, info = self.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info_dict = env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.store_transition(
                obs, action, reward, done,
                info['log_prob'], info['value']
            )
            
            current_episode_reward += reward
            obs = next_obs
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()
                
                # Update after each episode
                update_info = self.update()
                
                # Log progress
                if len(episode_rewards) % log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-log_interval:])
                    print(f"Timestep {timestep}: Avg Reward = {avg_reward:.3f}")
                    
                    if update_info:
                        print(f"  Policy Loss: {update_info['policy_loss']:.4f}, "
                              f"Value Loss: {update_info['value_loss']:.4f}")
                
                # Callback
                if callback:
                    callback(timestep, len(episode_rewards), episode_rewards[-1])
        
        # Final update
        self.update()
        
        self.training_stats['episode_rewards'] = episode_rewards
        
        print(f"\nTraining complete!")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average reward: {np.mean(episode_rewards):.3f}")
        print(f"Best reward: {max(episode_rewards):.3f}")
        
        return self.training_stats
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.training_stats = checkpoint['training_stats']
        print(f"Model loaded from {path}")
