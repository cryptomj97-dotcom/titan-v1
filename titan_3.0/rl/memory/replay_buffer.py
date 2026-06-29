"""
TITAN 3.0 - RL Memory Package (Replay Buffer)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.
    Supports prioritized experience replay.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, transition: Dict[str, any], priority: float = 1.0):
        """Add transition to buffer with priority."""
        self.buffer.append(transition)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, prioritized: bool = False) -> List[Dict]:
        """Sample batch of transitions."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        else:
            # Uniform sampling
            indices = random.sample(range(len(self.buffer)), batch_size)
        
        return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()


class TrajectoryBuffer:
    """
    Buffer for storing complete trajectories (episodes).
    Useful for policy gradient methods.
    """
    
    def __init__(self, max_trajectories: int = 1000):
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
    
    def add_trajectory(self, trajectory: List[Dict]):
        """Add complete trajectory."""
        self.trajectories.append(trajectory)
    
    def get_all_transitions(self) -> List[Dict]:
        """Flatten all trajectories into single list of transitions."""
        transitions = []
        for trajectory in self.trajectories:
            transitions.extend(trajectory)
        return transitions
    
    def sample_trajectories(self, n: int) -> List[List[Dict]]:
        """Sample n complete trajectories."""
        if len(self.trajectories) <= n:
            return list(self.trajectories)
        return random.sample(list(self.trajectories), n)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def clear(self):
        """Clear the buffer."""
        self.trajectories.clear()
