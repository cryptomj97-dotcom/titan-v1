"""
Online Continual Learning with Elastic Weight Consolidation (EWC).
Enables the model to learn from new data streams without forgetting previous knowledge.
Critical for adapting to changing market regimes while retaining historical patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List
import copy

class ElasticWeightConsolidation:
    """
    Implements Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
    Penalizes changes to parameters that were important for previous tasks.
    """
    def __init__(self, model: nn.Module, fisher_diagonal: Optional[Dict[str, torch.Tensor]] = None, 
                 prev_params: Optional[Dict[str, torch.Tensor]] = None, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_diagonal = fisher_diagonal if fisher_diagonal is not None else {}
        self.prev_params = prev_params if prev_params is not None else {}
        
    def compute_fisher_information(self, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device):
        """
        Computes the Fisher Information Matrix diagonal approximation.
        This identifies which parameters are most important for the current task.
        """
        self.model.train()
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            self.model.zero_grad()
            output = self.model(data)
            
            # For regression, use MSE; for classification, use CrossEntropy
            if isinstance(output, dict):
                output = output.get('predictions', list(output.values())[0])
                
            loss = loss_fn(output, target)
            loss.backward()
            
            # Square gradients and accumulate (Fisher diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) / len(dataloader)
                    
        self.fisher_diagonal = fisher
        # Store current parameters as "previous" for next task
        self.prev_params = {name: param.clone().detach() for name, param in self.model.named_parameters() if param.requires_grad}
        
    def ewc_loss(self) -> torch.Tensor:
        """
        Computes the EWC regularization term.
        L_ewc = lambda * sum( F_i * (theta_i - theta*_i)^2 )
        """
        if not self.fisher_diagonal or not self.prev_params:
            return torch.tensor(0.0)
            
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.fisher_diagonal and name in self.prev_params:
                fisher = self.fisher_diagonal[name]
                prev_param = self.prev_params[name]
                loss += (fisher * (param - prev_param).pow(2)).sum()
                
        return self.ewc_lambda * loss

class OnlineLearner:
    """
    Manages online learning lifecycle with EWC.
    Handles streaming data, periodic updates, and drift detection triggers.
    """
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, 
                 ewc_lambda: float = 1000.0, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ewc = ElasticWeightConsolidation(self.model, ewc_lambda=ewc_lambda)
        self.loss_fn = nn.MSELoss()
        self.task_count = 0
        
    def train_on_batch(self, data: torch.Tensor, target: torch.Tensor, use_ewc: bool = True) -> float:
        """
        Performs a single training step on incoming data.
        If EWC is enabled, adds regularization to prevent forgetting.
        """
        self.model.train()
        data, target = data.to(self.device), target.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        
        # Handle dict outputs from complex models
        if isinstance(output, dict):
            output = output.get('predictions', list(output.values())[0])
            
        base_loss = self.loss_fn(output, target)
        
        # Add EWC penalty if applicable
        total_loss = base_loss
        if use_ewc and self.task_count > 0:
            ewc_loss = self.ewc.ewc_loss()
            total_loss = total_loss + ewc_loss
            
        total_loss.backward()
        self.optimizer.step()
        
        return base_loss.item()
    
    def consolidate_knowledge(self, dataloader: DataLoader):
        """
        After training on a new regime/task, update Fisher Information.
        This "saves" the important weights for this task.
        """
        print(f"Consolidating knowledge after task {self.task_count + 1}...")
        self.ewc.compute_fisher_information(dataloader, self.loss_fn, self.device)
        self.task_count += 1
        
    def get_forgetting_measure(self, test_loader: DataLoader) -> float:
        """
        Evaluates how much the model has forgotten on previous tasks.
        Lower is better.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if isinstance(output, dict):
                    output = output.get('predictions', list(output.values())[0])
                total_loss += self.loss_fn(output, target).item()
                
        return total_loss / len(test_loader)

# Simple MLP for testing EWC
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Verification
if __name__ == "__main__":
    print("Initializing Online Learner with EWC...")
    
    # Create model and learner
    model = SimpleMLP(input_size=10, hidden_size=32, output_size=1)
    learner = OnlineLearner(model, learning_rate=0.01, ewc_lambda=500.0)
    
    # Simulate Task 1: Train on random data
    print("\n--- Training Task 1 ---")
    data1 = torch.randn(100, 10)
    target1 = torch.randn(100, 1)
    
    for epoch in range(10):
        loss = learner.train_on_batch(data1, target1, use_ewc=False)
        if epoch % 5 == 0:
            print(f"Task 1 Epoch {epoch}, Loss: {loss:.4f}")
            
    # Consolidate knowledge after Task 1
    dataset1 = TensorDataset(data1, target1)
    loader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
    learner.consolidate_knowledge(loader1)
    
    # Simulate Task 2: Train on new distribution WITH EWC
    print("\n--- Training Task 2 (with EWC) ---")
    data2 = torch.randn(100, 10) + 2.0  # Shifted distribution
    target2 = torch.randn(100, 1) + 2.0
    
    for epoch in range(10):
        loss = learner.train_on_batch(data2, target2, use_ewc=True)
        if epoch % 5 == 0:
            print(f"Task 2 Epoch {epoch}, Loss: {loss:.4f}")
            
    # Check forgetting on Task 1 data
    print("\n--- Evaluating Forgetting ---")
    forget_loss = learner.get_forgetting_measure(loader1)
    print(f"Forgetting measure on Task 1: {forget_loss:.4f}")
    print("Online Learner with EWC Initialization Successful.")
