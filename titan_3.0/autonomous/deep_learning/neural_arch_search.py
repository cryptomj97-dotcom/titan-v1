"""
Neural Architecture Search (NAS) for Automated Model Optimization.
Uses differentiable architecture search (DARTS-inspired) to find optimal network structures.
Eliminates manual hyperparameter tuning by letting the model discover its own best architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

class MixedOp(nn.Module):
    """Mixed operation representing a choice between multiple primitives."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self._ops = nn.ModuleList([
            # Convolutional operations
            nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.Sigmoid()
            ),
            # Skip connection
            nn.Identity() if in_channels == out_channels else 
                nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    nn.BatchNorm1d(out_channels)
                ),
            # Zero operation (dropout-like)
            nn.Dropout(0.5)
        ])
        
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted combination of all operations."""
        result = sum(w * op(x) for w, op in zip(weights, self._ops))
        return result

class Cell(nn.Module):
    """Neural cell with learnable architecture weights."""
    def __init__(self, num_nodes: int, in_channels: int, out_channels: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Architecture parameters (alphas) - learnable weights for each edge
        # For num_nodes=2: edges are (0->1), so 1 edge total with 4 operations
        num_edges = num_nodes * (num_nodes + 1) // 2
        self.arch_weights = nn.Parameter(torch.randn(num_edges, 4) * 0.01)
        
        # Create mixed operations for each edge - all use same in/out channels
        self._ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(i + 1):
                self._ops.append(MixedOp(in_channels, out_channels))
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with architecture weighting."""
        batch_size = x.shape[0]
        states = [x]
        
        op_idx = 0
        for i in range(1, self.num_nodes + 1):
            # Each node i receives input from all previous nodes j < i
            node_input = torch.zeros(batch_size, self.out_channels, device=x.device)
            for j in range(i):
                if op_idx < len(self._ops):
                    edge_weight = F.softmax(self.arch_weights[op_idx], dim=-1)
                    # Ensure states[j] has correct input size
                    state_j = states[j]
                    if state_j.shape[-1] != self.in_channels:
                        # Project to correct size if needed
                        state_j = F.linear(state_j, torch.randn(self.out_channels, state_j.shape[-1], device=x.device))
                    node_input = node_input + self._ops[op_idx](state_j, edge_weight)
                    op_idx += 1
            states.append(node_input)
            
        # Concatenate all node outputs (excluding initial input)
        return torch.cat(states[1:], dim=-1), self.arch_weights

class NASModel(nn.Module):
    """
    Differentiable Architecture Search Model.
    Jointly optimizes network weights and architecture parameters.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_cells: int = 3, num_nodes: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        
        self.cells = nn.ModuleList([
            Cell(num_nodes, hidden_size, hidden_size)
            for _ in range(num_cells)
        ])
        
        # Output size after concatenation: hidden_size * num_nodes (not num_nodes+1)
        output_features = hidden_size * num_nodes
        self.classifier = nn.Linear(output_features, output_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns predictions and architecture weights for analysis."""
        h = self.input_proj(x)
        
        arch_weights_list = []
        for cell in self.cells:
            h, arch_w = cell(h)
            arch_weights_list.append(arch_w)
            
        # Classifier expects 2D input: (batch, features)
        if h.dim() > 2:
            h = h.mean(dim=1)  # Average over sequence dimension if present
            
        logits = self.classifier(h)
        
        return logits, arch_weights_list
    
    def get_architecture_summary(self) -> Dict[str, float]:
        """Returns dominant operations in the learned architecture."""
        summary = {}
        for i, cell in enumerate(self.cells):
            weights = F.softmax(cell.arch_weights, dim=-1).detach().cpu().numpy()
            ops = ['Linear+ReLU', 'Linear+Sigmoid', 'Skip', 'Dropout']
            for j, w in enumerate(weights):
                dominant_op = ops[np.argmax(w)]
                summary[f'cell{i}_edge{j}'] = dominant_op
        return summary

class ArchitectureSearcher:
    """
    Manages the bi-level optimization process for NAS.
    Alternates between updating network weights and architecture parameters.
    """
    def __init__(self, model: NASModel, weight_lr: float = 1e-3, arch_lr: float = 1e-2,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Separate optimizers for weights and architecture
        self.weight_optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if 'arch_weights' not in n],
            lr=weight_lr
        )
        self.arch_optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if 'arch_weights' in n],
            lr=arch_lr
        )
        
        self.loss_fn = nn.MSELoss()
        
    def step(self, train_data: torch.Tensor, train_target: torch.Tensor,
             val_data: torch.Tensor, val_target: torch.Tensor):
        """
        Performs one step of bi-level optimization:
        1. Update architecture parameters on validation loss
        2. Update network weights on training loss
        """
        self.model.train()
        
        # Step 1: Update architecture (using validation gradient)
        self.arch_optimizer.zero_grad()
        val_pred, _ = self.model(val_data.to(self.device))
        val_loss = self.loss_fn(val_pred, val_target.to(self.device))
        val_loss.backward()
        self.arch_optimizer.step()
        
        # Step 2: Update weights (using training gradient)
        self.weight_optimizer.zero_grad()
        train_pred, _ = self.model(train_data.to(self.device))
        train_loss = self.loss_fn(train_pred, train_target.to(self.device))
        train_loss.backward()
        self.weight_optimizer.step()
        
        return train_loss.item(), val_loss.item()
    
    def search(self, train_loader, val_loader, epochs: int = 50) -> Dict:
        """Run full architecture search process."""
        print("Starting Neural Architecture Search...")
        
        for epoch in range(epochs):
            total_train_loss = 0
            total_val_loss = 0
            batches = 0
            
            # Simple iteration (in practice, would alternate more carefully)
            for (train_x, train_y), (val_x, val_y) in zip(train_loader, val_loader):
                t_loss, v_loss = self.step(train_x, train_y, val_x, val_y)
                total_train_loss += t_loss
                total_val_loss += v_loss
                batches += 1
                
            avg_train = total_train_loss / max(batches, 1)
            avg_val = total_val_loss / max(batches, 1)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")
                
        # Return discovered architecture
        return self.model.get_architecture_summary()

# Verification
if __name__ == "__main__":
    print("Initializing Neural Architecture Search...")
    
    input_size = 10
    hidden_size = 32
    output_size = 1
    
    model = NASModel(input_size, hidden_size, output_size, num_cells=2, num_nodes=3)
    searcher = ArchitectureSearcher(model, weight_lr=0.01, arch_lr=0.01)
    
    # Create dummy data
    train_data = torch.randn(64, input_size)
    train_target = torch.randn(64, output_size)
    val_data = torch.randn(20, input_size)
    val_target = torch.randn(20, output_size)
    
    # Quick search test
    print("\n--- Running Mini Architecture Search ---")
    summary = searcher.search(
        [(train_data, train_target)],
        [(val_data, val_target)],
        epochs=20
    )
    
    print("\n--- Discovered Architecture Summary ---")
    for key, value in summary.items():
        print(f"{key}: {value}")
        
    print("\nNAS Initialization Successful.")
