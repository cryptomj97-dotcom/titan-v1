"""
Dynamic Graph Neural Network (GNN) for Inter-Asset Correlation Modeling.
Models the market as a dynamic graph where assets are nodes and correlations are edges.
Updates edge weights in real-time to capture changing market regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class GraphConvolution(nn.Module):
    """Simple Graph Convolutional Layer."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj_matrix: Adjacency matrix (batch, num_nodes, num_nodes)
        """
        # Message passing: aggregate neighbor information
        aggregated = torch.matmul(adj_matrix, x)
        out = self.linear(aggregated)
        return self.dropout(F.relu(out))

class DynamicGraphLearner(nn.Module):
    """Learns dynamic adjacency matrices from node features."""
    def __init__(self, num_nodes: int, hidden_size: int, threshold: float = 0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.threshold = threshold
        
        # Learnable transformation for edge weights
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
        
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Computes dynamic adjacency matrix based on feature similarity.
        Args:
            node_features: (batch, num_nodes, hidden_size)
        Returns:
            adj_matrix: (batch, num_nodes, num_nodes) - Sparse adjacency
        """
        batch_size, _, hidden_size = node_features.shape
        
        # Compute attention-like scores between all pairs of nodes
        Q = self.query_linear(node_features)  # (batch, num_nodes, hidden)
        K = self.key_linear(node_features)    # (batch, num_nodes, hidden)
        
        # Scaled dot-product similarity
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(node_features.device)
        
        # Apply threshold to create sparse graph
        mask = (scores > self.threshold).float()
        adj_matrix = F.softmax(scores, dim=-1) * mask
        
        # Ensure diagonal is zero (no self-loops unless explicitly desired)
        adj_matrix = adj_matrix * (1 - torch.eye(self.num_nodes, device=node_features.device).unsqueeze(0))
        
        return adj_matrix

class GraphNeuralNetwork(nn.Module):
    """
    Multi-layer Graph Neural Network for asset correlation modeling.
    Combines dynamic graph learning with graph convolutions.
    """
    def __init__(self, 
                 num_assets: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_assets = num_assets
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Dynamic graph learner
        self.graph_learner = DynamicGraphLearner(num_assets, hidden_size)
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(hidden_size, hidden_size, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Asset features (batch, num_assets, input_size)
            
        Returns:
            output: Processed features (batch, num_assets, output_size)
            adj_matrix: Learned dynamic adjacency matrix
        """
        # Project input to hidden size
        h = self.input_proj(x)
        
        # Learn dynamic graph structure
        adj_matrix = self.graph_learner(h)
        
        # Apply graph convolutions
        for gcn in self.gcn_layers:
            h_new = gcn(h, adj_matrix)
            # Residual connection
            h = self.layer_norm(h + h_new)
            
        # Output projection
        output = self.output_layer(h)
        
        return output, adj_matrix

class SpatioTemporalGNN(nn.Module):
    """
    Combines GNN for spatial (cross-asset) relationships with TFT/Transformer for temporal dynamics.
    This is the full model for multi-asset forecasting.
    """
    def __init__(self,
                 num_assets: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_gnn_layers: int = 2,
                 num_temporal_heads: int = 4):
        super().__init__()
        
        # Spatial processing (GNN)
        self.gnn = GraphNeuralNetwork(
            num_assets, input_size, hidden_size, hidden_size, num_gnn_layers
        )
        
        # Temporal processing (simplified self-attention for now)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_temporal_heads, dropout=0.1
        )
        
        # Final output
        self.final_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, 
                x: torch.Tensor,  # (batch, time, num_assets, features)
                ) -> Dict[str, torch.Tensor]:
        """
        Processes spatio-temporal data.
        """
        batch_size, time_steps, num_assets, _ = x.shape
        
        # Reshape for GNN: treat each time step separately
        # (batch*time, num_assets, features)
        x_reshaped = x.view(batch_size * time_steps, num_assets, -1)
        
        # Apply GNN at each time step
        gnn_output, adj_matrix = self.gnn(x_reshaped)
        
        # Reshape back: (batch, time, num_assets, hidden)
        gnn_output = gnn_output.view(batch_size, time_steps, num_assets, -1)
        
        # Permute for temporal attention: (num_assets, batch, time, hidden)
        # We'll process each asset's time series through attention
        outputs = []
        for i in range(num_assets):
            asset_series = gnn_output[:, :, i, :].transpose(0, 1)  # (time, batch, hidden)
            attn_out, _ = self.temporal_attention(asset_series, asset_series, asset_series)
            outputs.append(attn_out.transpose(0, 1))  # (batch, time, hidden)
            
        # Stack assets: (batch, time, num_assets, hidden)
        final_repr = torch.stack(outputs, dim=2)
        
        # Project to output
        predictions = self.final_layer(final_repr)
        
        return {
            'predictions': predictions,
            'adjacency_matrix': adj_matrix.view(batch_size, time_steps, num_assets, num_assets)
        }

# Verification
if __name__ == "__main__":
    print("Initializing Dynamic Graph Neural Network...")
    
    num_assets = 5
    input_size = 10
    hidden_size = 64
    output_size = 1
    
    model = GraphNeuralNetwork(num_assets, input_size, hidden_size, output_size)
    
    # Dummy data: batch of 4, 5 assets, 10 features each
    x = torch.randn(4, num_assets, input_size)
    
    output, adj = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency sparsity: {(adj < 0.01).sum().item() / adj.numel() * 100:.1f}%")
    print("GNN Initialization Successful.")
