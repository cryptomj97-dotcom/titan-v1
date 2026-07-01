"""
Temporal Fusion Transformer (TFT) for Multi-Horizon Probabilistic Forecasting.
Replaces static LSTM/GRU models with attention-based architecture capable of
interpreting variable importance and providing quantile forecasts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

class TimeSeriesEmbedding(nn.Module):
    """Embeds static covariates and time-varying inputs."""
    def __init__(self, input_size: int, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.linear(x)
        return self.layer_norm(embedded)

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) as per TFT paper."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu1 = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        
        if context_size is not None:
            self.register_buffer('has_context', torch.tensor(True))
            self.context_layer = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.register_buffer('has_context', torch.tensor(False))
            self.context_layer = None
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.elu2 = nn.ELU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout2 = nn.Dropout(dropout)
        
        if input_size != output_size:
            self.skip_transform = nn.Linear(input_size, output_size)
        else:
            self.skip_transform = nn.Identity()
            
        self.gate = nn.Linear(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip = self.skip_transform(x)
        
        h = self.fc1(x)
        h = self.elu1(h)
        
        if self.context_layer is not None and context is not None:
            c = self.context_layer(context)
            h = h + c
            
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.elu2(h)
        h = self.fc3(h)
        h = self.dropout2(h)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(h))
        out = gate * h + (1 - gate) * skip
        return self.layer_norm(out)

class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability weights."""
    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = q.shape
        
        q = self.q_linear(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_linear(context)
        
        # Return output and attention weights for interpretability
        return output, attn_weights.mean(dim=1)

class TemporalFusionTransformer(nn.Module):
    """
    Full Temporal Fusion Transformer implementation.
    Handles static covariates, known future inputs, and observed historical inputs.
    Outputs quantile forecasts for risk-aware prediction.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_heads: int = 4,
                 num_encoder_steps: int = 10,
                 dropout: float = 0.1,
                 quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_encoder_steps = num_encoder_steps
        self.quantiles = quantiles
        
        # Input embeddings
        self.input_embedding = TimeSeriesEmbedding(input_size, hidden_size)
        
        # Static covariate encoders (simplified for now)
        self.static_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        
        # Temporal processing
        self.temporal_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.attention = InterpretableMultiHeadAttention(num_heads, hidden_size, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # Final processing
        self.final_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Output layers for each quantile
        self.quantile_layers = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for _ in quantiles
        ])

    def forward(self, 
                encoder_input: torch.Tensor, 
                decoder_input: torch.Tensor,
                static_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_input: (batch, encoder_seq, input_size) - Historical data
            decoder_input: (batch, decoder_seq, input_size) - Known future data
            static_context: (batch, hidden_size) - Static metadata
            
        Returns:
            Dictionary containing predictions for each quantile and attention weights
        """
        batch_size, enc_seq, _ = encoder_input.shape
        _, dec_seq, _ = decoder_input.shape
        
        # Embed inputs
        enc_emb = self.input_embedding(encoder_input)
        dec_emb = self.input_embedding(decoder_input)
        
        # Process static context
        if static_context is None:
            static_context = torch.zeros(batch_size, self.hidden_size, device=encoder_input.device)
        static_encoded = self.static_grn(static_context)
        
        # Concatenate encoder and decoder for attention
        combined_emb = torch.cat([enc_emb, dec_emb], dim=1)
        
        # Apply GRN
        temporal_features = self.temporal_grn(combined_emb, static_encoded)
        
        # Attention mechanism (causal mask for decoder part)
        total_seq = enc_seq + dec_seq
        causal_mask = torch.tril(torch.ones(total_seq, total_seq, device=encoder_input.device)).unsqueeze(0).unsqueeze(0)
        attn_output, attn_weights = self.attention(temporal_features, temporal_features, temporal_features, mask=causal_mask)
        
        # Residual connection
        attn_output = self.layer_norm1(attn_output + temporal_features)
        
        # Extract decoder outputs (last dec_seq steps)
        decoder_output = attn_output[:, -dec_seq:, :]
        
        # Final processing
        final_output = self.final_grn(decoder_output, static_encoded)
        final_output = self.layer_norm2(final_output + decoder_output)
        
        # Generate quantile predictions
        predictions = {}
        for i, q in enumerate(self.quantiles):
            pred = self.quantile_layers[i](final_output)
            predictions[f'quantile_{q}'] = pred
            
        predictions['attention_weights'] = attn_weights
        
        return predictions

class QuantileLoss(nn.Module):
    """Quantile loss function for probabilistic forecasting."""
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for q in self.quantiles:
            q_pred = preds[f'quantile_{q}']
            # Ensure shapes match
            if q_pred.shape != target.shape:
                q_pred = q_pred[:, -target.shape[1]:, :] # Align time steps if needed
                
            errors = target - q_pred
            loss_q = torch.max((q - 1) * errors, q * errors)
            total_loss += loss_q.mean()
        return total_loss

# Example usage for verification
if __name__ == "__main__":
    print("Initializing Temporal Fusion Transformer...")
    model = TemporalFusionTransformer(
        input_size=10,
        hidden_size=64,
        output_size=1,
        num_heads=4,
        num_encoder_steps=5
    )
    
    # Dummy data
    batch_size = 4
    enc_input = torch.randn(batch_size, 5, 10)
    dec_input = torch.randn(batch_size, 3, 10)
    
    output = model(enc_input, dec_input)
    
    print(f"Output keys: {output.keys()}")
    print(f"Prediction shape (median): {output['quantile_0.5'].shape}")
    print("TFT Initialization Successful.")
