# TITAN Deep Learning Module

## Advanced Self-Learning Architecture

This module implements state-of-the-art deep learning algorithms for autonomous quantitative trading.

### Core Components

#### 1. Temporal Fusion Transformer (TFT)
**File:** `temporal_fusion_transformer.py`

State-of-the-art attention-based architecture for multi-horizon probabilistic forecasting.

**Key Features:**
- Multi-head self-attention for temporal dependencies
- Gated Residual Networks (GRN) for non-linear processing
- Quantile outputs for uncertainty estimation (VaR, CVaR)
- Interpretable attention weights for feature importance
- Handles static covariates, known future inputs, and historical data

**Usage:**
```python
from autonomous.deep_learning import TemporalFusionTransformer

model = TemporalFusionTransformer(
    input_size=10,      # Number of features
    hidden_size=64,     # Model dimension
    output_size=1,      # Prediction target
    num_heads=4,        # Attention heads
    quantiles=[0.1, 0.5, 0.9]  # Risk quantiles
)

# Forward pass
output = model(encoder_input, decoder_input)
# Returns: {'quantile_0.1': ..., 'quantile_0.5': ..., 'quantile_0.9': ..., 'attention_weights': ...}
```

#### 2. Dynamic Graph Neural Network (GNN)
**File:** `graph_neural_network.py`

Models inter-asset correlations as a dynamic graph structure that evolves with market regimes.

**Key Features:**
- Dynamic adjacency matrix learning from node features
- Graph convolution for spatial (cross-asset) relationships
- Spatio-temporal processing combining GNN with attention
- Captures contagion effects and sector rotations
- Adaptive to changing correlation structures

**Usage:**
```python
from autonomous.deep_learning import GraphNeuralNetwork

model = GraphNeuralNetwork(
    num_assets=10,      # Number of assets in portfolio
    input_size=20,      # Features per asset
    hidden_size=64,
    output_size=1       # Expected return prediction
)

# Forward pass
output, adj_matrix = model(asset_features)
# adj_matrix shows learned correlation structure
```

#### 3. Online Continual Learner with EWC
**File:** `online_learner.py`

Enables continuous learning from streaming data without catastrophic forgetting.

**Key Features:**
- Elastic Weight Consolidation (EWC) to preserve important weights
- Fisher Information Matrix for parameter importance
- Task consolidation after each market regime
- Forgetting measurement for monitoring
- Critical for adapting to changing markets while retaining historical knowledge

**Usage:**
```python
from autonomous.deep_learning import OnlineLearner, SimpleMLP

model = SimpleMLP(input_size=10, hidden_size=32, output_size=1)
learner = OnlineLearner(model, ewc_lambda=1000.0)

# Train on new data stream
loss = learner.train_on_batch(data, target, use_ewc=True)

# Consolidate knowledge after regime change
learner.consolidate_knowledge(dataloader)

# Check forgetting on previous regimes
forget_measure = learner.get_forgetting_measure(test_loader)
```

#### 4. TimeGAN (Generative Simulator)
**File:** `generative_simulator.py`

Generates synthetic market scenarios for stress testing and data augmentation.

**Key Features:**
- Temporal GAN for realistic time series generation
- Embedder-recoverer architecture for stability
- Black swan scenario generation
- Stress testing with controlled severity
- Augments limited historical data

**Usage:**
```python
from autonomous.deep_learning import TimeGAN

gan = TimeGAN(
    input_size=5,       # OHLCV features
    hidden_size=32,
    seq_len=50          # Sequence length
)

# Train on historical data
gan.train_embedding(real_data, epochs=1000)
gan.train_gan(real_data, epochs=1000)

# Generate synthetic scenarios
synthetic = gan.generate_synthetic(num_samples=100)

# Generate extreme stress scenarios
stress_scenario = gan.generate_stress_scenario(base_data, severity=3.0)
```

#### 5. Neural Architecture Search (NAS)
**File:** `neural_arch_search.py`

Automatically discovers optimal network architectures using differentiable search.

**Key Features:**
- DARTS-inspired differentiable architecture search
- Bi-level optimization (weights + architecture)
- Automatic discovery of optimal operations
- Eliminates manual hyperparameter tuning
- Adapts network complexity to market conditions

**Usage:**
```python
from autonomous.deep_learning import NASModel, ArchitectureSearcher

model = NASModel(
    input_size=10,
    hidden_size=32,
    output_size=1,
    num_cells=3,
    num_nodes=4
)

searcher = ArchitectureSearcher(model)

# Run architecture search
architecture_summary = searcher.search(
    train_loader, 
    val_loader, 
    epochs=50
)

# Review discovered architecture
for edge, operation in architecture_summary.items():
    print(f"{edge}: {operation}")
```

## Integration with TITAN System

These modules integrate seamlessly with the existing TITAN autonomous pipeline:

```python
# In strategy_generator.py or backtester.py
from autonomous.deep_learning import (
    TemporalFusionTransformer,
    GraphNeuralNetwork,
    OnlineLearner,
    TimeGAN,
    NASModel
)

# Use TFT for forecasting
forecast_model = TemporalFusionTransformer(...)

# Use GNN for portfolio-level alpha
gnn_model = GraphNeuralNetwork(...)

# Enable online learning
learner = OnlineLearner(forecast_model)

# Generate stress scenarios for backtesting
gan = TimeGAN(...)
stress_scenarios = gan.generate_stress_scenario(historical_data)

# Auto-optimize architecture
nas = NASModel(...)
searcher = ArchitectureSearcher(nas)
optimal_arch = searcher.search(train_data, val_data)
```

## Requirements

See `requirements.txt` for full list:
- PyTorch >= 2.0.0
- torch-geometric >= 2.3.0 (for GNN)
- pytorch-lightning >= 2.0.0
- Additional ML libraries

## Performance Notes

- **GPU Recommended:** TFT and GNN benefit significantly from CUDA acceleration
- **Batch Size:** Use batch sizes of 32-128 for stable training
- **Sequence Length:** 20-100 time steps typical for financial data
- **Learning Rate:** Start with 1e-4 for TFT, 1e-3 for GNN

## Research References

1. Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
2. Kipf, T.N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks"
3. Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in neural networks" (EWC)
4. Yoon, J., et al. (2019). "Time-Series Generative Adversarial Networks"
5. Liu, H., et al. (2019). "DARTS: Differentiable Architecture Search"
