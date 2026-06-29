# TITAN 3.0 - Automated Trading System

## Overview

TITAN 3.0 is an advanced automated trading system featuring:

- **Advanced Mathematical Frameworks**: Topological Data Analysis, statistical features, and signal processing
- **Resilient Global Data Pipeline**: Multi-source data ingestion with circuit breakers and caching
- **Automated Strategy Lifecycle**: Strategy generation, backtesting, and optimization
- **Alternative Data Integration**: Sentiment analysis and alternative data fusion

## Project Structure

```
titan_3.0/
├── core/                    # Core infrastructure
│   ├── config/             # Configuration management
│   ├── logging/            # Structured logging
│   └── exceptions/         # Exception hierarchy
├── data/                   # Data pipeline
│   ├── ingestion/          # Data ingestion engine
│   ├── sources/            # Data source implementations
│   └── processors/         # Data validation & cleaning
├── strategies/             # Trading strategies
│   ├── base/               # Base strategy interfaces
│   ├── generators/         # Strategy generation
│   └── optimizers/         # Walk-forward optimization
├── ml/                     # Machine learning
│   ├── features/           # Feature engineering
│   ├── regime_detection/   # Market regime detection
│   └── rl/                 # Reinforcement learning
├── alt_data/               # Alternative data
│   ├── sentiment/          # News/social sentiment
│   └── satellite/          # Satellite data processing
├── execution/              # Trade execution
│   ├── risk/               # Risk management
│   └── order_manager/      # Order management
├── api/                    # REST API
└── frontend/               # Dashboard UI
```

## Installation

```bash
# Clone the repository
cd titan_3.0

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for advanced features
pip install PyWavelets hmmlearn
```

## Quick Start

```python
from titan_3.0 import (
    get_config,
    create_ingestion_engine,
    create_feature_engine,
    create_regime_detector
)
from datetime import datetime, timedelta

# Initialize configuration
config = get_config()

# Create data ingestion engine
engine = create_ingestion_engine()

# Fetch historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

data = engine.fetch_price_data(
    symbol='AAPL',
    start_date=start_date,
    end_date=end_date
)

# Generate features
feature_engine = create_feature_engine()
features = feature_engine.create_features(data, feature_sets=['all'])

# Detect market regimes
regime_detector = create_regime_detector(method='ensemble')
returns = data['close'].pct_change()
regimes = regime_detector.detect_regimes(returns, data['close'])

print(f"Detected regimes: {regimes.unique()}")
print(f"Feature count: {len(features.columns)}")
```

## Configuration

Create a `config.yaml` file:

```yaml
environment: development
log_level: INFO

data_sources:
  yahoo:
    provider: yahoo
    cache_enabled: true
    cache_ttl: 3600
  
  alphavantage:
    provider: alphavantage
    api_key: YOUR_API_KEY
    timeout: 30

risk:
  max_position_size: 0.1
  max_portfolio_risk: 0.02
  stop_loss_pct: 0.05

strategy:
  lookback_period: 252
  min_sharpe_ratio: 1.0
  walk_forward_ratio: 0.7

ml:
  regime_detection_method: tda
  n_regimes: 3
  rl_algorithm: ppo

execution:
  paper_trading: true
  broker: alpaca
```

## Key Features

### 1. Resilient Data Pipeline
- Multi-source data ingestion (Yahoo Finance, Alpha Vantage, Polygon)
- Circuit breaker pattern for API failures
- In-memory caching with TTL
- Automatic fallback to alternative sources

### 2. Advanced Feature Engineering
- 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Statistical features (rolling statistics, z-scores, entropy)
- Advanced mathematical transforms (Fourier, wavelets, Hurst exponent)

### 3. Regime Detection
- TDA-based regime detection
- Hidden Markov Models
- Volatility-based regimes
- Ensemble methods with voting

### 4. Risk Management
- Position sizing
- Stop-loss and take-profit
- Value at Risk (VaR)
- Maximum drawdown limits

## Testing

```bash
# Run tests
pytest tests/ -v --cov=titan_3.0

# Run with coverage report
pytest tests/ --cov-report=html
```

## Development

```bash
# Code formatting
black titan_3.0/

# Linting
flake8 titan_3.0/

# Type checking
mypy titan_3.0/
```

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. 
Past performance is not indicative of future results. Always do your own research before 
making investment decisions.
