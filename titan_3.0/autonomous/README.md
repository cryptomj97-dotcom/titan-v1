# TITAN 3.0 Autonomous Trading System

## Overview

This module implements the advanced quantitative algorithms and fully autonomous trading system architecture for TITAN 3.0.

## Modules

### 1. `quant_algorithms.py` - Advanced Quantitative Algorithms

Implements mathematical foundations for trading:

- **FractionalDifferentiation**: Memory-preserving stationarity transformation
- **RegimeDetector**: HMM-based market regime identification
- **WaveletDecomposer**: Multi-resolution signal analysis and denoising
- **CointegrationTracker**: Pairs trading with Engle-Granger test
- **HawkesProcess**: Self-exciting point process for order book modeling
- **Utility Functions**:
  - `calculate_rough_volatility()`: Rough volatility using fractional Brownian motion
  - `spectral_analysis()`: FFT-based frequency analysis
  - `kalman_filter()`: State estimation with Kalman filtering
  - `vpín_indicator()`: Volume-synchronized probability of informed trading

**Dependencies**: numpy, pandas, scipy, scikit-learn, statsmodels, hmmlearn, pywt

### 2. `strategy_generator.py` - Strategy Generation Engine

Automated strategy discovery using:

- **GeneticProgrammingEngine**: Population-based evolution of trading rules
  - Tournament selection
  - Parameter crossover
  - Gaussian mutation
  - Elitism preservation

- **RLTrainer**: PPO-based reinforcement learning
  - Compatible with gym environments
  - Checkpoint support
  - Experience replay ready

- **HyperparameterOptimizer**: Bayesian optimization
  - Gaussian Process surrogate
  - Acquisition functions (EI, UCB, POI)
  - Fallback to random search

- **Strategy Templates**: Pre-built strategies
  - Mean reversion
  - Trend following
  - Momentum
  - Breakout
  - Pairs trading

**Dependencies**: numpy, pandas, stable-baselines3 (for RL), scikit-optimize (optional)

### 3. `backtester.py` - Advanced Backtesting Engine

Rigorous validation framework:

- **WalkForwardAnalyzer**: Combinatorial purged cross-validation
  - Configurable train/test/embargo splits
  - Out-of-sample validation
  - Decay ratio calculation

- **MetricsCalculator**: Comprehensive metrics
  - Sharpe Ratio
  - Probabilistic Sharpe Ratio (PSR)
  - Deflated Sharpe Ratio (DSR)
  - Maximum Drawdown
  - Calmar Ratio
  - Profit Factor
  - VaR/CVaR

- **AdvancedBacktester**: Multi-asset backtesting
  - Minimum trade requirements (≥100 per asset)
  - Minimum asset requirements (≥10 assets)
  - Transaction cost modeling
  - Slippage simulation

**Dependencies**: numpy, pandas, scipy

## Usage Example

```python
from autonomous.quant_algorithms import FractionalDifferentiation, CointegrationTracker
from autonomous.strategy_generator import GeneticProgrammingEngine, Strategy
from autonomous.backtester import AdvancedBacktester, MetricsCalculator

# 1. Feature Engineering
frac_diff = FractionalDifferentiation(d=0.5)
stationary_series = frac_diff.fit_transform(prices)

# 2. Find Cointegrated Pairs
tracker = CointegrationTracker()
pairs = tracker.find_pairs(prices_df, min_correlation=0.7)

# 3. Generate Strategies
gp_engine = GeneticProgrammingEngine(
    population_size=100,
    generations=50,
    mutation_rate=0.1
)
gp_engine.initialize_population(feature_names=['rsi', 'macd', 'volatility'])

best_strategy = gp_engine.evolve(features, returns, fitness_function)

# 4. Backtest with Rigorous Validation
backtester = AdvancedBacktester(
    initial_capital=100000,
    min_trades_per_asset=100,
    min_assets=10
)

result = backtester.backtest_strategy(
    strategy=best_strategy,
    prices=price_dataframe,
    features=feature_dataframe,
    signals_func=generate_signals
)

# 5. Evaluate Results
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Probabilistic Sharpe: {result.probabilistic_sharpe:.3f}")
print(f"Deflated Sharpe: {result.deflated_sharpe:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
print(f"Valid: {result.is_valid}")
```

## Validation Criteria

Strategies must pass:
- PSR > 0.95 (95% confidence Sharpe > 0)
- DSR > 0.5 (adjusted for multiple testing)
- Max Drawdown < 15%
- ≥100 trades per asset
- ≥10 assets tested
- Positive OOS performance in ≥70% of walk-forward splits

## Next Steps (Roadmap)

1. **Data Ingestion Pipeline**: Real-time and historical data loaders
2. **Feature Engineering Module**: Automated feature generation
3. **User Approval Gateway**: Dashboard for strategy review
4. **Paper Trading Module**: Live simulation without capital
5. **Execution Algorithms**: VWAP, Almgren-Chriss implementation
6. **Risk Management Layer**: Position limits, kill switches
7. **Continuous Learning**: Weekly re-training pipeline

See `AUTONOMOUS_SYSTEM_ROADMAP.md` for complete details.

## Warnings

⚠️ **Trading involves substantial risk of loss.** This software is for research purposes only. Always:
- Backtest thoroughly before live deployment
- Paper trade extensively
- Start with small position sizes
- Use proper risk management
- Monitor continuously
