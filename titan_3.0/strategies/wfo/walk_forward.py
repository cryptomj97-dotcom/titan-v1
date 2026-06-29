"""
TITAN 3.0 - Walk-Forward Optimization Engine
Implements robust walk-forward analysis for strategy validation and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from copy import deepcopy

from ..core.strategy_base import BaseStrategy, StrategyPerformance, TradeSignal


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward iteration."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    train_max_drawdown: float
    test_max_drawdown: float
    train_trades: int
    test_trades: int
    parameters: Dict[str, Any]
    is_profitable: bool = True


@dataclass
class WalkForwardSummary:
    """Aggregate results from all walk-forward iterations."""
    total_iterations: int
    profitable_iterations: int
    avg_train_sharpe: float
    avg_test_sharpe: float
    std_train_sharpe: float
    std_test_sharpe: float
    avg_train_return: float
    avg_test_return: float
    avg_train_drawdown: float
    avg_test_drawdown: float
    efficiency_ratio: float  # test_sharpe / train_sharpe
    stability_score: float  # 0-1 score based on consistency
    final_parameters: Dict[str, Any]
    iteration_results: List[WalkForwardResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'total_iterations': self.total_iterations,
            'profitable_iterations': self.profitable_iterations,
            'profitability_rate': self.profitable_iterations / max(1, self.total_iterations),
            'avg_train_sharpe': self.avg_train_sharpe,
            'avg_test_sharpe': self.avg_test_sharpe,
            'std_train_sharpe': self.std_train_sharpe,
            'std_test_sharpe': self.std_test_sharpe,
            'avg_train_return': self.avg_train_return,
            'avg_test_return': self.avg_test_return,
            'avg_train_drawdown': self.avg_train_drawdown,
            'avg_test_drawdown': self.avg_test_drawdown,
            'efficiency_ratio': self.efficiency_ratio,
            'stability_score': self.stability_score,
            'final_parameters': self.final_parameters
        }


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization engine for strategy validation.
    
    Implements rolling window optimization with out-of-sample testing
    to prevent overfitting and validate strategy robustness.
    """
    
    def __init__(self,
                 strategy: BaseStrategy,
                 train_window_days: int = 90,
                 test_window_days: int = 30,
                 step_days: int = 15,
                 min_train_samples: int = 60,
                 min_test_samples: int = 20):
        """
        Initialize Walk-Forward Optimizer.
        
        Args:
            strategy: Base strategy to optimize
            train_window_days: Length of training window in days
            test_window_days: Length of test (out-of-sample) window
            step_days: Step size for rolling window
            min_train_samples: Minimum samples required for training
            min_test_samples: Minimum samples required for testing
        """
        self.strategy = strategy
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        
        self.results: List[WalkForwardResult] = []
        self.best_parameters: Dict[str, Any] = {}
        self.is_optimized = False
    
    def generate_windows(self, 
                        data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test window pairs for walk-forward analysis.
        
        Returns:
            List of (train_data, test_data) tuples
        """
        windows = []
        
        if len(data) < self.min_train_samples + self.min_test_samples:
            raise ValueError("Insufficient data for walk-forward analysis")
        
        # Convert days to approximate number of bars (assuming daily data)
        train_bars = self.train_window_days
        test_bars = self.test_window_days
        step_bars = self.step_days
        
        start_idx = 0
        while start_idx + train_bars + test_bars <= len(data):
            train_end_idx = start_idx + train_bars
            test_end_idx = train_end_idx + test_bars
            
            train_data = data.iloc[start_idx:train_end_idx].copy()
            test_data = data.iloc[train_end_idx:test_end_idx].copy()
            
            if len(train_data) >= self.min_train_samples and len(test_data) >= self.min_test_samples:
                windows.append((train_data, test_data))
            
            start_idx += step_bars
        
        if not windows:
            raise ValueError("Could not generate any valid walk-forward windows")
        
        return windows
    
    def backtest_strategy(self,
                         data: pd.DataFrame,
                         features: Dict[str, pd.Series],
                         regimes: pd.Series,
                         initial_capital: float = 100000.0) -> Tuple[float, float, float, int]:
        """
        Run simple backtest and return metrics.
        
        Returns:
            (sharpe_ratio, total_return, max_drawdown, num_trades)
        """
        capital = initial_capital
        position = None
        returns = []
        trades = 0
        equity_curve = [initial_capital]
        
        for i in range(len(data) - 1):
            data_slice = data.iloc[:i+1].copy()
            features_slice = {k: v.iloc[:i+1] if hasattr(v, 'iloc') else v for k, v in features.items()}
            regime = regimes.iloc[i] if hasattr(regimes, 'iloc') else regimes[i] if i < len(regimes) else 'neutral'
            
            signal = self.strategy.generate_signal(data_slice, features_slice, regime, position)
            
            if signal is None:
                equity_curve.append(capital)
                continue
            
            if signal.signal_type.value.startswith('LONG') or signal.signal_type.value.startswith('SHORT'):
                if position is None and signal.signal_type in [TradeSignal, SignalType]:
                    from ..core.strategy_base import SignalType
                    if signal.signal_type == SignalType.LONG:
                        size = self.strategy.calculate_position_size(signal, capital, 0.02, 0.02)
                        if size > 0:
                            position = {'entry_price': signal.price, 'size': size, 'side': 'long'}
                    elif signal.signal_type == SignalType.SHORT:
                        size = self.strategy.calculate_position_size(signal, capital, 0.02, 0.02)
                        if size > 0:
                            position = {'entry_price': signal.price, 'size': size, 'side': 'short'}
            
            elif signal.signal_type.value.startswith('CLOSE'):
                if position is not None:
                    current_price = data['close'].iloc[i]
                    if position['side'] == 'long':
                        pnl = (current_price - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['size']
                    
                    capital += pnl
                    returns.append(pnl / initial_capital)
                    trades += 1
                    position = None
            
            equity_curve.append(capital)
        
        # Calculate metrics
        if len(returns) < 2:
            return 0.0, 0.0, 0.0, trades
        
        returns_array = np.array(returns)
        sharpe = (np.mean(returns_array) / (np.std(returns_array) + 1e-8)) * np.sqrt(252)
        total_return = (capital - initial_capital) / initial_capital
        
        # Calculate max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        return sharpe, total_return, max_drawdown, trades
    
    def optimize_parameters(self,
                           train_data: pd.DataFrame,
                           features: Dict[str, pd.Series],
                           regimes: pd.Series,
                           param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize strategy parameters on training data using grid search.
        
        Args:
            train_data: Training OHLCV data
            features: Pre-computed features
            regimes: Market regimes
            param_grid: Dictionary of parameter names to list of values to try
            
        Returns:
            Best parameter set
        """
        best_sharpe = -np.inf
        best_params = self.strategy.parameters.copy()
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combo in product(*param_values):
            test_params = dict(zip(param_names, combo))
            
            # Temporarily update strategy parameters
            original_params = self.strategy.parameters.copy()
            self.strategy.parameters.update(test_params)
            
            try:
                sharpe, _, _, _ = self.backtest_strategy(train_data, features, regimes)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = self.strategy.parameters.copy()
            except Exception:
                pass
            finally:
                # Restore original parameters
                self.strategy.parameters = original_params
        
        return best_params
    
    def run_walk_forward(self,
                        data: pd.DataFrame,
                        features: Dict[str, pd.Series],
                        regimes: pd.Series,
                        param_grid: Optional[Dict[str, List[Any]]] = None,
                        initial_capital: float = 100000.0) -> WalkForwardSummary:
        """
        Run complete walk-forward optimization.
        
        Args:
            data: Full OHLCV dataset
            features: Pre-computed features for entire dataset
            regimes: Market regimes for entire dataset
            param_grid: Optional parameter grid for optimization
            initial_capital: Starting capital for backtests
            
        Returns:
            WalkForwardSummary with aggregate statistics
        """
        print(f"Starting Walk-Forward Optimization...")
        print(f"Train window: {self.train_window_days} days, Test window: {self.test_window_days} days")
        
        windows = self.generate_windows(data)
        print(f"Generated {len(windows)} walk-forward windows")
        
        self.results = []
        train_sharpes = []
        test_sharpes = []
        train_returns = []
        test_returns = []
        train_drawdowns = []
        test_drawdowns = []
        profitable_count = 0
        
        for i, (train_data, test_data) in enumerate(windows):
            print(f"\nIteration {i+1}/{len(windows)}")
            print(f"  Train: {train_data.index[0].date()} to {train_data.index[-1].date()}")
            print(f"  Test:  {test_data.index[0].date()} to {test_data.index[-1].date()}")
            
            # Align features and regimes with data
            train_features = {k: v.loc[train_data.index] if hasattr(v, 'loc') else v for k, v in features.items()}
            test_features = {k: v.loc[test_data.index] if hasattr(v, 'loc') else v for k, v in features.items()}
            
            train_regimes = regimes.loc[train_data.index] if hasattr(regimes, 'loc') else regimes[:len(train_data)]
            test_regimes = regimes.loc[test_data.index] if hasattr(regimes, 'loc') else regimes[len(train_data):len(train_data)+len(test_data)]
            
            # Optimize parameters on training data
            if param_grid:
                best_params = self.optimize_parameters(train_data, train_features, train_regimes, param_grid)
                self.strategy.parameters = best_params
            else:
                best_params = self.strategy.parameters.copy()
            
            # Backtest on training data
            train_sharpe, train_return, train_dd, train_trades = self.backtest_strategy(
                train_data, train_features, train_regimes, initial_capital
            )
            
            # Backtest on test data (out-of-sample)
            test_sharpe, test_return, test_dd, test_trades = self.backtest_strategy(
                test_data, test_features, test_regimes, initial_capital
            )
            
            result = WalkForwardResult(
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                train_return=train_return,
                test_return=test_return,
                train_max_drawdown=train_dd,
                test_max_drawdown=test_dd,
                train_trades=train_trades,
                test_trades=test_trades,
                parameters=best_params,
                is_profitable=(test_return > 0)
            )
            
            self.results.append(result)
            
            train_sharpes.append(train_sharpe)
            test_sharpes.append(test_sharpe)
            train_returns.append(train_return)
            test_returns.append(test_return)
            train_drawdowns.append(train_dd)
            test_drawdowns.append(test_dd)
            
            if test_return > 0:
                profitable_count += 1
            
            print(f"  Train Sharpe: {train_sharpe:.3f}, Test Sharpe: {test_sharpe:.3f}")
            print(f"  Train Return: {train_return:.2%}, Test Return: {test_return:.2%}")
        
        # Calculate summary statistics
        avg_train_sharpe = np.mean(train_sharpes)
        avg_test_sharpe = np.mean(test_sharpes)
        std_train_sharpe = np.std(train_sharpes)
        std_test_sharpe = np.std(test_sharpes)
        
        # Efficiency ratio: how well does out-of-sample match in-sample?
        efficiency_ratio = avg_test_sharpe / (avg_train_sharpe + 1e-8)
        
        # Stability score: based on consistency of results
        sharpe_consistency = 1 - (std_test_sharpe / (abs(avg_test_sharpe) + 1e-8))
        profitability_rate = profitable_count / len(self.results)
        stability_score = (sharpe_consistency * 0.5 + profitability_rate * 0.5)
        stability_score = max(0, min(1, stability_score))
        
        # Use parameters from last iteration as final
        final_params = self.results[-1].parameters if self.results else {}
        
        summary = WalkForwardSummary(
            total_iterations=len(self.results),
            profitable_iterations=profitable_count,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            std_train_sharpe=std_train_sharpe,
            std_test_sharpe=std_test_sharpe,
            avg_train_return=np.mean(train_returns),
            avg_test_return=np.mean(test_returns),
            avg_train_drawdown=np.mean(train_drawdowns),
            avg_test_drawdown=np.mean(test_drawdowns),
            efficiency_ratio=efficiency_ratio,
            stability_score=stability_score,
            final_parameters=final_params,
            iteration_results=self.results
        )
        
        self.best_parameters = final_params
        self.is_optimized = True
        
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Iterations: {summary.total_iterations}")
        print(f"Profitable: {summary.profitable_iterations}/{summary.total_iterations} ({profitability_rate:.1%})")
        print(f"Avg Train Sharpe: {avg_train_sharpe:.3f} (+/- {std_train_sharpe:.3f})")
        print(f"Avg Test Sharpe:  {avg_test_sharpe:.3f} (+/- {std_test_sharpe:.3f})")
        print(f"Efficiency Ratio: {efficiency_ratio:.3f}")
        print(f"Stability Score:  {stability_score:.3f}")
        print(f"{'='*60}")
        
        return summary
    
    def get_validated_strategy(self) -> BaseStrategy:
        """
        Get the strategy with validated parameters from walk-forward optimization.
        
        Returns:
            Strategy with optimized parameters
        """
        if not self.is_optimized:
            raise ValueError("Walk-forward optimization has not been performed yet.")
        
        self.strategy.parameters = self.best_parameters
        return self.strategy
