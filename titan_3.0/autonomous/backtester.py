"""
Advanced Backtesting Engine for TITAN 3.0

Implements walk-forward analysis with embargo, rigorous validation,
and comprehensive metrics calculation including Probabilistic Sharpe Ratio
and Deflated Sharpe Ratio.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_id: str
    asset: str
    equity_curve: pd.Series
    returns: pd.Series
    sharpe_ratio: float
    probabilistic_sharpe: float
    deflated_sharpe: float
    max_drawdown: float
    calmar_ratio: float
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    turnover: float
    is_valid: bool
    validation_message: str


class WalkForwardAnalyzer:
    """Walk-forward analysis with embargo periods."""
    
    def __init__(
        self,
        n_splits: int = 10,
        train_pct: float = 0.6,
        test_pct: float = 0.2,
        embargo_pct: float = 0.2,
        min_samples_per_split: int = 100
    ):
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.embargo_pct = embargo_pct
        self.min_samples_per_split = min_samples_per_split
    
    def generate_splits(self, n_samples: int) -> List[Tuple[int, int, int, int, int, int]]:
        """Generate walk-forward split indices."""
        if n_samples < self.min_samples_per_split * self.n_splits:
            raise ValueError(f"Insufficient samples ({n_samples})")
        
        split_size = n_samples // self.n_splits
        splits = []
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            train_end = int(start_idx + split_size * self.train_pct)
            embargo_start = train_end
            embargo_end = int(embargo_start + split_size * self.embargo_pct)
            test_start = embargo_end
            test_end = min(int(test_start + split_size * self.test_pct), n_samples)
            
            if test_end - test_start >= self.min_samples_per_split // self.n_splits:
                splits.append((start_idx, train_end, embargo_start, embargo_end, test_start, test_end))
        
        return splits


class MetricsCalculator:
    """Comprehensive metrics calculator for backtest evaluation."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, annualization_factor: int = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(annualization_factor)
    
    @staticmethod
    def calculate_probabilistic_sharpe(
        returns: pd.Series,
        benchmark_sharpe: float = 0.0,
        annualization_factor: int = 252
    ) -> float:
        """Calculate Probabilistic Sharpe Ratio (PSR)."""
        n = len(returns)
        if n < 10:
            return 0.0
        
        sr = MetricsCalculator.calculate_sharpe_ratio(returns, annualization_factor)
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) * sr**2 / 4) / n
        if var_sr <= 0:
            return 1.0 if sr > benchmark_sharpe else 0.0
        
        z = (sr - benchmark_sharpe) / np.sqrt(var_sr)
        return stats.norm.cdf(z)
    
    @staticmethod
    def calculate_deflated_sharpe(
        returns: pd.Series,
        n_trials: int = 100,
        annualization_factor: int = 252
    ) -> float:
        """Calculate Deflated Sharpe Ratio (DSR)."""
        n = len(returns)
        if n < 10 or n_trials < 1:
            return 0.0
        
        sr = MetricsCalculator.calculate_sharpe_ratio(returns, annualization_factor)
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) * sr**2 / 4) / n
        if var_sr <= 0:
            return 1.0 if sr > 0 else 0.0
        
        std_sr = np.sqrt(var_sr)
        expected_max_sr = stats.norm.ppf(1 - 1/n_trials) * std_sr
        z = (sr - expected_max_sr) / std_sr
        
        return stats.norm.cdf(z)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return abs(drawdown.min())


class AdvancedBacktester:
    """Advanced backtesting engine with rigorous validation."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        annualization_factor: int = 252,
        min_trades_per_asset: int = 100,
        min_assets: int = 10
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.annualization_factor = annualization_factor
        self.min_trades_per_asset = min_trades_per_asset
        self.min_assets = min_assets
        self.metrics = MetricsCalculator()
    
    def backtest_strategy(
        self,
        strategy: Any,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        signals_func: callable
    ) -> BacktestResult:
        """Backtest strategy on multiple assets."""
        all_equity_curves = []
        all_trades = []
        
        for asset in prices.columns:
            try:
                signals = signals_func(strategy, features, prices[asset])
                equity_curve, trades = self._run_single_asset_backtest(
                    prices[asset], signals, asset
                )
                
                if len(trades) >= self.min_trades_per_asset:
                    all_equity_curves.append(equity_curve)
                    all_trades.append(trades)
            except Exception as e:
                logger.warning(f"Backtest failed for {asset}: {e}")
                continue
        
        if not all_equity_curves:
            return self._empty_result(strategy)
        
        # Aggregate results
        portfolio_equity = pd.concat(all_equity_curves, axis=1).sum(axis=1)
        portfolio_returns = portfolio_equity.pct_change().dropna()
        combined_trades = pd.concat(all_trades, ignore_index=True)
        
        # Calculate metrics
        sharpe = self.metrics.calculate_sharpe_ratio(portfolio_returns, self.annualization_factor)
        psr = self.metrics.calculate_probabilistic_sharpe(portfolio_returns, 0.0, self.annualization_factor)
        dsr = self.metrics.calculate_deflated_sharpe(portfolio_returns, 100, self.annualization_factor)
        max_dd = self.metrics.calculate_max_drawdown(portfolio_equity)
        
        winning_trades = len(combined_trades[combined_trades['pnl'] > 0])
        win_rate = winning_trades / len(combined_trades) if len(combined_trades) > 0 else 0.0
        profit_factor = self._calculate_profit_factor(combined_trades)
        
        # Validation
        is_valid = (
            len(all_equity_curves) >= self.min_assets and
            psr > 0.95 and
            dsr > 0.5 and
            max_dd < 0.15
        )
        
        return BacktestResult(
            strategy_id=strategy.id if hasattr(strategy, 'id') else 'unknown',
            asset='portfolio',
            equity_curve=portfolio_equity,
            returns=portfolio_returns,
            sharpe_ratio=sharpe,
            probabilistic_sharpe=psr,
            deflated_sharpe=dsr,
            max_drawdown=max_dd,
            calmar_ratio=sharpe / max_dd if max_dd > 0 else 0.0,
            n_trades=len(combined_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=combined_trades[combined_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0.0,
            avg_loss=abs(combined_trades[combined_trades['pnl'] < 0]['pnl'].mean()) if len(combined_trades) > winning_trades else 0.0,
            skewness=portfolio_returns.skew(),
            kurtosis=portfolio_returns.kurtosis(),
            var_95=portfolio_returns.quantile(0.05),
            cvar_95=portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean(),
            turnover=combined_trades['size'].sum() / self.initial_capital if 'size' in combined_trades.columns else 0.0,
            is_valid=is_valid,
            validation_message='Passed' if is_valid else 'Failed validation'
        )
    
    def _run_single_asset_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        asset: str
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Run backtest for single asset."""
        equity = [self.initial_capital]
        position = 0
        cash = self.initial_capital
        trades = []
        
        for i in range(len(prices)):
            signal = signals.iloc[i] if i < len(signals) else 0
            price = prices.iloc[i]
            
            if signal > 0.5 and position == 0:
                shares = int(cash * 0.95 / price)
                if shares > 0:
                    cost = shares * price * (1 + self.transaction_cost + self.slippage)
                    cash -= cost
                    position = shares
                    trades.append({'asset': asset, 'timestamp': prices.index[i], 'type': 'buy', 'price': price, 'size': shares, 'pnl': 0.0})
            
            elif signal < 0.5 and position > 0:
                proceeds = position * price * (1 - self.transaction_cost - self.slippage)
                pnl = proceeds - (position * prices.iloc[max(0, i-1)] if i > 0 else position * price)
                cash += proceeds
                if trades:
                    trades[-1]['pnl'] = pnl
                trades.append({'asset': asset, 'timestamp': prices.index[i], 'type': 'sell', 'price': price, 'size': position, 'pnl': pnl})
                position = 0
            
            equity.append(cash + (position * price if position > 0 else 0))
        
        return pd.Series(equity[1:], index=prices.index), pd.DataFrame(trades) if trades else pd.DataFrame(columns=['asset', 'timestamp', 'type', 'price', 'size', 'pnl'])
    
    def _empty_result(self, strategy: Any) -> BacktestResult:
        """Return empty result."""
        return BacktestResult(
            strategy_id=strategy.id if hasattr(strategy, 'id') else 'unknown',
            asset='portfolio',
            equity_curve=pd.Series(),
            returns=pd.Series(),
            sharpe_ratio=0.0,
            probabilistic_sharpe=0.0,
            deflated_sharpe=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            n_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            skewness=0.0,
            kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
            turnover=0.0,
            is_valid=False,
            validation_message='No valid results'
        )
    
    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor."""
        if len(trades) == 0:
            return 1.0
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
        return gross_profit / gross_loss
