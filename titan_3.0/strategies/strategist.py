"""
TITAN 3.0 - Phase 4: Automated Strategy Lifecycle
Modules:
- strategist.py: Auto-generates trading strategies based on market regime
- backtester.py: Event-driven backtester with Walk-Forward Optimization (WFO)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    side: SignalType
    pnl: float
    return_pct: float

@dataclass
class StrategyConfig:
    name: str
    regime_filter: List[str]  # e.g., ["BULL_LOW_VOL", "BEAR_HIGH_VOL"]
    entry_condition: str      # Python expression string
    exit_condition: str       # Python expression string
    stop_loss_pct: float
    take_profit_pct: float
    position_size_pct: float

class Strategist:
    """
    Auto-generates trading strategies based on detected market regimes.
    Uses a template-based approach to combine technical indicators with regime filters.
    """
    
    def __init__(self):
        self.strategy_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict]:
        return {
            "momentum_breakout": {
                "regime_filter": ["BULL_LOW_VOL", "BULL_HIGH_VOL"],
                "entry": "(df['close'] > df['close'].shift(20)) & (df['rsi'] < 70)",
                "exit": "(df['close'] < df['close'].shift(5)) | (df['rsi'] > 80)",
                "sl": 0.02,
                "tp": 0.06,
                "size": 0.1
            },
            "mean_reversion": {
                "regime_filter": ["SIDEWAYS_LOW_VOL"],
                "entry": "(df['close'] < df['bb_lower']) & (df['rsi'] < 30)",
                "exit": "(df['close'] > df['bb_middle']) | (df['rsi'] > 50)",
                "sl": 0.03,
                "tp": 0.04,
                "size": 0.05
            },
            "trend_following": {
                "regime_filter": ["BULL_LOW_VOL", "BEAR_LOW_VOL"],
                "entry": "(df['macd'] > df['macd_signal']) & (df['adx'] > 25)",
                "exit": "(df['macd'] < df['macd_signal']) | (df['adx'] < 20)",
                "sl": 0.025,
                "tp": 0.08,
                "size": 0.15
            },
            "volatility_contrarian": {
                "regime_filter": ["BEAR_HIGH_VOL"],
                "entry": "(df['close'] < df['bb_lower']) & (df['atr'] > df['atr'].rolling(50).mean() * 1.5)",
                "exit": "(df['close'] > df['bb_middle']) | (df['rsi'] > 40)",
                "sl": 0.04,
                "tp": 0.05,
                "size": 0.03
            }
        }

    def generate_strategies(self, current_regime: str, feature_columns: List[str]) -> List[StrategyConfig]:
        """
        Generates a list of viable strategies based on the current market regime.
        """
        viable_strategies = []
        
        for name, template in self.strategy_templates.items():
            if current_regime in template["regime_filter"]:
                config = StrategyConfig(
                    name=f"{name}_{current_regime}",
                    regime_filter=template["regime_filter"],
                    entry_condition=template["entry"],
                    exit_condition=template["exit"],
                    stop_loss_pct=template["sl"],
                    take_profit_pct=template["tp"],
                    position_size_pct=template["size"]
                )
                viable_strategies.append(config)
                logger.info(f"Generated strategy: {config.name} for regime {current_regime}")
        
        if not viable_strategies:
            # Fallback to neutral strategy
            logger.warning(f"No specific strategies for regime {current_regime}. Using default mean reversion.")
            fallback = self.strategy_templates["mean_reversion"]
            viable_strategies.append(StrategyConfig(
                name="fallback_mean_reversion",
                regime_filter=["SIDEWAYS_LOW_VOL"],
                entry_condition=fallback["entry"],
                exit_condition=fallback["exit"],
                stop_loss_pct=fallback["sl"],
                take_profit_pct=fallback["tp"],
                position_size_pct=fallback["size"]
            ))
            
        return viable_strategies

class WalkForwardOptimizer:
    """
    Implements Walk-Forward Optimization to prevent overfitting.
    Splits data into multiple in-sample (training) and out-of-sample (testing) periods.
    """
    def __init__(self, train_window: int = 252, test_window: int = 63, step: int = 21):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def generate_splits(self, data_length: int) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of tuples: (train_start, train_end, test_start, test_end)
        """
        splits = []
        start = 0
        while start + self.train_window + self.test_window <= data_length:
            train_start = start
            train_end = start + self.train_window
            test_start = train_end
            test_end = train_end + self.test_window
            
            splits.append((train_start, train_end, test_start, test_end))
            start += self.step
            
        return splits

class Backtester:
    """
    Event-driven backtester supporting Walk-Forward Optimization.
    """
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.wfo = WalkForwardOptimizer()

    def run(self, df: pd.DataFrame, strategy: StrategyConfig, regime_series: pd.Series) -> Dict[str, Any]:
        """
        Runs backtest on the entire dataset or specific WFO segments.
        Returns performance metrics.
        """
        # Add regime column to df for filtering
        df = df.copy()
        df['regime'] = regime_series
        
        all_trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        position = None  # {'entry_price': float, 'shares': int, 'side': SignalType}

        # Pre-calculate stop/take profit levels dynamically
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check if regime matches strategy filter
            if row['regime'] not in strategy.regime_filter:
                if position:
                    # Force exit if regime changes invalidating strategy
                    exit_price = row['close']
                    pnl = self._calculate_pnl(position, exit_price)
                    current_capital += pnl
                    all_trades.append(self._create_trade(position, row.name, exit_price, pnl))
                    position = None
            
            # Entry Logic
            if position is None:
                try:
                    # Evaluate entry condition string safely
                    if eval(strategy.entry_condition, {"df": df, "np": np, "pd": pd}, {"i": i}):
                        shares = int((current_capital * strategy.position_size_pct) / row['close'])
                        if shares > 0:
                            position = {
                                'entry_price': row['close'],
                                'shares': shares,
                                'side': SignalType.LONG, # Simplified for long-only demo
                                'entry_date': row.name,
                                'stop_loss': row['close'] * (1 - strategy.stop_loss_pct),
                                'take_profit': row['close'] * (1 + strategy.take_profit_pct)
                            }
                except Exception as e:
                    logger.error(f"Error evaluating entry condition: {e}")

            # Exit Logic (Stop Loss / Take Profit / Condition)
            elif position:
                exit_triggered = False
                exit_price = row['close']
                
                # Check Stop Loss
                if row['close'] <= position['stop_loss']:
                    exit_triggered = True
                # Check Take Profit
                elif row['close'] >= position['take_profit']:
                    exit_triggered = True
                # Check Strategy Exit Condition
                else:
                    try:
                        if eval(strategy.exit_condition, {"df": df, "np": np, "pd": pd}, {"i": i}):
                            exit_triggered = True
                    except Exception as e:
                        logger.error(f"Error evaluating exit condition: {e}")

                if exit_triggered:
                    pnl = self._calculate_pnl(position, exit_price)
                    current_capital += pnl
                    all_trades.append(self._create_trade(position, row.name, exit_price, pnl))
                    position = None
            
            equity_curve.append(current_capital)

        # Close any open position at end
        if position:
            exit_price = df.iloc[-1]['close']
            pnl = self._calculate_pnl(position, exit_price)
            current_capital += pnl
            all_trades.append(self._create_trade(position, df.iloc[-1].name, exit_price, pnl))

        return self._calculate_metrics(all_trades, equity_curve)

    def run_walk_forward(self, df: pd.DataFrame, strategy: StrategyConfig, regime_series: pd.Series) -> Dict[str, Any]:
        """
        Runs Walk-Forward Optimization across multiple time windows.
        """
        splits = self.wfo.generate_splits(len(df))
        oos_results = []
        
        for train_start, train_end, test_start, test_end in splits:
            # In this simple version, we use the same strategy params for OOS
            # In advanced WFO, we would optimize params on Train, then apply to Test
            test_df = df.iloc[test_start:test_end]
            test_regime = regime_series.iloc[test_start:test_end]
            
            if len(test_df) < 10:
                continue
                
            result = self.run(test_df, strategy, test_regime)
            result['period'] = f"{test_df.index[0]} to {test_df.index[-1]}"
            oos_results.append(result)
            
        return self._aggregate_wfo_results(oos_results)

    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        if position['side'] == SignalType.LONG:
            return (exit_price - position['entry_price']) * position['shares']
        else:
            return (position['entry_price'] - exit_price) * position['shares']

    def _create_trade(self, position: Dict, exit_date: pd.Timestamp, exit_price: float, pnl: float) -> Trade:
        return_pct = pnl / (position['entry_price'] * position['shares'])
        return Trade(
            entry_date=position['entry_date'],
            exit_date=exit_date,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            shares=position['shares'],
            side=position['side'],
            pnl=pnl,
            return_pct=return_pct
        )

    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict[str, Any]:
        if not trades:
            return {"total_return": 0, "sharpe": 0, "max_drawdown": 0, "trade_count": 0}
        
        series = pd.Series(equity_curve)
        returns = series.pct_change().dropna()
        
        total_return = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Max Drawdown
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        win_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(win_trades) / len(trades) if trades else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "trade_count": len(trades),
            "win_rate": win_rate,
            "final_equity": series.iloc[-1],
            "equity_curve": equity_curve
        }

    def _aggregate_wfo_results(self, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {}
        
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        worst_dd = min([r['max_drawdown'] for r in results])
        
        return {
            "wfo_avg_return": avg_return,
            "wfo_avg_sharpe": avg_sharpe,
            "wfo_worst_drawdown": worst_dd,
            "segments_tested": len(results),
            "robustness_score": (avg_sharpe * abs(avg_return)) / (abs(worst_dd) + 0.01)
        }
