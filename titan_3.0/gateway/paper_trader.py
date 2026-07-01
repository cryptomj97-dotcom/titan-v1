"""Paper trading module for live simulation."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Paper trade record."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    strategy_id: str
    pnl: float = 0.0
    closed: bool = False
    close_price: Optional[float] = None
    close_timestamp: Optional[datetime] = None


class PaperTrader:
    """
    Paper trading engine for live simulation.
    
    Executes trades with real-time data but without real money,
    tracking performance against backtest expectations.
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        Initialize paper trader.
        
        Args:
            initial_capital: Starting capital for paper trading
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.open_trades: List[PaperTrade] = []
        self.closed_trades: List[PaperTrade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = [datetime.now()]
        
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        logger.info(f"Paper trader initialized with ${initial_capital:,.2f}")
        
    def update_price(self, symbol: str, price: float) -> Dict:
        """
        Update price for a symbol and recalculate P&L.
        
        Args:
            symbol: Symbol to update
            price: Current price
            
        Returns:
            Updated P&L information
        """
        if symbol in self.positions and self.positions[symbol] != 0:
            # Update unrealized P&L
            qty = self.positions[symbol]
            entry_price = self.entry_prices[symbol]
            
            if qty > 0:  # Long
                self.unrealized_pnl += (price - entry_price) * qty
            else:  # Short
                self.unrealized_pnl += (entry_price - price) * abs(qty)
            
            # Update entry price for remaining position
            self.entry_prices[symbol] = price
        
        # Update equity curve
        current_equity = self.cash + self.unrealized_pnl + self.realized_pnl
        self.equity_curve.append(current_equity)
        self.timestamps.append(datetime.now())
        
        return {
            'symbol': symbol,
            'price': price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_equity': current_equity
        }
    
    def execute_trade(self, symbol: str, side: str, quantity: float,
                     price: float, strategy_id: str) -> Dict:
        """
        Execute a paper trade.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Quantity to trade
            price: Execution price
            strategy_id: Strategy generating the trade
            
        Returns:
            Trade execution result
        """
        timestamp = datetime.now()
        
        if side.lower() == 'buy':
            # Check if we have enough cash
            cost = quantity * price
            if cost > self.cash:
                return {
                    'success': False,
                    'error': 'Insufficient cash',
                    'required': cost,
                    'available': self.cash
                }
            
            # Update cash
            self.cash -= cost
            
            # Update position
            current_qty = self.positions.get(symbol, 0.0)
            new_qty = current_qty + quantity
            
            if current_qty < 0:  # Covering short
                # Calculate P&L on cover
                entry_price = self.entry_prices.get(symbol, price)
                pnl = (entry_price - price) * min(quantity, abs(current_qty))
                self.realized_pnl += pnl
                
            self.positions[symbol] = new_qty
            self.entry_prices[symbol] = price
            
            trade = PaperTrade(
                symbol=symbol,
                side='buy',
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                strategy_id=strategy_id
            )
            
            self.open_trades.append(trade)
            
            logger.info(f"Paper BUY: {quantity} {symbol} @ {price:.4f}")
            
        elif side.lower() == 'sell':
            # Check if we have the position
            current_qty = self.positions.get(symbol, 0.0)
            if current_qty <= 0:
                return {
                    'success': False,
                    'error': 'No position to sell',
                    'current_qty': current_qty
                }
            
            # Update cash
            proceeds = quantity * price
            self.cash += proceeds
            
            # Calculate P&L
            entry_price = self.entry_prices.get(symbol, price)
            pnl = (price - entry_price) * quantity
            self.realized_pnl += pnl
            
            # Update position
            new_qty = current_qty - quantity
            self.positions[symbol] = new_qty
            
            if new_qty == 0:
                del self.positions[symbol]
                del self.entry_prices[symbol]
            
            trade = PaperTrade(
                symbol=symbol,
                side='sell',
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                strategy_id=strategy_id,
                pnl=pnl,
                closed=True,
                close_price=price,
                close_timestamp=timestamp
            )
            
            self.open_trades.append(trade)
            self.closed_trades.append(trade)
            
            logger.info(f"Paper SELL: {quantity} {symbol} @ {price:.4f}, P&L: ${pnl:,.2f}")
        
        else:
            return {'success': False, 'error': 'Invalid side'}
        
        # Update equity curve
        total_equity = self.cash + self.unrealized_pnl + self.realized_pnl
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
        
        return {
            'success': True,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'cash_remaining': self.cash,
            'position': self.positions.get(symbol, 0.0),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_equity': total_equity
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate paper trading performance metrics."""
        if len(self.equity_curve) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate returns
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Remove zeros and infinities
        returns = returns[np.isfinite(returns) & (returns != 0)]
        
        if len(returns) == 0:
            return {'status': 'no_returns'}
        
        # Metrics
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        daily_returns = returns  # Assuming intraday, approximate as daily
        
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        if self.closed_trades:
            winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
            win_rate = winning_trades / len(self.closed_trades)
        else:
            win_rate = 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'current_equity': self.equity_curve[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'num_trades': len(self.closed_trades),
            'num_open_positions': len(self.positions),
            'trading_days': len(self.equity_curve)
        }
    
    def get_current_positions(self) -> List[Dict]:
        """Get current open positions."""
        positions = []
        for symbol, qty in self.positions.items():
            if qty != 0:
                current_price = self.entry_prices.get(symbol, 0)
                market_value = qty * current_price
                positions.append({
                    'symbol': symbol,
                    'quantity': qty,
                    'entry_price': current_price,
                    'market_value': market_value,
                    'weight': market_value / self.equity_curve[-1] if self.equity_curve[-1] > 0 else 0
                })
        return positions
    
    def reset(self):
        """Reset paper trader to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.entry_prices = {}
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = [self.initial_capital]
        self.timestamps = [datetime.now()]
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        logger.info("Paper trader reset")
