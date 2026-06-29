"""
TITAN 3.0 - Phase 8: Execution Module
Modules:
- order_manager.py: Handles order routing, sizing, and simulation
- risk_manager.py: Pre-trade risk checks and portfolio limits
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class RiskManager:
    """
    Enforces pre-trade risk limits.
    """
    def __init__(self, max_position_size: float = 100000.0, 
                 max_daily_loss: float = 2000.0,
                 max_portfolio_risk: float = 0.02):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_risk = max_portfolio_risk
        
        self.daily_pnl = 0.0
        self.current_positions = {} # symbol -> quantity

    def check_order(self, order: Order, current_price: float, portfolio_value: float) -> tuple[bool, str]:
        """
        Returns (approved, reason).
        """
        # 1. Check Position Size Limit
        notional = order.quantity * current_price
        if notional > self.max_position_size:
            return False, f"Order size {notional} exceeds limit {self.max_position_size}"
            
        # 2. Check Daily Loss Limit
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl}"
            
        # 3. Check Portfolio Risk (simplified)
        if order.side == OrderSide.BUY:
            # Ensure buying doesn't exceed cash (simplified check)
            if notional > portfolio_value * 0.5: # Max 50% in one trade
                return False, "Order exceeds 50% of portfolio value"
                
        return True, "Approved"

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl

class OrderManager:
    """
    Manages order lifecycle: creation, validation, execution (simulated).
    """
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.order_book = []
        self.fills = []

    def create_order(self, symbol: str, side: str, quantity: int, 
                     order_type: str = "MARKET", price: float = None) -> Optional[Order]:
        try:
            o_side = OrderSide[side.upper()]
            o_type = OrderType[order_type.upper()]
        except KeyError:
            logger.error("Invalid order side or type")
            return None
            
        order = Order(symbol=symbol, side=o_side, quantity=quantity, order_type=o_type, price=price)
        return order

    def submit_order(self, order: Order, current_price: float, portfolio_value: float) -> Dict[str, Any]:
        # Risk Check
        approved, reason = self.risk_manager.check_order(order, current_price, portfolio_value)
        
        if not approved:
            logger.warning(f"Order rejected: {reason}")
            return {"status": "REJECTED", "reason": reason}
            
        # Execute (Simulated)
        fill_price = current_price
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price > order.price:
                return {"status": "PENDING", "reason": "Limit price not met"}
            elif order.side == OrderSide.SELL and current_price < order.price:
                return {"status": "PENDING", "reason": "Limit price not met"}
            fill_price = order.price # Assume immediate fill at limit or better
            
        # Record Fill
        fill = {
            "order": order,
            "fill_price": fill_price,
            "fill_time": datetime.now(),
            "notional": fill_price * order.quantity
        }
        self.fills.append(fill)
        self.order_book.append(order)
        
        # Update Risk Manager Position
        qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        self.risk_manager.current_positions[order.symbol] = \
            self.risk_manager.current_positions.get(order.symbol, 0) + qty
            
        logger.info(f"Order filled: {order.side.value} {order.quantity} {order.symbol} @ {fill_price}")
        
        return {"status": "FILLED", "fill_price": fill_price, "fill": fill}

    def get_execution_report(self) -> Dict[str, Any]:
        total_volume = sum(f['notional'] for f in self.fills)
        return {
            "total_orders": len(self.order_book),
            "total_fills": len(self.fills),
            "total_volume": total_volume,
            "daily_pnl": self.risk_manager.daily_pnl
        }
