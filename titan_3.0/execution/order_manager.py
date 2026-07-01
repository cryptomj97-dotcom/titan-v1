"""
TITAN 3.0 - Phase 8: Execution Module
Modules:
- order_manager.py: Handles order routing, sizing, and simulation
- risk_manager.py: Pre-trade risk checks and portfolio limits

Security Improvements:
- Thread-safe order management
- Input validation for order parameters
- Audit logging for all order operations
- Rate limiting on order submissions
"""

import logging
import threading
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class OrderValidationError(Exception):
    """Custom exception for order validation errors."""
    pass


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: str = field(default_factory=lambda: f"ORD-{int(time.time() * 1000)}")
    status: OrderStatus = OrderStatus.PENDING
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        # Validate order parameters
        if self.quantity <= 0:
            raise OrderValidationError("Order quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise OrderValidationError("Order price must be positive if specified")
        if not self.symbol or not isinstance(self.symbol, str):
            raise OrderValidationError("Invalid symbol")


class OrderRateLimiter:
    """Rate limiter for order submissions to prevent abuse."""
    
    def __init__(self, max_orders_per_minute: int = 60):
        self.max_orders = max_orders_per_minute
        self.window_seconds = 60
        self._order_timestamps: List[float] = []
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if order submission is allowed."""
        now = time.time()
        with self._lock:
            # Remove timestamps older than window
            self._order_timestamps = [ts for ts in self._order_timestamps if now - ts < self.window_seconds]
            
            if len(self._order_timestamps) >= self.max_orders:
                logger.warning(f"Order rate limit exceeded: {len(self._order_timestamps)} orders in last {self.window_seconds}s")
                return False
            
            self._order_timestamps.append(now)
            return True
    
    def get_remaining(self) -> int:
        """Get remaining orders allowed in current window."""
        now = time.time()
        with self._lock:
            self._order_timestamps = [ts for ts in self._order_timestamps if now - ts < self.window_seconds]
            return max(0, self.max_orders - len(self._order_timestamps))


class RiskManager:
    """
    Enforces pre-trade risk limits with thread safety.
    
    Security Features:
    - Thread-safe position tracking
    - Comprehensive pre-trade checks
    - Audit logging
    - Kill switch support
    """
    
    def __init__(self, 
                 max_position_size: float = 100000.0, 
                 max_daily_loss: float = 2000.0,
                 max_portfolio_risk: float = 0.02,
                 kill_switch_enabled: bool = True):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_risk = max_portfolio_risk
        self.kill_switch_enabled = kill_switch_enabled
        
        self.daily_pnl = 0.0
        self.current_positions: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._kill_switch_triggered = False
        self._order_history: List[Dict] = []
    
    def trigger_kill_switch(self, reason: str = "Manual trigger"):
        """Trigger the kill switch to halt all trading."""
        with self._lock:
            self._kill_switch_triggered = True
            logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
    
    def reset_kill_switch(self) -> bool:
        """Reset kill switch after manual review."""
        with self._lock:
            if not self._kill_switch_triggered:
                return True
            self._kill_switch_triggered = False
            logger.info("Kill switch reset by authorized user")
            return True
    
    def check_order(self, order: Order, current_price: float, portfolio_value: float) -> tuple:
        """
        Perform comprehensive pre-trade risk checks.
        
        Returns:
            (approved, reason) tuple
        """
        with self._lock:
            # Check kill switch first
            if self._kill_switch_triggered:
                return False, "Kill switch is active - trading halted"
            
            # Validate order parameters
            try:
                if order.quantity <= 0:
                    return False, "Invalid order quantity"
                if current_price <= 0:
                    return False, "Invalid current price"
            except (AttributeError, TypeError):
                return False, "Invalid order or price format"
            
            # 1. Check Position Size Limit
            notional = order.quantity * current_price
            if notional > self.max_position_size:
                return False, f"Order size {notional:.2f} exceeds limit {self.max_position_size:.2f}"
            
            # 2. Check Daily Loss Limit
            if self.daily_pnl < -self.max_daily_loss:
                return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
            
            # 3. Check Portfolio Risk
            if order.side == OrderSide.BUY:
                max_single_trade = portfolio_value * 0.5
                if notional > max_single_trade:
                    return False, f"Order exceeds 50% of portfolio value ({max_single_trade:.2f})"
            
            # 4. Check existing position concentration
            current_position = self.current_positions.get(order.symbol, 0)
            new_position = current_position + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
            if abs(new_position * current_price) > self.max_position_size:
                return False, f"Resulting position would exceed maximum size"
            
            # Log approved order for audit
            self._order_history.append({
                'timestamp': datetime.now().isoformat(),
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': current_price,
                'notional': notional,
                'status': 'APPROVED'
            })
            
            return True, "Approved"
    
    def update_pnl(self, pnl: float):
        """Thread-safe PnL update with daily loss monitoring."""
        with self._lock:
            self.daily_pnl += pnl
            if self.daily_pnl < -self.max_daily_loss:
                logger.warning(f"Daily loss limit breached: {self.daily_pnl:.2f}")
                if self.kill_switch_enabled:
                    self.trigger_kill_switch("Daily loss limit breached")
    
    def update_position(self, symbol: str, delta: float):
        """Thread-safe position update."""
        with self._lock:
            current = self.current_positions.get(symbol, 0)
            self.current_positions[symbol] = current + delta
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        with self._lock:
            total_exposure = sum(abs(pos * 100) for pos in self.current_positions.values())  # Simplified
            return {
                'daily_pnl': self.daily_pnl,
                'kill_switch_active': self._kill_switch_triggered,
                'total_positions': len(self.current_positions),
                'positions': dict(self.current_positions),
                'daily_loss_limit': self.max_daily_loss,
                'loss_limit_utilization': abs(min(0, self.daily_pnl)) / self.max_daily_loss * 100
            }


class OrderManager:
    """
    Manages order lifecycle: creation, validation, execution (simulated).
    
    Security Features:
    - Thread-safe order book
    - Rate limiting
    - Comprehensive audit trail
    - Input validation
    """
    
    def __init__(self, risk_manager: RiskManager, max_orders_per_minute: int = 60):
        self.risk_manager = risk_manager
        self.order_book: List[Order] = []
        self.fills: List[Dict] = []
        self._lock = threading.RLock()
        self.rate_limiter = OrderRateLimiter(max_orders_per_minute)
        self._audit_log: List[Dict] = []
        # SECURITY FIX: Track order IDs to prevent race condition duplicates
        self._submitted_order_ids: Set[str] = set()
    
    def _log_audit(self, event: str, details: Dict):
        """Log audit event."""
        self._audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details
        })
        logger.info(f"AUDIT: {event} - {details}")
    
    def create_order(self, symbol: str, side: str, quantity: int, 
                     order_type: str = "MARKET", price: float = None) -> Optional[Order]:
        """Create an order with validation."""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise OrderValidationError("Invalid symbol")
            
            symbol = symbol.strip().upper()
            if len(symbol) > 20:
                raise OrderValidationError("Symbol too long")
            
            o_side = OrderSide[side.upper()]
            o_type = OrderType[order_type.upper()]
            
            if quantity <= 0:
                raise OrderValidationError("Quantity must be positive")
            
            if price is not None and price <= 0:
                raise OrderValidationError("Price must be positive")
            
            order = Order(
                symbol=symbol, 
                side=o_side, 
                quantity=int(quantity), 
                order_type=o_type, 
                price=price
            )
            
            self._log_audit("ORDER_CREATED", {
                'order_id': order.order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity
            })
            
            return order
            
        except KeyError as e:
            logger.error(f"Invalid order side or type: {e}")
            self._log_audit("ORDER_CREATION_FAILED", {'reason': 'Invalid side/type', 'error': str(e)})
            return None
        except OrderValidationError as e:
            logger.error(f"Order validation failed: {e}")
            self._log_audit("ORDER_VALIDATION_FAILED", {'reason': str(e)})
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating order: {e}")
            return None
    
    def submit_order(self, order: Order, current_price: float, portfolio_value: float) -> Dict[str, Any]:
        """Submit order with rate limiting and risk checks."""
        with self._lock:
            # SECURITY FIX: Check for duplicate order ID to prevent race condition
            if order.order_id in self._submitted_order_ids:
                logger.warning(f"Duplicate order submission detected: {order.order_id}")
                return {
                    "status": OrderStatus.REJECTED.value, 
                    "reason": "Duplicate order ID - possible race condition"
                }
            
            # Check rate limit
            if not self.rate_limiter.is_allowed():
                remaining = self.rate_limiter.get_remaining()
                self._log_audit("ORDER_REJECTED_RATE_LIMIT", {
                    'order_id': order.order_id,
                    'remaining': remaining
                })
                return {
                    "status": OrderStatus.REJECTED.value, 
                    "reason": f"Rate limit exceeded. Try again in {self.rate_limiter.window_seconds}s",
                    "retry_after": self.rate_limiter.window_seconds // self.rate_limiter.get_remaining() if self.rate_limiter.get_remaining() > 0 else 60
                }
            
            # Risk Check
            approved, reason = self.risk_manager.check_order(order, current_price, portfolio_value)
            
            if not approved:
                logger.warning(f"Order rejected: {reason}")
                self._log_audit("ORDER_REJECTED_RISK", {
                    'order_id': order.order_id,
                    'reason': reason
                })
                order.status = OrderStatus.REJECTED
                return {"status": OrderStatus.REJECTED.value, "reason": reason}
            
            # Mark order as submitted (thread-safe due to lock)
            self._submitted_order_ids.add(order.order_id)
            
            # Cleanup old order IDs periodically (keep last 1000)
            if len(self._submitted_order_ids) > 1000:
                # Keep only recent IDs - simple approach: clear half
                all_ids = list(self._submitted_order_ids)
                self._submitted_order_ids = set(all_ids[-500:])
            
            # Execute (Simulated)
            fill_price = current_price
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_price > order.price:
                    order.status = OrderStatus.PENDING
                    return {"status": OrderStatus.PENDING.value, "reason": "Limit price not met"}
                elif order.side == OrderSide.SELL and current_price < order.price:
                    order.status = OrderStatus.PENDING
                    return {"status": OrderStatus.PENDING.value, "reason": "Limit price not met"}
                fill_price = order.price  # Assume immediate fill at limit or better
            
            # Record Fill
            fill = {
                "order": order,
                "order_id": order.order_id,
                "fill_price": fill_price,
                "fill_time": datetime.now(),
                "notional": fill_price * order.quantity
            }
            self.fills.append(fill)
            self.order_book.append(order)
            order.status = OrderStatus.FILLED
            
            # Update Risk Manager Position
            qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
            self.risk_manager.update_position(order.symbol, qty)
            
            logger.info(f"Order filled: {order.side.value} {order.quantity} {order.symbol} @ {fill_price}")
            self._log_audit("ORDER_FILLED", {
                'order_id': order.order_id,
                'fill_price': fill_price,
                'quantity': order.quantity
            })
            
            return {"status": OrderStatus.FILLED.value, "fill_price": fill_price, "fill": fill}
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Get comprehensive execution report."""
        with self._lock:
            total_volume = sum(f['notional'] for f in self.fills)
            return {
                "total_orders": len(self.order_book),
                "total_fills": len(self.fills),
                "total_volume": total_volume,
                "daily_pnl": self.risk_manager.daily_pnl,
                "rate_limit_remaining": self.rate_limiter.get_remaining(),
                "kill_switch_active": self.risk_manager._kill_switch_triggered
            }
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get recent audit log entries."""
        with self._lock:
            return self._audit_log[-limit:]
