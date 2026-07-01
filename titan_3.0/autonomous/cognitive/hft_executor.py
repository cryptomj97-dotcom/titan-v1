"""
5. High-Frequency Micro-Structure Executor
Executes trades with sub-millisecond latency using advanced order algorithms.
"""

import time
import queue
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import threading

logger = logging.getLogger(__name__)

@dataclass
class Order:
    order_id: str
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    order_type: str  # 'MARKET', 'LIMIT', 'PEGGED'
    timestamp: float

@dataclass
class ExecutionReport:
    order_id: str
    status: str  # 'EXECUTED', 'PARTIAL', 'REJECTED', 'CANCELLED'
    executed_quantity: float
    executed_price: float
    timestamp: float
    latency_ms: float
    slippage: float

class HFTExecutor:
    """
    High-Frequency Trading Executor
    Implements micro-structure aware order execution with minimal latency
    """
    
    def __init__(self, default_position_size: float = 0.01):
        self.default_position_size = default_position_size
        self.order_queue = queue.Queue(maxsize=10000)
        self.executed_orders: List[ExecutionReport] = []
        self.pending_orders: Dict[str, Order] = {}
        self.running = False
        self.execution_thread: Optional[threading.Thread] = None
        self.stats = {
            'total_orders': 0,
            'executed_orders': 0,
            'rejected_orders': 0,
            'avg_latency_ms': 0.0,
            'total_slippage': 0.0
        }
        self.lock = threading.Lock()
        
    def start(self) -> None:
        """Start execution engine"""
        if self.running:
            logger.warning("HFT Executor already running")
            return
            
        self.running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        logger.info("HFT Executor started")
        
    def stop(self) -> None:
        """Stop execution engine"""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=2.0)
        logger.info("HFT Executor stopped")
        
    def execute_order(
        self, 
        action: str, 
        market_data: Any,
        quantity: Optional[float] = None,
        confidence: float = 0.5
    ) -> ExecutionReport:
        """
        Execute a trade with micro-structure optimization
        
        Args:
            action: 'BUY' or 'SELL'
            market_data: Current market data
            quantity: Order quantity (uses default if None)
            confidence: Trade confidence score (affects sizing)
            
        Returns:
            Execution report with latency and slippage metrics
        """
        if action == 'HOLD':
            return ExecutionReport(
                order_id=f"skip_{int(time.time()*1000000)}",
                status='SKIPPED',
                executed_quantity=0.0,
                executed_price=0.0,
                timestamp=time.time(),
                latency_ms=0.0,
                slippage=0.0
            )
        
        # Determine order size based on confidence
        base_quantity = quantity or self.default_position_size
        if confidence > 0.8:
            order_quantity = base_quantity * 2.0  # Double size for high confidence
        elif confidence < 0.4:
            order_quantity = base_quantity * 0.5  # Half size for low confidence
        else:
            order_quantity = base_quantity
        
        # Create order
        order_id = f"hft_{int(time.time()*1000000)}"
        execution_start = time.perf_counter()
        
        # Determine execution price based on order book microstructure
        if action == 'BUY':
            execution_price = market_data.ask  # Pay ask price
        else:
            execution_price = market_data.bid  # Receive bid price
        
        # Simulate ultra-low latency execution
        time.sleep(0.00005)  # 50 microseconds simulated latency
        
        execution_end = time.perf_counter()
        latency_ms = (execution_end - execution_start) * 1000
        
        # Calculate slippage (difference from mid-price)
        mid_price = (market_data.bid + market_data.ask) / 2
        slippage = abs(execution_price - mid_price) / mid_price
        
        # Create execution report
        report = ExecutionReport(
            order_id=order_id,
            status='EXECUTED',
            executed_quantity=order_quantity,
            executed_price=execution_price,
            timestamp=time.time(),
            latency_ms=latency_ms,
            slippage=slippage
        )
        
        # Store execution
        with self.lock:
            self.executed_orders.append(report)
            self.stats['total_orders'] += 1
            self.stats['executed_orders'] += 1
            self.stats['avg_latency_ms'] = (
                (self.stats['avg_latency_ms'] * (self.stats['executed_orders'] - 1) + latency_ms) 
                / self.stats['executed_orders']
            )
            self.stats['total_slippage'] += slippage
            
            # Keep only recent executions
            if len(self.executed_orders) > 10000:
                self.executed_orders = self.executed_orders[-5000:]
        
        return report
    
    def _execution_loop(self) -> None:
        """Background execution loop for queued orders"""
        while self.running:
            try:
                # Process queued orders with priority
                if not self.order_queue.empty():
                    order = self.order_queue.get_nowait()
                    # Process order (simplified - real implementation would be more complex)
                    
                time.sleep(0.0001)  # 100 microsecond cycle
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"HFT execution loop error: {e}")
                time.sleep(0.001)
    
    def get_recent_executions(self, count: int = 100) -> List[ExecutionReport]:
        """Get most recent executions"""
        with self.lock:
            return self.executed_orders[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        with self.lock:
            stats = self.stats.copy()
            stats['recent_executions'] = len(self.executed_orders)
            stats['pending_orders'] = len(self.pending_orders)
            return stats
    
    def cancel_all_pending(self) -> int:
        """Cancel all pending orders"""
        with self.lock:
            count = len(self.pending_orders)
            self.pending_orders.clear()
            logger.info(f"Cancelled {count} pending orders")
            return count
