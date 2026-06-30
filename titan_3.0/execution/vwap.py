"""Volume Weighted Average Price (VWAP) execution algorithm."""

import numpy as np
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VWAPExecutor:
    """
    Volume Weighted Average Price execution strategy.
    
    Slices orders based on historical volume profiles to minimize
    market impact and achieve execution close to VWAP.
    """
    
    def __init__(self, volume_profile: Optional[np.ndarray] = None,
                 participation_rate: float = 0.1,
                 max_order_size: float = 10000):
        """
        Initialize VWAP executor.
        
        Args:
            volume_profile: Historical volume distribution across time buckets
            participation_rate: Max % of market volume to participate in
            max_order_size: Maximum order size per slice
        """
        self.volume_profile = volume_profile
        self.participation_rate = participation_rate
        self.max_order_size = max_order_size
        self.executed_quantity = 0.0
        self.executed_value = 0.0
        self.slices_executed = []
        
    def generate_schedule(self, total_quantity: float, 
                         time_buckets: int = 13,  # 30-min buckets in trading day
                         start_time: datetime = None) -> List[Dict]:
        """
        Generate execution schedule based on volume profile.
        
        Args:
            total_quantity: Total quantity to execute
            time_buckets: Number of time buckets for execution
            start_time: Start time for execution
            
        Returns:
            List of scheduled orders with time and quantity
        """
        if start_time is None:
            start_time = datetime.now()
            
        # Use uniform profile if none provided
        if self.volume_profile is None:
            self.volume_profile = np.ones(time_buckets) / time_buckets
        
        # Normalize volume profile
        volume_profile = self.volume_profile / np.sum(self.volume_profile)
        
        schedule = []
        remaining_quantity = total_quantity
        
        for i, vol_weight in enumerate(volume_profile):
            # Calculate quantity for this bucket
            bucket_quantity = min(
                total_quantity * vol_weight,
                remaining_quantity,
                self.max_order_size
            )
            
            if bucket_quantity <= 0:
                continue
                
            # Calculate time for this bucket
            bucket_time = start_time + timedelta(minutes=30 * i)
            
            schedule.append({
                'bucket': i,
                'time': bucket_time,
                'quantity': bucket_quantity,
                'cumulative_qty': total_quantity * np.sum(volume_profile[:i+1]),
                'target_vwap_pct': np.sum(volume_profile[:i+1]) * 100
            })
            
            remaining_quantity -= bucket_quantity
            
        logger.info(f"Generated VWAP schedule: {len(schedule)} buckets, "
                   f"total qty: {total_quantity}")
        
        return schedule
    
    def execute_slice(self, slice_order: Dict, current_price: float,
                     market_volume: float) -> Dict:
        """
        Execute a single slice of the VWAP schedule.
        
        Args:
            slice_order: Order details from schedule
            current_price: Current market price
            market_volume: Current market volume in this bucket
            
        Returns:
            Execution result with filled quantity and price
        """
        # Limit participation to avoid market impact
        max_participate = market_volume * self.participation_rate
        execute_qty = min(slice_order['quantity'], max_participate, self.max_order_size)
        
        if execute_qty <= 0:
            return {
                'filled_qty': 0,
                'price': current_price,
                'status': 'skipped',
                'reason': 'volume_limit'
            }
        
        # Simulate fill at current price (in real system, would send to broker)
        fill_price = current_price
        fill_value = execute_qty * fill_price
        
        self.executed_quantity += execute_qty
        self.executed_value += fill_value
        
        result = {
            'filled_qty': execute_qty,
            'price': fill_price,
            'value': fill_value,
            'timestamp': datetime.now(),
            'bucket': slice_order['bucket'],
            'status': 'filled'
        }
        
        self.slices_executed.append(result)
        
        logger.debug(f"VWAP slice executed: {execute_qty} @ {fill_price}")
        
        return result
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        if self.executed_quantity == 0:
            return {'status': 'no_execution'}
            
        avg_price = self.executed_value / self.executed_quantity
        
        return {
            'total_quantity': self.executed_quantity,
            'total_value': self.executed_value,
            'average_price': avg_price,
            'slices_executed': len(self.slices_executed),
            'vwap': avg_price,
            'status': 'in_progress' if self.slices_executed else 'not_started'
        }
    
    def reset(self):
        """Reset execution state."""
        self.executed_quantity = 0.0
        self.executed_value = 0.0
        self.slices_executed = []
        logger.info("VWAP executor reset")
