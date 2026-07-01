import asyncio
import logging
import os
import json
import redis.asyncio as redis
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Set
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TitanExecutor:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.mode = os.getenv("MODE", "PAPER")
        self._running = False
        # Rate limiting for executions
        self.max_executions_per_minute = int(os.getenv("MAX_EXECUTIONS_PER_MINUTE", "30"))
        self._execution_timestamps = []
        # Track pending orders to prevent duplicates
        self._pending_orders: Set[str] = set()
        # Kill switch
        self._kill_switch_triggered = False
        
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits for execution"""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - 60
        self._execution_timestamps = [ts for ts in self._execution_timestamps if ts > cutoff]
        
        if len(self._execution_timestamps) >= self.max_executions_per_minute:
            logger.warning("Execution rate limit exceeded")
            return False
            
        self._execution_timestamps.append(now.timestamp())
        return True
    
    def _validate_signal(self, signal: dict) -> bool:
        """Validate trade signal before execution"""
        required_fields = ['symbol', 'action', 'confidence']
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Missing required field: {field}")
                return False
        
        if signal.get('action') not in ['BUY', 'SELL', 'HOLD']:
            logger.warning(f"Invalid action: {signal.get('action')}")
            return False
            
        if not isinstance(signal.get('confidence'), (int, float)) or not 0 <= signal['confidence'] <= 1:
            logger.warning(f"Invalid confidence: {signal.get('confidence')}")
            return False
            
        # Validate symbol format
        symbol = signal.get('symbol', '')
        if not symbol or not isinstance(symbol, str) or len(symbol) > 20:
            logger.warning(f"Invalid symbol: {symbol}")
            return False
            
        return True
    
    def _generate_order_id(self, signal: dict) -> str:
        """Generate unique order ID with collision prevention"""
        timestamp = datetime.now(timezone.utc).isoformat()
        signal_hash = hashlib.sha256(json.dumps(signal, sort_keys=True).encode()).hexdigest()[:16]
        return f"ORD_{timestamp}_{signal_hash}"
    
    async def execute_trade(self, signal: dict):
        """Execute trade with validation and rate limiting"""
        if self._kill_switch_triggered:
            logger.error("Kill switch triggered - trading disabled")
            return
        
        if not await self._check_rate_limit():
            logger.warning("Skipping execution due to rate limit")
            return
        
        # Validate signal
        if not self._validate_signal(signal):
            logger.error(f"Invalid signal rejected: {signal}")
            return
        
        # Generate order ID
        order_id = self._generate_order_id(signal)
        
        # Check for duplicate pending orders
        if order_id in self._pending_orders:
            logger.warning(f"Duplicate order detected: {order_id}")
            return
        
        # Add to pending orders
        self._pending_orders.add(order_id)
        
        # Cleanup old pending orders (keep last 100)
        if len(self._pending_orders) > 100:
            # Remove oldest entries
            self._pending_orders = set(list(self._pending_orders)[-100:])
        
        logger.info(f"Executing trade for {signal.get('symbol')} - Order ID: {order_id}")
        
        try:
            # Execute trade based on mode
            if self.mode == "PAPER":
                result = await self._execute_paper_trade(signal, order_id)
            elif self.mode == "LIVE":
                result = await self._execute_live_trade(signal, order_id)
            else:
                logger.error(f"Unknown mode: {self.mode}")
                result = {"status": "REJECTED", "reason": "Unknown mode"}
            
            # Publish execution result
            await self.redis.publish("trade_executions", json.dumps(result))
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}", exc_info=True)
            result = {
                "status": "FAILED",
                "order_id": order_id,
                "error": str(e),
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
            }
            await self.redis.publish("trade_executions", json.dumps(result))
        finally:
            # Remove from pending orders
            self._pending_orders.discard(order_id)
    
    async def _execute_paper_trade(self, signal: dict, order_id: str) -> dict:
        """Execute paper trade (simulation)"""
        return {
            "status": "EXECUTED",
            "mode": self.mode,
            "order_id": order_id,
            "signal": signal,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "paper_trade": True
        }
    
    async def _execute_live_trade(self, signal: dict, order_id: str) -> dict:
        """Execute live trade - placeholder for real exchange integration"""
        # In production, this would integrate with exchange APIs
        logger.warning("LIVE mode not fully implemented - using paper execution")
        return {
            "status": "EXECUTED",
            "mode": self.mode,
            "order_id": order_id,
            "signal": signal,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "live_trade": False,
            "warning": "Live trading not configured"
        }
    
    def trigger_kill_switch(self):
        """Manually trigger kill switch to stop all trading"""
        logger.critical("Kill switch manually triggered")
        self._kill_switch_triggered = True
    
    def reset_kill_switch(self):
        """Reset kill switch after review"""
        logger.info("Kill switch reset")
        self._kill_switch_triggered = False
    
    async def start(self):
        """Start executor with proper error handling"""
        self._running = True
        logger.info(f"TITAN Executor started in {self.mode} mode")
        
        retry_count = 0
        max_retries = 5
        backoff = 1
        
        while self._running and retry_count < max_retries:
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe("trade_signals")
                logger.info("Subscribed to trade_signals channel")
                
                retry_count = 0
                
                async for message in pubsub.listen():
                    if not self._running:
                        break
                        
                    if message['type'] == 'message':
                        try:
                            signal = json.loads(message['data'])
                            logger.info(f"Received signal: {signal}")
                            await self.execute_trade(signal)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing signal: {e}", exc_info=True)
                            
            except redis.ConnectionError as e:
                retry_count += 1
                logger.error(f"Redis connection error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries and self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
            except Exception as e:
                logger.error(f"Executor error: {e}", exc_info=True)
                if self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        
        if retry_count >= max_retries:
            logger.error("Max retries reached, shutting down executor")
    
    def stop(self):
        """Stop executor gracefully"""
        logger.info("Stopping TITAN Executor...")
        self._running = False

async def main():
    redis_host = os.getenv("REDIS_HOST", "redis")
    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    
    executor = TitanExecutor(r)
    await executor.start()

if __name__ == "__main__":
    asyncio.run(main())
