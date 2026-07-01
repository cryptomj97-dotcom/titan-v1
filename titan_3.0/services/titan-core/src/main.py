import asyncio
import logging
import os
import json
import redis.asyncio as redis
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TitanCore:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._running = False
        self._message_count = 0
        self._last_signal_time: Optional[datetime] = None
        # Rate limiting for signals (max signals per minute)
        self.max_signals_per_minute = int(os.getenv("MAX_SIGNALS_PER_MINUTE", "60"))
        self._signal_timestamps = []
        
    async def start(self):
        """Start the core strategy engine with proper error handling"""
        self._running = True
        logger.info("TITAN Core Strategy Engine started")
        
        retry_count = 0
        max_retries = 5
        backoff = 1
        
        while self._running and retry_count < max_retries:
            try:
                # Subscribe to market data
                pubsub = self.redis.pubsub()
                await pubsub.subscribe("market_data")
                logger.info("Subscribed to market_data channel")
                
                retry_count = 0  # Reset on successful connection
                
                async for message in pubsub.listen():
                    if not self._running:
                        break
                        
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            
                            # Validate incoming data
                            if not self._validate_market_data(data):
                                logger.warning(f"Invalid market data received: {data}")
                                continue
                            
                            # Process through regime detection
                            regime = await self.detect_regime(data)
                            
                            # Generate strategy signal with rate limiting
                            if await self._check_rate_limit():
                                signal = await self.generate_signal(data, regime)
                                
                                # Publish signal to executor if valid
                                if signal:
                                    await self._publish_signal(signal)
                                    self._message_count += 1
                                    self._last_signal_time = datetime.now(timezone.utc)
                                    
                                    if self._message_count % 10 == 0:
                                        logger.info(f"Generated {self._message_count} signals")
                            else:
                                logger.debug("Rate limit exceeded, skipping signal generation")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing market data: {e}", exc_info=True)
                            continue
                            
            except redis.ConnectionError as e:
                retry_count += 1
                logger.error(f"Redis connection error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries and self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
            except Exception as e:
                logger.error(f"Core engine error: {e}", exc_info=True)
                if self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        
        if retry_count >= max_retries:
            logger.error("Max retries reached, shutting down core engine")
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits for signal generation"""
        now = datetime.now(timezone.utc)
        # Keep only timestamps from the last minute
        cutoff = now.timestamp() - 60
        self._signal_timestamps = [ts for ts in self._signal_timestamps if ts > cutoff]
        
        if len(self._signal_timestamps) >= self.max_signals_per_minute:
            return False
            
        self._signal_timestamps.append(now.timestamp())
        return True
    
    def _validate_market_data(self, data: dict) -> bool:
        """Validate market data structure"""
        required_fields = ['symbol', 'price', 'timestamp']
        for field in required_fields:
            if field not in data:
                return False
        
        if not isinstance(data.get('price'), (int, float)) or data['price'] <= 0:
            return False
            
        if not isinstance(data.get('timestamp'), int) or data['timestamp'] <= 0:
            return False
            
        return True
    
    async def _publish_signal(self, signal: dict) -> None:
        """Publish trade signal with validation"""
        # Add metadata
        signal['generated_at'] = int(datetime.now(timezone.utc).timestamp() * 1000)
        signal['core_version'] = '3.0.0'
        
        # Validate signal before publishing
        if not self._validate_signal(signal):
            logger.error(f"Invalid signal generated: {signal}")
            return
        
        await self.redis.publish("trade_signals", json.dumps(signal))
        logger.debug(f"Published signal: {signal.get('symbol')} - {signal.get('action')}")
    
    def _validate_signal(self, signal: dict) -> bool:
        """Validate trade signal structure"""
        required_fields = ['symbol', 'action', 'confidence']
        for field in required_fields:
            if field not in signal:
                return False
        
        if signal.get('action') not in ['BUY', 'SELL', 'HOLD']:
            return False
            
        if not isinstance(signal.get('confidence'), (int, float)) or not 0 <= signal['confidence'] <= 1:
            return False
            
        return True
    
    async def detect_regime(self, data: dict) -> str:
        """
        Detect market regime from price data
        Returns: BULLISH, BEARISH, NEUTRAL, VOLATILE
        """
        try:
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            
            # Simple regime detection (placeholder for ML model)
            if volume > 1000000:  # High volume
                return "VOLATILE"
            elif price > data.get('bid', 0) and price < data.get('ask', 0):
                return "NEUTRAL"
            else:
                return "NEUTRAL"
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "NEUTRAL"
    
    async def generate_signal(self, data: dict, regime: str) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on market data and regime
        Returns signal dict or None if no signal
        """
        try:
            # Placeholder for actual strategy logic
            # In production, this would use ML models, technical indicators, etc.
            
            symbol = data.get('symbol', 'UNKNOWN')
            price = data.get('price', 0)
            
            # Simple momentum-based signal (placeholder)
            confidence = 0.5  # Neutral confidence
            
            if regime == "VOLATILE":
                # Reduce confidence in volatile markets
                confidence = 0.3
                action = "HOLD"
            else:
                action = "HOLD"
            
            # Only return signal if confidence is above threshold
            if confidence >= 0.7:
                return {
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence,
                    'price': price,
                    'regime': regime,
                    'strategy': 'core_default'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None
    
    def stop(self) -> None:
        """Stop the core engine gracefully"""
        logger.info("Stopping TITAN Core...")
        self._running = False

async def main():
    """Main entry point with proper configuration and error handling"""
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    
    # Validate configuration
    if not redis_host:
        logger.error("REDIS_HOST environment variable is required")
        return
    
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    
    logger.info("Starting TITAN Core Service...")
    core = TitanCore(r)
    
    try:
        await core.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        core.stop()
    except Exception as e:
        logger.error(f"Fatal error in core: {e}", exc_info=True)
        core.stop()
        raise
    finally:
        logger.info("Closing Redis connection...")
        await r.close()
        logger.info("TITAN Core shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        exit(1)
