import asyncio
import logging
import os
import json
import redis.asyncio as redis
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from datetime import datetime, timezone
from typing import Optional, Dict, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics with proper labels
MESSAGES_RECEIVED = Counter(
    'titan_messages_total', 
    'Total messages processed',
    ['channel', 'service']
)
ACTIVE_CONNECTIONS = Gauge(
    'titan_connections_active', 
    'Active WebSocket connections',
    ['endpoint']
)
MESSAGE_PROCESSING_TIME = Histogram(
    'titan_message_processing_seconds',
    'Time spent processing messages',
    ['channel']
)
ERROR_COUNT = Counter(
    'titan_errors_total',
    'Total errors encountered',
    ['error_type', 'service']
)

class TitanMonitor:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._running = False
        self._channels: Set[str] = set()
        self._start_time: Optional[datetime] = None
        
    async def start(self):
        """Start monitor with proper error handling"""
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        logger.info("TITAN Monitor started - Prometheus metrics on :9090")
        
        # Start Prometheus server
        try:
            start_http_server(9090)
            logger.info("Prometheus metrics server started on port 9090")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            ERROR_COUNT.labels(error_type='startup', service='monitor').inc()
        
        retry_count = 0
        max_retries = 5
        backoff = 1
        
        channels_to_monitor = ["market_data", "trade_signals", "trade_executions"]
        
        while self._running and retry_count < max_retries:
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(*channels_to_monitor)
                self._channels = set(channels_to_monitor)
                logger.info(f"Subscribed to channels: {channels_to_monitor}")
                
                retry_count = 0
                
                async for message in pubsub.listen():
                    if not self._running:
                        break
                        
                    if message['type'] == 'message':
                        channel = message.get('channel', 'unknown')
                        try:
                            # Validate message data
                            data = message.get('data', '')
                            if not data:
                                logger.warning("Empty message received")
                                continue
                                
                            # Try to parse as JSON for validation
                            try:
                                json.loads(data)
                            except json.JSONDecodeError:
                                logger.debug("Non-JSON message received (acceptable)")
                            
                            # Record metrics
                            MESSAGES_RECEIVED.labels(channel=channel, service='monitor').inc()
                            
                            logger.debug(f"Message on {channel}")
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {e}", exc_info=True)
                            ERROR_COUNT.labels(error_type='processing', service='monitor').inc()
                            
            except redis.ConnectionError as e:
                retry_count += 1
                logger.error(f"Redis connection error (attempt {retry_count}/{max_retries}): {e}")
                ERROR_COUNT.labels(error_type='redis_connection', service='monitor').inc()
                if retry_count < max_retries and self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)
                ERROR_COUNT.labels(error_type='general', service='monitor').inc()
                if self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        
        if retry_count >= max_retries:
            logger.error("Max retries reached, shutting down monitor")
    
    def get_uptime(self) -> float:
        """Get monitor uptime in seconds"""
        if self._start_time:
            return (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return 0.0
    
    def stop(self):
        """Stop monitor gracefully"""
        logger.info("Stopping TITAN Monitor...")
        self._running = False
    
async def main():
    redis_host = os.getenv("REDIS_HOST", "redis")
    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    
    monitor = TitanMonitor(r)
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
