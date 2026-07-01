import asyncio
import logging
import os
import json
import redis.asyncio as redis
from prometheus_client import start_http_server, Counter, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
MESSAGES_RECEIVED = Counter('titan_messages_total', 'Total messages processed')
ACTIVE_CONNECTIONS = Gauge('titan_connections_active', 'Active WebSocket connections')

class TitanMonitor:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
    async def start(self):
        logger.info("TITAN Monitor started - Prometheus metrics on :9090")
        
        # Start Prometheus server
        start_http_server(9090)
        
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("market_data", "trade_signals", "trade_executions")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                MESSAGES_RECEIVED.inc()
                channel = message['channel']
                logger.debug(f"Message on {channel}")
    
async def main():
    redis_host = os.getenv("REDIS_HOST", "redis")
    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    
    monitor = TitanMonitor(r)
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
