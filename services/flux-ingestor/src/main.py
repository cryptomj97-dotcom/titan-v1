import asyncio
import websockets
import json
import logging
import os
import redis.asyncio as redis
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceIngestor:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.uri = "wss://stream.binance.com:9443/ws"
        self.symbols = ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt"]
        self.backoff = 1

    async def start(self):
        while True:
            try:
                streams = "/".join([f"{sym}@miniTicker" for sym in self.symbols])
                full_uri = f"{self.uri}/{streams}"
                
                logger.info(f"Connecting to Binance WebSocket: {full_uri}")
                async with websockets.connect(full_uri) as ws:
                    self.backoff = 1
                    async for message in ws:
                        data = json.loads(message)
                        normalized = self.normalize(data)
                        
                        # Publish to Redis
                        await self.redis.publish("market_data", json.dumps(normalized))
                        
                        # Store latest state
                        await self.redis.hset("latest_prices", mapping={normalized['symbol']: json.dumps(normalized)})
                        
                        if len(self.symbols) < 10:  # Only log for small sets
                            logger.debug(f"Published: {normalized['symbol']} @ {normalized['price']}")
                        
            except Exception as e:
                logger.error(f"Binance connection error: {e}")
                await asyncio.sleep(self.backoff)
                self.backoff = min(self.backoff * 2, 60)

    def normalize(self, data: dict) -> dict:
        return {
            "symbol": data.get('s', 'UNKNOWN').lower(),
            "exchange": "BINANCE",
            "price": float(data.get('c', 0)),
            "volume": float(data.get('v', 0)),
            "timestamp": int(datetime.now().timestamp() * 1000),
            "bid": float(data.get('b', 0)),
            "ask": float(data.get('a', 0)),
            "asset_class": "CRYPTO"
        }

async def main():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    
    logger.info("Starting Flux Ingestor Service...")
    ingestor = BinanceIngestor(redis_client)
    
    try:
        await ingestor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully")
    finally:
        await redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())
