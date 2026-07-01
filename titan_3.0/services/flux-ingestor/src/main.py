import asyncio
import websockets
import json
import logging
import os
import redis.asyncio as redis
from datetime import datetime, timezone
from typing import Optional, Set, List
import ssl
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceIngestor:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        # Configurable WebSocket URL with environment override
        self.uri = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws")
        # Load symbols from environment or use defaults
        default_symbols = "btcusdt,ethusdt,bnbusdt,solusdt,xrpusdt"
        self.symbols: List[str] = os.getenv("BINANCE_SYMBOLS", default_symbols).split(",")
        self.backoff = 1
        self.max_backoff = int(os.getenv("MAX_BACKOFF_SECONDS", "60"))
        self._running = False
        self._message_count = 0
        self._last_message_time: Optional[datetime] = None
        # Validate symbols to prevent injection
        self._validate_symbols()
        
    def _validate_symbols(self) -> None:
        """Validate symbol names to prevent injection attacks"""
        valid_pattern = set("abcdefghijklmnopqrstuvwxyz0123456789")
        invalid_symbols = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower().strip()
            if not symbol_lower or not all(c in valid_pattern for c in symbol_lower):
                logger.warning(f"Invalid symbol '{symbol}' removed from list")
                invalid_symbols.append(symbol)
        
        for symbol in invalid_symbols:
            self.symbols.remove(symbol)
            
        if not self.symbols:
            raise ValueError("No valid symbols configured")

    async def start(self):
        """Start the ingestor with proper error handling and monitoring"""
        self._running = True
        logger.info(f"Starting Binance Ingestor with symbols: {self.symbols}")
        
        # SSL context for secure WebSocket connection
        ssl_context = ssl.create_default_context()
        
        while self._running:
            try:
                streams = "/".join([f"{sym}@miniTicker" for sym in self.symbols])
                full_uri = f"{self.uri}/{streams}"
                
                logger.info(f"Connecting to Binance WebSocket: {full_uri}")
                async with websockets.connect(
                    full_uri,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=1024 * 1024  # 1MB max message size
                ) as ws:
                    self.backoff = 1
                    logger.info("WebSocket connection established")
                    
                    async for message in ws:
                        if not self._running:
                            break
                            
                        try:
                            data = json.loads(message)
                            normalized = self.normalize(data)
                            
                            # Validate normalized data before publishing
                            if not self._validate_normalized_data(normalized):
                                logger.warning(f"Invalid normalized data: {normalized}")
                                continue
                            
                            # Publish to Redis with error handling
                            await self.redis.publish("market_data", json.dumps(normalized))
                            
                            # Store latest state with atomic operation
                            await self.redis.hset(
                                "latest_prices", 
                                normalized['symbol'], 
                                json.dumps(normalized)
                            )
                            
                            # Update metrics
                            self._message_count += 1
                            self._last_message_time = datetime.now(timezone.utc)
                            
                            if self._message_count % 100 == 0:
                                logger.info(f"Processed {self._message_count} messages")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}", exc_info=True)
                            continue
                        
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                if self._running:
                    await self._handle_reconnect()
            except Exception as e:
                logger.error(f"Binance connection error: {e}", exc_info=True)
                if self._running:
                    await self._handle_reconnect()

    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff"""
        logger.info(f"Reconnecting in {self.backoff} seconds...")
        await asyncio.sleep(self.backoff)
        self.backoff = min(self.backoff * 2, self.max_backoff)
    
    def _validate_normalized_data(self, data: dict) -> bool:
        """Validate normalized data structure"""
        required_fields = ['symbol', 'exchange', 'price', 'timestamp']
        for field in required_fields:
            if field not in data:
                return False
        if not isinstance(data['price'], (int, float)) or data['price'] <= 0:
            return False
        if not isinstance(data['timestamp'], int) or data['timestamp'] <= 0:
            return False
        return True

    def normalize(self, data: dict) -> dict:
        """Normalize WebSocket message to standard format"""
        symbol = data.get('s', 'UNKNOWN')
        if not symbol or not isinstance(symbol, str):
            symbol = 'UNKNOWN'
            
        price = data.get('c', 0)
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 0.0
            
        volume = data.get('v', 0)
        try:
            volume = float(volume)
        except (TypeError, ValueError):
            volume = 0.0
            
        bid = data.get('b', 0)
        try:
            bid = float(bid)
        except (TypeError, ValueError):
            bid = 0.0
            
        ask = data.get('a', 0)
        try:
            ask = float(ask)
        except (TypeError, ValueError):
            ask = 0.0
        
        return {
            "symbol": symbol.lower(),
            "exchange": "BINANCE",
            "price": price,
            "volume": volume,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "bid": bid,
            "ask": ask,
            "asset_class": "CRYPTO"
        }
    
    def stop(self) -> None:
        """Stop the ingestor gracefully"""
        logger.info("Stopping Binance Ingestor...")
        self._running = False

async def main():
    """Main entry point with proper error handling and graceful shutdown"""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    
    # Validate Redis configuration
    if not redis_host:
        logger.error("REDIS_HOST environment variable is required")
        return
    
    redis_client = redis.Redis(
        host=redis_host, 
        port=redis_port, 
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    
    logger.info("Starting Flux Ingestor Service...")
    ingestor = BinanceIngestor(redis_client)
    
    try:
        await ingestor.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        ingestor.stop()
    except Exception as e:
        logger.error(f"Fatal error in ingestor: {e}", exc_info=True)
        ingestor.stop()
        raise
    finally:
        logger.info("Closing Redis connection...")
        await redis_client.close()
        logger.info("Flux Ingestor shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        exit(1)
