from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Set
import json
import asyncio
import redis.asyncio as redis
import logging
import os
from datetime import datetime, timezone
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:8080"
).split(",")

API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
REQUIRED_API_KEY = os.getenv("REQUIRED_API_KEY", None)  # Set in production

app = FastAPI(
    title="TITAN API Gateway", 
    version="3.0.0",
    docs_url="/docs" if os.getenv("ENV", "development") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENV", "development") == "development" else None
)

# Add CORS middleware with security controls
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", API_KEY_HEADER],
    max_age=600,
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_timestamps: Dict[str, datetime] = {}
        self.max_connections = int(os.getenv("MAX_WS_CONNECTIONS", "100"))
        self._lock = asyncio.Lock()
        
    async def validate_api_key(self, api_key: Optional[str]) -> bool:
        """Validate API key if required"""
        if not REQUIRED_API_KEY:
            return True
        return api_key == REQUIRED_API_KEY

    async def connect(self, websocket: WebSocket, client_id: str, api_key: Optional[str] = None):
        # Validate API key
        if not await self.validate_api_key(api_key):
            await websocket.close(code=4001, reason="Invalid API key")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        async with self._lock:
            # Check connection limit
            if len(self.active_connections) >= self.max_connections:
                await websocket.close(code=4002, reason="Too many connections")
                raise HTTPException(status_code=503, detail="Connection limit reached")
            
            # Check for duplicate connection
            if client_id in self.active_connections:
                logger.warning(f"Duplicate connection attempt for client {client_id}")
                old_ws = self.active_connections[client_id]
                try:
                    await old_ws.close()
                except:
                    pass
            
            await websocket.accept()
            self.active_connections[client_id] = websocket
            self.connection_timestamps[client_id] = datetime.now(timezone.utc)
            
        logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")

    async def disconnect(self, client_id: str):
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.connection_timestamps:
                del self.connection_timestamps[client_id]
                
        logger.info(f"Client {client_id} disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients with error handling"""
        disconnected_clients = []
        
        async with self._lock:
            for client_id, connection in list(self.active_connections.items()):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients outside the loop
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def send_to_client(self, client_id: str, message: dict) -> bool:
        """Send message to specific client"""
        async with self._lock:
            if client_id not in self.active_connections:
                return False
            connection = self.active_connections[client_id]
        
        try:
            await connection.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "clients": list(self.active_connections.keys())
        }

manager = ConnectionManager()

async def get_redis_client() -> redis.Redis:
    """Create Redis client with proper configuration"""
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    
    return redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )

@app.on_event("startup")
async def startup_event():
    logger.info("TITAN API Gateway starting...")
    
    # Validate required configuration
    if not os.getenv("REDIS_HOST"):
        logger.warning("REDIS_HOST not set, using default 'redis'")
    
    # Start Redis listener task
    asyncio.create_task(redis_listener())
    logger.info("Redis listener task started")

async def redis_listener():
    """Listen to Redis pub/sub and broadcast to WebSocket clients"""
    retry_count = 0
    max_retries = 5
    backoff = 1
    
    while retry_count < max_retries:
        try:
            r = await get_redis_client()
            pubsub = r.pubsub()
            
            await pubsub.subscribe("market_data")
            logger.info("Subscribed to Redis channel: market_data")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await manager.broadcast(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from Redis: {e}")
                    except Exception as e:
                        logger.error(f"Error broadcasting message: {e}")
                        
            retry_count = 0  # Reset on successful connection
            
        except redis.ConnectionError as e:
            retry_count += 1
            logger.error(f"Redis connection error (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            else:
                logger.error("Max Redis retries reached")
                raise
        except Exception as e:
            logger.error(f"Redis listener error: {e}", exc_info=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

@app.websocket("/ws/market")
async def websocket_endpoint(
    websocket: WebSocket,
    api_key: Optional[str] = None
):
    """WebSocket endpoint for real-time market data"""
    # Generate client ID or use provided one
    client_id = f"client_{secrets.token_hex(8)}"
    
    try:
        await manager.connect(websocket, client_id, api_key)
        
        # Send initial snapshot
        try:
            r = await get_redis_client()
            snapshot = await r.hgetall("latest_prices")
            if snapshot:
                await websocket.send_json({
                    "type": "snapshot", 
                    "data": snapshot,
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
                })
        except Exception as e:
            logger.error(f"Error fetching snapshot: {e}")
        
        # Keep connection alive with heartbeat
        while True:
            try:
                # Wait for ping or close message
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Echo back as pong
                await websocket.send_text(f"pong: {message}")
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
                    })
                except:
                    break
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected normally")
                break
            except Exception as e:
                logger.error(f"WebSocket error for {client_id}: {e}")
                break
                
    except HTTPException:
        pass  # Already handled in connect
    finally:
        await manager.disconnect(client_id)

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint with Redis connectivity verification"""
    health_status = {
        "status": "healthy",
        "service": "titan-api",
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "version": "3.0.0"
    }
    
    # Check Redis connectivity
    try:
        r = await get_redis_client()
        await r.ping()
        health_status["redis"] = "connected"
    except Exception as e:
        health_status["redis"] = f"disconnected: {str(e)}"
        health_status["status"] = "degraded"
    
    # Connection stats
    health_status["websocket_connections"] = manager.get_stats()["active_connections"]
    
    return health_status

@app.get("/")
async def root():
    return {
        "message": "Welcome to TITAN 3.0 API Gateway",
        "docs": "/docs" if os.getenv("ENV", "development") == "development" else "Disabled in production",
        "websocket": "/ws/market",
        "health": "/health"
    }

@app.get("/stats")
async def get_stats():
    """Get API gateway statistics"""
    return {
        "websocket": manager.get_stats(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
    }

# Middleware for logging and monitoring
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now(timezone.utc)
    
    response = await call_next(request)
    
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    
    return response
