from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio
import redis.asyncio as redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TITAN API Gateway", version="3.0.0")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    logger.info("TITAN API Gateway starting...")
    # Start Redis listener task
    asyncio.create_task(redis_listener())

async def redis_listener():
    redis_host = "redis"  # Docker service name
    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    pubsub = r.pubsub()
    
    await pubsub.subscribe("market_data")
    logger.info("Subscribed to Redis channel: market_data")
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            await manager.broadcast(data)

@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send initial snapshot
    r = redis.Redis(host="redis", port=6379, decode_responses=True)
    snapshot = await r.hgetall("latest_prices")
    if snapshot:
        await websocket.send_json({"type": "snapshot", "data": snapshot})
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "titan-api"}

@app.get("/")
async def root():
    return {
        "message": "Welcome to TITAN 3.0 API Gateway",
        "docs": "/docs",
        "websocket": "/ws/market"
    }
