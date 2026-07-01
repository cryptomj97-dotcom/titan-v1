import asyncio
import logging
import os
import json
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TitanWorker:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
    async def start(self):
        logger.info("TITAN Worker started - Ready for backtest jobs")
        
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("backtest_jobs")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                job = json.loads(message['data'])
                logger.info(f"Received backtest job: {job.get('id', 'UNKNOWN')}")
                
                # Process backtest (placeholder)
                result = await self.run_backtest(job)
                
                # Publish results
                await self.redis.publish("backtest_results", json.dumps(result))
    
    async def run_backtest(self, job: dict) -> dict:
        # Placeholder for backtest logic
        return {
            "job_id": job.get('id'),
            "status": "COMPLETED",
            "sharpe_ratio": 1.5,
            "total_return": 0.15
        }

async def main():
    redis_host = os.getenv("REDIS_HOST", "redis")
    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    
    worker = TitanWorker(r)
    await worker.start()

if __name__ == "__main__":
    asyncio.run(main())
