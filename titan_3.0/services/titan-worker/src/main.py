import asyncio
import logging
import os
import json
import redis.asyncio as redis
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TitanWorker:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._running = False
        # Rate limiting for backtests
        self.max_backtests_per_hour = int(os.getenv("MAX_BACKTESTS_PER_HOUR", "10"))
        self._backtest_timestamps = []
        
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits for backtest execution"""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - 3600  # 1 hour
        self._backtest_timestamps = [ts for ts in self._backtest_timestamps if ts > cutoff]
        
        if len(self._backtest_timestamps) >= self.max_backtests_per_hour:
            logger.warning("Backtest rate limit exceeded")
            return False
            
        self._backtest_timestamps.append(now.timestamp())
        return True
    
    def _validate_job(self, job: dict) -> bool:
        """Validate backtest job parameters"""
        required_fields = ['id', 'strategy', 'start_date', 'end_date']
        for field in required_fields:
            if field not in job:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate date format
        try:
            start = job.get('start_date', '')
            end = job.get('end_date', '')
            if not isinstance(start, str) or not isinstance(end, str):
                return False
        except Exception:
            return False
            
        # Validate strategy name
        strategy = job.get('strategy', '')
        if not strategy or not isinstance(strategy, str) or len(strategy) > 50:
            logger.warning(f"Invalid strategy name: {strategy}")
            return False
            
        return True
    
    async def start(self):
        """Start worker with proper error handling"""
        self._running = True
        logger.info("TITAN Worker started - Ready for backtest jobs")
        
        retry_count = 0
        max_retries = 5
        backoff = 1
        
        while self._running and retry_count < max_retries:
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe("backtest_jobs")
                logger.info("Subscribed to backtest_jobs channel")
                
                retry_count = 0
                
                async for message in pubsub.listen():
                    if not self._running:
                        break
                        
                    if message['type'] == 'message':
                        try:
                            job = json.loads(message['data'])
                            logger.info(f"Received backtest job: {job.get('id', 'UNKNOWN')}")
                            
                            # Validate job
                            if not self._validate_job(job):
                                logger.error(f"Invalid job rejected: {job}")
                                continue
                            
                            # Check rate limit
                            if not await self._check_rate_limit():
                                logger.warning("Skipping job due to rate limit")
                                continue
                            
                            # Process backtest
                            result = await self.run_backtest(job)
                            
                            # Publish results
                            await self.redis.publish("backtest_results", json.dumps(result))
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing job: {e}", exc_info=True)
                            
            except redis.ConnectionError as e:
                retry_count += 1
                logger.error(f"Redis connection error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries and self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                if self._running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        
        if retry_count >= max_retries:
            logger.error("Max retries reached, shutting down worker")
    
    async def run_backtest(self, job: dict) -> dict:
        """Run backtest with validation and error handling"""
        job_id = job.get('id', 'UNKNOWN')
        logger.info(f"Running backtest for job {job_id}")
        
        try:
            # Placeholder for actual backtest logic
            # In production, this would run the actual strategy backtest
            await asyncio.sleep(0.1)  # Simulate processing
            
            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "sharpe_ratio": 1.5,
                "total_return": 0.15,
                "max_drawdown": 0.08,
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
            }
        except Exception as e:
            logger.error(f"Backtest failed for job {job_id}: {e}", exc_info=True)
            return {
                "job_id": job_id,
                "status": "FAILED",
                "error": str(e),
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
            }
    
    def stop(self):
        """Stop worker gracefully"""
        logger.info("Stopping TITAN Worker...")
        self._running = False

async def main():
    redis_host = os.getenv("REDIS_HOST", "redis")
    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    
    worker = TitanWorker(r)
    await worker.start()

if __name__ == "__main__":
    asyncio.run(main())
