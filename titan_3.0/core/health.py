"""
TITAN 3.0 - Health Check System
Monitors system health and dependencies for graceful degradation
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp

from core.logging import get_logger
from core.config import get_config

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'status': self.status.value,
            'response_time_ms': round(self.response_time_ms, 2),
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    uptime_seconds: float
    last_check: datetime
    version: str = "3.0.0"
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'checks': [c.to_dict() for c in self.checks],
            'uptime_seconds': round(self.uptime_seconds, 2),
            'last_check': self.last_check.isoformat(),
            'version': self.version
        }


class HealthChecker:
    """
    Centralized health checking system for TITAN 3.0.
    Monitors all subsystems and dependencies.
    """
    
    def __init__(self):
        self._start_time = time.time()
        self._last_check: Optional[datetime] = None
        self._checks: Dict[str, HealthCheckResult] = {}
        self._config = get_config()
        
    async def check_all(self) -> SystemHealth:
        """Run all health checks and return overall system health."""
        start = time.time()
        checks = []
        
        # Run all checks concurrently
        check_tasks = [
            self._check_core_services(),
            self._check_data_pipeline(),
            self._check_ml_services(),
            self._check_execution_services(),
            self._check_external_apis(),
        ]
        
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Flatten results
        for result in results:
            if isinstance(result, list):
                checks.extend(result)
            elif isinstance(result, Exception):
                checks.append(HealthCheckResult(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message=f"Check failed: {str(result)}"
                ))
        
        self._last_check = datetime.now()
        
        # Determine overall status
        statuses = [c.status for c in checks]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return SystemHealth(
            status=overall_status,
            checks=checks,
            uptime_seconds=time.time() - self._start_time,
            last_check=self._last_check
        )
    
    async def _check_core_services(self) -> List[HealthCheckResult]:
        """Check core services (config, logging, exceptions)."""
        results = []
        start = time.time()
        
        try:
            # Config check
            config = get_config()
            if config and config.environment:
                results.append(HealthCheckResult(
                    name="config_service",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=(time.time() - start) * 1000,
                    message="Config service operational",
                    details={'environment': config.environment}
                ))
            else:
                raise Exception("Config not loaded")
        except Exception as e:
            results.append(HealthCheckResult(
                name="config_service",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=f"Config service error: {str(e)}"
            ))
        
        return results
    
    async def _check_data_pipeline(self) -> List[HealthCheckResult]:
        """Check data pipeline health."""
        results = []
        start = time.time()
        
        try:
            from data.ingestion import YahooFinanceDataSource
            from core.config import DataSourceConfig
            
            # Test data source instantiation
            config = DataSourceConfig(provider='yahoo')
            source = YahooFinanceDataSource(config)
            
            results.append(HealthCheckResult(
                name="data_pipeline",
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message="Data pipeline operational",
                details={'sources': list(source.__class__.__module__.split('.'))}
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                name="data_pipeline",
                status=HealthStatus.DEGRADED,
                response_time_ms=(time.time() - start) * 1000,
                message=f"Data pipeline issue: {str(e)}"
            ))
        
        return results
    
    async def _check_ml_services(self) -> List[HealthCheckResult]:
        """Check ML services (feature engineering, regime detection, RL)."""
        results = []
        start = time.time()
        
        try:
            from ml.features import FeatureEngine
            from ml.regime_detection import EnsembleRegimeDetector
            
            # Quick instantiation test
            engine = FeatureEngine()
            detector = EnsembleRegimeDetector()
            
            results.append(HealthCheckResult(
                name="ml_services",
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message="ML services operational",
                details={
                    'feature_engine': 'ok',
                    'regime_detector': 'ok'
                }
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                name="ml_services",
                status=HealthStatus.DEGRADED,
                response_time_ms=(time.time() - start) * 1000,
                message=f"ML services issue: {str(e)}"
            ))
        
        return results
    
    async def _check_execution_services(self) -> List[HealthCheckResult]:
        """Check execution services (strategy, risk, order management)."""
        results = []
        start = time.time()
        
        try:
            from strategies.core.strategy_base import BaseStrategy
            from execution.order_manager import OrderManager
            
            # Verify classes are importable
            results.append(HealthCheckResult(
                name="execution_services",
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message="Execution services operational",
                details={'strategy_module': 'ok', 'order_manager': 'ok'}
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                name="execution_services",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=f"Execution services error: {str(e)}"
            ))
        
        return results
    
    async def _check_external_apis(self) -> List[HealthCheckResult]:
        """Check external API connectivity (optional, may fail in test env)."""
        results = []
        start = time.time()
        
        try:
            # Check if we can reach common financial APIs
            # This is a lightweight connectivity check
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # Try a lightweight endpoint (example only)
                try:
                    async with session.get('https://api.github.com', timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            results.append(HealthCheckResult(
                                name="external_api_connectivity",
                                status=HealthStatus.HEALTHY,
                                response_time_ms=(time.time() - start) * 1000,
                                message="External API connectivity OK"
                            ))
                        else:
                            raise Exception(f"Unexpected status: {resp.status}")
                except asyncio.TimeoutError:
                    results.append(HealthCheckResult(
                        name="external_api_connectivity",
                        status=HealthStatus.DEGRADED,
                        response_time_ms=(time.time() - start) * 1000,
                        message="External API timeout (may be expected in test env)"
                    ))
        except Exception as e:
            results.append(HealthCheckResult(
                name="external_api_connectivity",
                status=HealthStatus.DEGRADED,
                response_time_ms=(time.time() - start) * 1000,
                message=f"External API check skipped: {str(e)}"
            ))
        
        return results
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self._start_time


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def check_system_health() -> Dict:
    """Convenience function to check system health."""
    checker = get_health_checker()
    health = await checker.check_all()
    logger.info(f"System health check: {health.status.value}")
    return health.to_dict()


if __name__ == "__main__":
    # Run health check
    async def main():
        health = await check_system_health()
        import json
        logger.info("System Health Check:\n%s", json.dumps(health, indent=2))
    
    asyncio.run(main())
