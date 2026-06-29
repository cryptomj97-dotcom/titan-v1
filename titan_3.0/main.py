#!/usr/bin/env python3
"""
TITAN 3.0 - Main Orchestration Script
Starts the complete autonomous trading ecosystem.

Usage:
    python main.py              # Start full system (API + Monitoring)
    python main.py --backtest   # Run backtesting engine
    python main.py --train      # Train RL models
    python main.py --debate     # Run standalone debate simulation
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import TITANConfig, ConfigManager
from core.logging import TITANLogger
from core.health import HealthChecker

def setup_logging(config: TITANConfig):
    """Initialize logging system."""
    logger = TITANLogger(
        name="TITAN_3.0",
        log_level=config.system.log_level,
        log_dir=config.system.logs_dir
    )
    return logger.get_logger()

async def run_api_server(config: TITANConfig, logger: logging.Logger):
    """Start the FastAPI backend server."""
    logger.info("Starting TITAN 3.0 API Server...")
    try:
        from api.main import app
        import uvicorn
        
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level=config.system.log_level.lower()
        )
    except Exception as e:
        logger.error(f"API Server failed: {e}")
        raise

async def run_monitoring_dashboard(config: TITANConfig, logger: logging.Logger):
    """Start the monitoring dashboard server."""
    logger.info("Starting TITAN 3.0 Monitoring Dashboard...")
    try:
        from monitoring.dashboard_server import start_dashboard
        await start_dashboard(
            port=config.monitoring.dashboard_port,
            log_level=config.system.log_level
        )
    except Exception as e:
        logger.error(f"Monitoring Dashboard failed: {e}")
        raise

async def run_backtest(config: TITANConfig, logger: logging.Logger):
    """Execute comprehensive backtesting."""
    logger.info("Starting TITAN 3.0 Backtesting Engine...")
    try:
        from backtest.engine import BacktestEngine
        from backtest.walk_forward import WalkForwardAnalyzer
        
        # Initialize engine
        engine = BacktestEngine(config=config, logger=logger)
        
        # Run backtest
        results = await engine.run(
            start_date=config.backtest.start_date,
            end_date=config.backtest.end_date,
            initial_capital=config.backtest.initial_capital
        )
        
        # Generate report
        from backtest.report_generator import ReportGenerator
        reporter = ReportGenerator(results, config)
        reporter.generate_html_report("./reports/backtest_report.html")
        
        logger.info(f"Backtest Complete. Final Equity: ${results.final_equity:,.2f}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        raise

async def train_rl_models(config: TITANConfig, logger: logging.Logger):
    """Train Reinforcement Learning models."""
    logger.info("Starting RL Model Training...")
    try:
        from ml.rl_agent import PPOAgent
        from data.ingestion.data_pipeline import DataPipeline
        
        # Fetch training data
        pipeline = DataPipeline(config)
        data = await pipeline.fetch_historical("AAPL", period="2y")
        
        # Initialize and train agent
        agent = PPOAgent(config=config, logger=logger)
        await agent.train(data, total_timesteps=config.rl.training.total_timesteps)
        
        # Save model
        agent.save_model(config.rl.model_path)
        logger.info(f"RL Model saved to {config.rl.model_path}")
        
    except Exception as e:
        logger.error(f"RL Training failed: {e}")
        raise

async def run_debate_simulation(config: TITANConfig, logger: logging.Logger):
    """Run a standalone adversarial debate simulation."""
    logger.info("Starting Adversarial Debate Simulation...")
    try:
        from debate.integration import DebateOrchestrator
        from ml.regime_detection.regime_detector import RegimeDetector
        
        # Mock signal for demonstration
        signal = {
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 0.78,
            "price": 175.50,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Get current regime
        detector = RegimeDetector(config)
        regime = await detector.detect("AAPL")
        
        # Run debate
        orchestrator = DebateOrchestrator(config, logger)
        verdict = await orchestrator.run_debate(signal, regime)
        
        logger.info(f"Debate Verdict: {verdict.decision}")
        logger.info(f"Consensus Score: {verdict.consensus_score:.2f}")
        logger.info(f"Judge Reasoning: {verdict.judge_reasoning}")
        
    except Exception as e:
        logger.error(f"Debate Simulation failed: {e}")
        raise

async def health_monitor(config: TITANConfig, logger: logging.Logger):
    """Continuous health monitoring loop."""
    checker = HealthChecker(config, logger)
    
    while True:
        await asyncio.sleep(30)  # Check every 30 seconds
        health_status = await checker.check_all()
        
        if not health_status.healthy:
            logger.warning(f"Health Check Failed: {health_status.issues}")
            
            if health_status.critical:
                logger.error("CRITICAL FAILURE - Triggering Kill Switch")
                # Trigger kill switch logic here
        else:
            logger.debug("System Health: OK")

async def main():
    parser = argparse.ArgumentParser(description="TITAN 3.0 Autonomous Trading System")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--train", action="store_true", help="Train RL models")
    parser.add_argument("--debate", action="store_true", help="Run debate simulation")
    parser.add_argument("--config", type=str, default="config/master_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.load()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info(f"TITAN 3.0 - Version {config.system.version}")
    logger.info(f"Environment: {config.system.environment}")
    logger.info("=" * 60)
    
    # Initialize health checker
    health_checker = HealthChecker(config, logger)
    startup_health = await health_checker.check_all()
    
    if not startup_health.healthy:
        logger.error(f"Startup Health Check Failed: {startup_health.issues}")
        sys.exit(1)
    
    logger.info("System Health Check: PASSED")
    
    try:
        if args.backtest:
            await run_backtest(config, logger)
        elif args.train:
            await train_rl_models(config, logger)
        elif args.debate:
            await run_debate_simulation(config, logger)
        else:
            # Run full system: API + Monitoring + Health Monitor
            tasks = [
                run_api_server(config, logger),
                run_monitoring_dashboard(config, logger),
                health_monitor(config, logger)
            ]
            
            logger.info("Starting all services...")
            await asyncio.gather(*tasks)
    
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Closing TITAN 3.0...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("TITAN 3.0 shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
