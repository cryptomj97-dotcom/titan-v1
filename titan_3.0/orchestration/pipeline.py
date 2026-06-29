"""
TITAN 3.0 - Phase 9: Orchestration & Main Pipeline
Modules:
- pipeline.py: The main orchestrator connecting all components
- main.py: Entry point for running the system
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

# Import internal modules
from data.ingestion import DataIngestionEngine as DataIngestionPipeline
from ml.features import FeatureEngine as FeatureEngineer
from ml.regime_detection import EnsembleRegimeDetector as RegimeDetector
from strategies.strategist import Strategist, Backtester
from ml.rl.rl_environment import TradingEnvironment
from alt_data.sentiment_analyzer import AltDataPipeline
from execution.debate_engine import DebateEngine
from execution.order_manager import OrderManager, RiskManager

logger = logging.getLogger(__name__)

class TITANPipeline:
    """
    Main Orchestrator for TITAN 3.0.
    Connects Data -> Features -> Regime -> Strategy -> RL -> Debate -> Execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_pipeline = DataIngestionPipeline()
        self.feature_engineer = FeatureEngineer()
        self.regime_detector = RegimeDetector()
        self.strategist = Strategist()
        self.backtester = Backtester()
        self.alt_data = AltDataPipeline()
        self.debate_engine = DebateEngine()
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager(self.risk_manager)
        
        self.current_regime = None
        self.market_data = None

    def run_full_cycle(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        logger.info(f"Starting TITAN 3.0 cycle for {symbol}")
        
        # 1. Ingest Data
        logger.info("Step 1: Ingesting Data...")
        from core.config import DataSourceConfig
        from data.ingestion import YahooFinanceDataSource
        config = DataSourceConfig(provider="yahoo", api_key=None)
        datasource = YahooFinanceDataSource(config)
        df = datasource.fetch_price_data(symbol, pd.Timestamp(start_date), pd.Timestamp(end_date), "1d")
        if df.empty:
            return {"error": "No data retrieved"}
        self.market_data = df
        
        # 2. Feature Engineering
        logger.info("Step 2: Engineering Features...")
        df_features = self.feature_engineer.create_features(df)
        
        # 3. Regime Detection
        logger.info("Step 3: Detecting Market Regime...")
        returns = df['close'].pct_change().dropna()
        regimes = self.regime_detector.detect_regimes(returns)
        self.current_regime = regimes.iloc[-1] # Latest regime
        
        # 4. Alternative Data (Simulated)
        logger.info("Step 4: Processing Alternative Data...")
        headlines = [f"{symbol} shows strong momentum", f"Analysts upgrade {symbol}"]
        sentiment_result = self.alt_data.process_news_feed(headlines)
        
        # 5. Strategy Selection
        logger.info("Step 5: Selecting Strategy...")
        strategies = self.strategist.generate_strategies(self.current_regime, list(df_features.columns))
        selected_strategy = strategies[0] if strategies else None
        
        if not selected_strategy:
            return {"error": "No viable strategy found"}
            
        # 6. Backtesting (Walk-Forward)
        logger.info("Step 6: Running Walk-Forward Backtest...")
        # Use last 2 years for demo
        bt_df = df_features.tail(500)
        bt_regimes = regimes.tail(500)
        
        bt_results = self.backtester.run_walk_forward(bt_df, selected_strategy, bt_regimes)
        
        # 7. Signal Generation
        logger.info("Step 7: Generating Signal...")
        # Simple logic: if last close > sma_20, signal BUY
        last_row = df_features.iloc[-1]
        initial_signal = "BUY" if last_row['close'] > last_row.get('sma_20', last_row['close'] * 0.95) else "SELL"
        
        # 8. Adversarial Debate
        logger.info("Step 8: Running Adversarial Debate...")
        market_context = {
            'rsi': last_row.get('rsi', 50),
            'macd_hist': last_row.get('macd', 0) - last_row.get('macd_signal', 0),
            'price': last_row['close'],
            'sma_50': last_row.get('sma_50', 0),
            'sma_200': last_row.get('sma_200', 0),
            'sentiment_score': sentiment_result['avg_score'],
            'volatility_regime': 'HIGH' if self.current_regime in ['BEAR_HIGH_VOL'] else 'LOW'
        }
        
        debate_result = self.debate_engine.run_debate(market_context, initial_signal)
        final_decision = debate_result['verdict']['decision']
        
        # 9. Execution (Simulated)
        logger.info(f"Step 9: Executing Decision: {final_decision}")
        if final_decision != "HOLD":
            quantity = 10 # Demo quantity
            order = self.order_manager.create_order(symbol, final_decision, quantity)
            if order:
                exec_result = self.order_manager.submit_order(order, last_row['close'], 100000)
            else:
                exec_result = {"status": "ERROR"}
        else:
            exec_result = {"status": "NO_TRADE"}
            
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "regime": self.current_regime,
            "sentiment": sentiment_result,
            "strategy": selected_strategy.name,
            "backtest_metrics": bt_results,
            "initial_signal": initial_signal,
            "debate_verdict": debate_result['verdict'],
            "execution": exec_result
        }

def main():
    # Configuration
    config = {
        "data_source": "yahoo",
        "symbols": ["AAPL"],
        "start_date": "2022-01-01",
        "end_date": "2023-12-31"
    }
    
    titan = TITANPipeline(config)
    result = titan.run_full_cycle("AAPL", config["start_date"], config["end_date"])
    
    print("\n=== TITAN 3.0 ANALYSIS COMPLETE ===")
    print(f"Symbol: {result['symbol']}")
    print(f"Regime: {result['regime']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Debate Decision: {result['debate_verdict']['decision']} (Conf: {result['debate_verdict']['confidence']})")
    print(f"Execution: {result['execution']['status']}")
    if 'backtest_metrics' in result and 'wfo_avg_sharpe' in result['backtest_metrics']:
        print(f"WFO Sharpe: {result['backtest_metrics']['wfo_avg_sharpe']:.2f}")

if __name__ == "__main__":
    main()
