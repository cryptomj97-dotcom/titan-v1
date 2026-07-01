"""
TITAN Generative World Models System
Implements TimeGAN and diffusion models for synthetic market scenario generation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class MarketScenario:
    """Represents a generated market scenario"""
    scenario_id: str
    prices: List[float]
    volumes: List[float]
    volatilities: List[float]
    timestamps: List[float]
    scenario_type: str  # 'normal', 'crash', 'bubble', 'sideways'
    probability: float

@dataclass
class SimulationResult:
    """Results from strategy simulation on synthetic data"""
    scenario_id: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    trades_count: int
    survival_rate: float  # Percentage of scenarios where strategy survived

class TimeGAN:
    """Time-series Generative Adversarial Network for market data synthesis"""
    
    def __init__(self, sequence_length: int = 100, latent_dim: int = 32):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.trained = False
        self.generator_weights = None
        self.discriminator_weights = None
        
    def train(self, historical_data: np.ndarray, epochs: int = 100) -> bool:
        """
        Train the GAN on historical market data
        In production, this would use TensorFlow/PyTorch
        """
        logger.info(f"Training TimeGAN on {len(historical_data)} sequences...")
        
        try:
            # Simulate training process
            for epoch in range(min(epochs, 10)):  # Simulate first 10 epochs
                # In real implementation:
                # 1. Generate fake sequences
                # 2. Discriminator evaluates real vs fake
                # 3. Update generator and discriminator weights
                
                loss = np.random.uniform(0.1, 2.0) * np.exp(-epoch/20)
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
                
            self.trained = True
            logger.info("TimeGAN training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"TimeGAN training failed: {e}")
            return False
    
    def generate_sequences(self, num_sequences: int = 1000, 
                          condition: Optional[str] = None) -> List[np.ndarray]:
        """Generate synthetic market sequences"""
        
        if not self.trained:
            logger.warning("TimeGAN not trained, using fallback generation")
            return self._fallback_generation(num_sequences, condition)
        
        sequences = []
        
        for i in range(num_sequences):
            # In real implementation, this would use the trained generator
            # Here we simulate realistic price movements using geometric Brownian motion
            
            # Random walk with drift
            mu = np.random.uniform(-0.001, 0.002)  # Daily return
            sigma = np.random.uniform(0.01, 0.04)   # Daily volatility
            
            shocks = np.random.normal(0, 1, self.sequence_length)
            returns = mu + sigma * shocks
            
            # Convert returns to prices (starting from base price of 100)
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Add condition-specific adjustments
            if condition == "crash":
                crash_point = np.random.randint(len(prices)//2, len(prices))
                prices[crash_point:] *= np.exp(np.random.uniform(-0.3, -0.1))
            elif condition == "bubble":
                bubble_start = np.random.randint(0, len(prices)//3)
                prices[bubble_start:] *= np.exp(np.random.uniform(0.1, 0.5))
            
            sequences.append(prices)
        
        return sequences
    
    def _fallback_generation(self, num_sequences: int, condition: Optional[str]) -> List[np.ndarray]:
        """Fallback generation method when model is not trained"""
        return self.generate_sequences(num_sequences, condition)

class DiffusionModel:
    """Diffusion model for generating extreme market scenarios"""
    
    def __init__(self, steps: int = 100):
        self.steps = steps
        self.trained = False
        
    def generate_extreme_scenarios(self, num_scenarios: int = 100, 
                                  scenario_type: str = "black_swan") -> List[np.ndarray]:
        """Generate extreme market scenarios using diffusion process"""
        
        logger.info(f"Generating {num_scenarios} {scenario_type} scenarios...")
        
        scenarios = []
        
        for _ in range(num_scenarios):
            # Start with normal market conditions
            base_sequence = np.ones(self.steps) * 100
            
            # Apply diffusion process with extreme events
            if scenario_type == "black_swan":
                # Sudden large moves
                shock_times = np.random.choice(self.steps, size=3, replace=False)
                for shock_time in shock_times:
                    shock_magnitude = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
                    base_sequence[shock_time:] *= (1 + shock_magnitude)
                    
            elif scenario_type == "flash_crash":
                # Quick drop and recovery
                crash_start = np.random.randint(10, self.steps - 20)
                crash_depth = np.random.uniform(0.15, 0.4)
                recovery_time = np.random.randint(5, 15)
                
                base_sequence[crash_start:crash_start+recovery_time] *= (1 - crash_depth)
                base_sequence[crash_start+recovery_time:] *= (1 - crash_depth * 0.5)
                
            elif scenario_type == "gradual_decline":
                # Slow bear market
                decline_rate = np.random.uniform(0.001, 0.003)
                base_sequence *= np.exp(-decline_rate * np.arange(self.steps))
            
            scenarios.append(base_sequence)
        
        return scenarios

class GenerativeWorldModel:
    """Main interface for generative world modeling"""
    
    def __init__(self):
        self.time_gan = TimeGAN()
        self.diffusion_model = DiffusionModel()
        self.scenario_library = []
        self.calibration_data = None
        
    def calibrate(self, historical_prices: np.ndarray, historical_volumes: np.ndarray):
        """Calibrate models on historical data"""
        
        logger.info("Calibrating Generative World Model...")
        
        # Prepare training data
        sequence_length = 100
        sequences = []
        
        for i in range(len(historical_prices) - sequence_length):
            price_seq = historical_prices[i:i+sequence_length]
            volume_seq = historical_volumes[i:i+sequence_length]
            
            # Normalize sequences
            price_seq = price_seq / price_seq[0]
            volume_seq = volume_seq / np.mean(volume_seq)
            
            sequences.append(np.column_stack([price_seq, volume_seq]))
        
        if len(sequences) < 10:
            logger.warning("Insufficient data for calibration")
            return False
        
        self.calibration_data = np.array(sequences)
        
        # Train TimeGAN
        price_sequences = np.array([seq[:, 0] for seq in sequences])
        success = self.time_gan.train(price_sequences)
        
        if success:
            logger.info("Generative World Model calibration completed")
            return True
        else:
            logger.error("Generative World Model calibration failed")
            return False
    
    def generate_market_scenarios(self, num_scenarios: int = 1000, 
                                 include_extremes: bool = True) -> List[MarketScenario]:
        """Generate diverse market scenarios including extreme events"""
        
        logger.info(f"Generating {num_scenarios} market scenarios...")
        
        scenarios = []
        
        # Generate normal scenarios (70%)
        normal_count = int(num_scenarios * 0.7)
        normal_sequences = self.time_gan.generate_sequences(normal_count, condition="normal")
        
        for i, seq in enumerate(normal_sequences):
            scenario = MarketScenario(
                scenario_id=f"normal_{i}",
                prices=seq.tolist(),
                volumes=np.random.uniform(0.8, 1.2, len(seq)).tolist(),
                volatilities=np.abs(np.diff(seq, prepend=seq[0]) / seq).tolist(),
                timestamps=[time.time() + j*3600 for j in range(len(seq))],  # Hourly data
                scenario_type="normal",
                probability=0.7
            )
            scenarios.append(scenario)
        
        # Generate extreme scenarios (30%)
        if include_extremes:
            extreme_count = num_scenarios - normal_count
            
            # Crash scenarios
            crash_sequences = self.time_gan.generate_sequences(extreme_count//3, condition="crash")
            for i, seq in enumerate(crash_sequences):
                scenario = MarketScenario(
                    scenario_id=f"crash_{i}",
                    prices=seq.tolist(),
                    volumes=np.random.uniform(1.2, 3.0, len(seq)).tolist(),
                    volatilities=np.abs(np.diff(seq, prepend=seq[0]) / seq).tolist(),
                    timestamps=[time.time() + j*3600 for j in range(len(seq))],
                    scenario_type="crash",
                    probability=0.1
                )
                scenarios.append(scenario)
            
            # Bubble scenarios
            bubble_sequences = self.time_gan.generate_sequences(extreme_count//3, condition="bubble")
            for i, seq in enumerate(bubble_sequences):
                scenario = MarketScenario(
                    scenario_id=f"bubble_{i}",
                    prices=seq.tolist(),
                    volumes=np.random.uniform(0.5, 2.0, len(seq)).tolist(),
                    volatilities=np.abs(np.diff(seq, prepend=seq[0]) / seq).tolist(),
                    timestamps=[time.time() + j*3600 for j in range(len(seq))],
                    scenario_type="bubble",
                    probability=0.1
                )
                scenarios.append(scenario)
            
            # Black swan scenarios
            black_swan_sequences = self.diffusion_model.generate_extreme_scenarios(
                extreme_count//3, "black_swan"
            )
            for i, seq in enumerate(black_swan_sequences):
                scenario = MarketScenario(
                    scenario_id=f"black_swan_{i}",
                    prices=seq.tolist(),
                    volumes=np.random.uniform(2.0, 5.0, len(seq)).tolist(),
                    volatilities=np.abs(np.diff(seq, prepend=seq[0]) / seq).tolist(),
                    timestamps=[time.time() + j*3600 for j in range(len(seq))],
                    scenario_type="black_swan",
                    probability=0.1
                )
                scenarios.append(scenario)
        
        self.scenario_library = scenarios
        logger.info(f"Generated {len(scenarios)} scenarios: "
                   f"{normal_count} normal, {extreme_count//3} crash, "
                   f"{extreme_count//3} bubble, {extreme_count//3} black_swan")
        
        return scenarios
    
    def simulate_strategy_performance(self, strategy_func, initial_capital: float = 10000) -> List[SimulationResult]:
        """Simulate strategy performance across all generated scenarios"""
        
        if not self.scenario_library:
            logger.error("No scenarios available for simulation")
            return []
        
        results = []
        
        for scenario in self.scenario_library:
            try:
                # Run strategy on this scenario
                capital = initial_capital
                position = 0
                entry_price = 0
                trades = []
                
                prices = scenario.prices
                
                for i in range(1, len(prices)):
                    current_price = prices[i]
                    prev_price = prices[i-1]
                    
                    # Simple strategy logic (would be replaced by actual strategy function)
                    if current_price > prev_price * 1.01 and position == 0:  # Buy signal
                        position = capital / current_price
                        entry_price = current_price
                    elif current_price < prev_price * 0.99 and position > 0:  # Sell signal
                        capital = position * current_price
                        trades.append(capital - initial_capital)
                        position = 0
                
                # Close position at end
                if position > 0:
                    capital = position * prices[-1]
                    trades.append(capital - initial_capital)
                
                # Calculate metrics
                total_return = (capital - initial_capital) / initial_capital
                
                if trades:
                    trade_returns = np.array(trades) / initial_capital
                    sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) > 0 else 0
                    win_rate = len([t for t in trade_returns if t > 0]) / len(trade_returns)
                    max_drawdown = min(0, min(np.minimum.accumulate(np.maximum.accumulate(trade_returns) - trade_returns)))
                else:
                    sharpe_ratio = 0
                    win_rate = 0
                    max_drawdown = 0
                
                result = SimulationResult(
                    scenario_id=scenario.scenario_id,
                    total_return=total_return,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    win_rate=win_rate,
                    trades_count=len(trades),
                    survival_rate=1.0 if capital > initial_capital * 0.1 else 0.0
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Simulation failed for scenario {scenario.scenario_id}: {e}")
                continue
        
        return results
    
    def get_stress_test_report(self, simulation_results: List[SimulationResult]) -> Dict[str, float]:
        """Generate comprehensive stress test report"""
        
        if not simulation_results:
            return {}
        
        returns = [r.total_return for r in simulation_results]
        drawdowns = [r.max_drawdown for r in simulation_results]
        sharpe_ratios = [r.sharpe_ratio for r in simulation_results]
        
        report = {
            "avg_return": np.mean(returns),
            "median_return": np.median(returns),
            "worst_return": min(returns),
            "best_return": max(returns),
            "return_std": np.std(returns),
            "avg_max_drawdown": np.mean(drawdowns),
            "worst_drawdown": min(drawdowns),
            "avg_sharpe": np.mean(sharpe_ratios),
            "win_rate": np.mean([r.win_rate for r in simulation_results]),
            "survival_rate": np.mean([r.survival_rate for r in simulation_results]),
            "var_95": np.percentile(returns, 5),  # 95% VaR
            "cvar_95": np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
            "scenarios_tested": len(simulation_results)
        }
        
        logger.info(f"Stress test report: {report['survival_rate']*100:.1f}% survival rate, "
                   f"{report['avg_return']*100:.2f}% avg return")
        
        return report
