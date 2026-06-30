"""
Strategy Generation Engine for TITAN 3.0

Implements genetic programming, reinforcement learning auto-training,
and hyperparameter optimization for automated strategy discovery.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Represents a trading strategy."""
    id: str
    name: str
    logic: str
    parameters: Dict[str, Any]
    fitness_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    profit_factor: float = 0.0


class GeneticProgrammingEngine:
    """
    Genetic Programming for evolving trading strategies.
    
    Uses population-based evolution with mutation, crossover, and elitism
    to discover profitable trading rules.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_rate: float = 0.1,
        max_tree_depth: int = 5,
        min_trades: int = 100
    ):
        """
        Initialize genetic programming engine.
        
        Args:
            population_size: Number of strategies in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_rate: Fraction of best strategies to preserve
            max_tree_depth: Maximum depth of strategy trees
            min_trades: Minimum trades required for valid strategy
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.max_tree_depth = max_tree_depth
        self.min_trades = min_trades
        
        self.population: List[Strategy] = []
        self.best_strategy: Optional[Strategy] = None
        self.fitness_history: List[float] = []
    
    def initialize_population(self, feature_names: List[str]) -> None:
        """
        Initialize random population of strategies.
        
        Args:
            feature_names: List of available feature names
        """
        self.population = []
        
        for i in range(self.population_size):
            strategy = self._generate_random_strategy(feature_names, f"GP_{i}")
            self.population.append(strategy)
        
        logger.info(f"Initialized population of {len(self.population)} strategies")
    
    def _generate_random_strategy(
        self,
        feature_names: List[str],
        strategy_id: str
    ) -> Strategy:
        """Generate a random strategy."""
        # Random logic templates
        templates = [
            "RSI < {rsi_low} AND {feature} > {threshold}",
            "RSI > {rsi_high} AND {feature} < {threshold}",
            "{feature}_ma_short > {feature}_ma_long",
            "abs({feature}) > {std_mult} * {feature}_std",
            "regime == {regime_val} AND momentum > {momentum_threshold}"
        ]
        
        template = np.random.choice(templates)
        
        # Random parameters
        params = {
            'rsi_low': np.random.uniform(20, 35),
            'rsi_high': np.random.uniform(65, 80),
            'threshold': np.random.uniform(-0.5, 0.5),
            'std_mult': np.random.uniform(1.5, 3.0),
            'momentum_threshold': np.random.uniform(-0.1, 0.1),
            'regime_val': np.random.randint(0, 3)
        }
        
        # Select random feature
        feature = np.random.choice(feature_names) if feature_names else 'returns'
        logic = template.format(feature=feature, **params)
        
        return Strategy(
            id=strategy_id,
            name=f"Random_{strategy_id}",
            logic=logic,
            parameters=params
        )
    
    def evolve(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        fitness_func: callable
    ) -> Strategy:
        """
        Run genetic algorithm evolution.
        
        Args:
            features: Feature matrix
            returns: Return series
            fitness_func: Function to evaluate strategy fitness
            
        Returns:
            Best evolved strategy
        """
        if not self.population:
            raise ValueError("Population not initialized. Call initialize_population() first.")
        
        logger.info(f"Starting evolution for {self.generations} generations")
        
        for gen in range(self.generations):
            # Evaluate fitness
            for strategy in self.population:
                try:
                    fitness = fitness_func(strategy, features, returns)
                    strategy.fitness_score = fitness
                except Exception as e:
                    logger.warning(f"Error evaluating strategy {strategy.id}: {e}")
                    strategy.fitness_score = -np.inf
            
            # Track best strategy
            self.population.sort(key=lambda s: s.fitness_score, reverse=True)
            best_gen_fitness = self.population[0].fitness_score
            self.fitness_history.append(best_gen_fitness)
            
            if self.best_strategy is None or best_gen_fitness > self.best_strategy.fitness_score:
                self.best_strategy = self.population[0]
            
            logger.info(f"Generation {gen+1}/{self.generations}: Best fitness = {best_gen_fitness:.4f}")
            
            # Create next generation
            self.population = self._create_next_generation()
        
        logger.info(f"Evolution complete. Best fitness: {self.best_strategy.fitness_score:.4f}")
        return self.best_strategy
    
    def _create_next_generation(self) -> List[Strategy]:
        """Create next generation through selection, crossover, and mutation."""
        n_elite = int(self.population_size * self.elitism_rate)
        new_population = []
        
        # Elitism: keep best strategies
        new_population.extend(self.population[:n_elite])
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Selection (tournament)
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population
    
    def _tournament_selection(self, k: int = 5) -> Strategy:
        """Tournament selection."""
        contestants = np.random.choice(self.population, size=k, replace=False)
        return max(contestants, key=lambda s: s.fitness_score)
    
    def _crossover(
        self,
        parent1: Strategy,
        parent2: Strategy
    ) -> Tuple[Strategy, Strategy]:
        """Perform crossover between two parents."""
        # Simple parameter crossover
        params1 = parent1.parameters.copy()
        params2 = parent2.parameters.copy()
        
        # Single-point crossover on parameter keys
        keys = list(params1.keys())
        if len(keys) > 1:
            point = np.random.randint(1, len(keys))
            
            for key in keys[point:]:
                params1[key], params2[key] = params2[key], params1[key]
        
        child1 = Strategy(
            id=f"child1_{parent1.id}_{parent2.id}",
            name=f"Crossover_{parent1.id[:4]}_{parent2.id[:4]}",
            logic=parent1.logic,  # Simplified: could combine logic trees
            parameters=params1
        )
        
        child2 = Strategy(
            id=f"child2_{parent1.id}_{parent2.id}",
            name=f"Crossover_{parent2.id[:4]}_{parent1.id[:4]}",
            logic=parent2.logic,
            parameters=params2
        )
        
        return child1, child2
    
    def _mutate(self, strategy: Strategy) -> Strategy:
        """Mutate strategy parameters."""
        mutated_params = strategy.parameters.copy()
        
        # Randomly mutate one parameter
        keys = list(mutated_params.keys())
        key_to_mutate = np.random.choice(keys)
        
        current_value = mutated_params[key_to_mutate]
        
        # Add Gaussian noise
        noise = np.random.normal(0, abs(current_value) * 0.1)
        mutated_params[key_to_mutate] = current_value + noise
        
        # Ensure bounds
        if 'rsi' in key_to_mutate:
            mutated_params[key_to_mutate] = np.clip(mutated_params[key_to_mutate], 0, 100)
        
        mutated_strategy = Strategy(
            id=f"mutant_{strategy.id}",
            name=f"Mutant_{strategy.id[:4]}",
            logic=strategy.logic,
            parameters=mutated_params
        )
        
        return mutated_strategy


class RLTrainer:
    """
    Reinforcement Learning trainer for trading strategies.
    
    Implements PPO (Proximal Policy Optimization) for training
    agents to trade optimally.
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64
    ):
        """
        Initialize RL trainer.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_steps: Number of steps per rollout
            n_epochs: Number of PPO epochs
            batch_size: Mini-batch size
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.model = None
        self.training_history = []
    
    def train(
        self,
        env,
        total_timesteps: int = 100000,
        checkpoint_freq: int = 10000
    ) -> Dict:
        """
        Train PPO agent on trading environment.
        
        Args:
            env: Trading environment (gym-compatible)
            total_timesteps: Total training timesteps
            checkpoint_freq: Frequency of checkpoints
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback
        except ImportError:
            logger.error("stable-baselines3 not installed. Install with: pip install stable-baselines3")
            raise
        
        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            clip_range=self.clip_epsilon,
            ent_coef=self.entropy_coef,
            vf_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            verbose=1
        )
        
        # Train
        self.model.learn(total_timesteps=total_timesteps)
        
        logger.info("Training complete")
        
        return {
            'model': self.model,
            'total_timesteps': total_timesteps,
            'final_reward': self._evaluate_model(env)
        }
    
    def _evaluate_model(self, env) -> float:
        """Evaluate trained model."""
        obs = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        return total_reward
    
    def save_model(self, path: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model trained yet")
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable-baselines3 not installed")
            raise
        
        self.model = PPO.load(path)
        logger.info(f"Model loaded from {path}")


class HyperparameterOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    
    Uses Gaussian Processes to efficiently search hyperparameter space.
    """
    
    def __init__(
        self,
        param_space: Dict[str, Tuple],
        n_iterations: int = 50,
        acquisition_function: str = 'ei'
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            param_space: Dictionary of parameter names and (min, max) tuples
            n_iterations: Number of optimization iterations
            acquisition_function: Acquisition function ('ei', 'ucb', 'poi')
        """
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.acquisition_function = acquisition_function
        
        self.best_params = None
        self.best_score = -np.inf
        self.history = []
    
    def optimize(
        self,
        objective_func: callable,
        n_initial_points: int = 10
    ) -> Dict:
        """
        Optimize hyperparameters.
        
        Args:
            objective_func: Function to maximize (takes params dict, returns score)
            n_initial_points: Number of random initial points
            
        Returns:
            Best parameters found
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            logger.error("scikit-optimize not installed. Install with: pip install scikit-optimize")
            logger.info("Falling back to random search")
            return self._random_search(objective_func, n_initial_points + self.n_iterations)
        
        # Define search space
        dimensions = []
        param_names = []
        
        for param_name, (min_val, max_val, param_type) in self.param_space.items():
            if param_type == 'real':
                dimensions.append(Real(min_val, max_val))
            elif param_type == 'int':
                dimensions.append(Integer(int(min_val), int(max_val)))
            else:
                dimensions.append(Categorical([min_val, max_val]))
            param_names.append(param_name)
        
        # Objective wrapper
        def wrapped_objective(params):
            param_dict = {name: val for name, val in zip(param_names, params)}
            score = objective_func(param_dict)
            self.history.append({'params': param_dict, 'score': score})
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict
            
            return -score  # Minimize negative (maximize original)
        
        # Run optimization
        result = gp_minimize(
            func=wrapped_objective,
            dimensions=dimensions,
            n_calls=self.n_iterations,
            n_random_starts=n_initial_points,
            acq_func=self.acquisition_function,
            random_state=42
        )
        
        logger.info(f"Optimization complete. Best score: {-result.fun:.4f}")
        logger.info(f"Best params: {result.x}")
        
        return self.best_params
    
    def _random_search(self, objective_func: callable, n_points: int) -> Dict:
        """Fallback random search."""
        logger.info(f"Running random search with {n_points} points")
        
        param_names = list(self.param_space.keys())
        
        for i in range(n_points):
            params = {}
            for name, (min_val, max_val, param_type) in self.param_space.items():
                if param_type == 'real':
                    params[name] = np.random.uniform(min_val, max_val)
                elif param_type == 'int':
                    params[name] = np.random.randint(int(min_val), int(max_val) + 1)
                else:
                    params[name] = np.random.choice([min_val, max_val])
            
            score = objective_func(params)
            self.history.append({'params': params, 'score': score})
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        logger.info(f"Random search complete. Best score: {self.best_score:.4f}")
        return self.best_params


def create_strategy_template(
    template_type: str,
    asset_class: str = 'equity'
) -> Strategy:
    """
    Create a pre-defined strategy template.
    
    Args:
        template_type: Type of strategy ('mean_reversion', 'trend_following', 
                       'momentum', 'breakout', 'pairs_trading')
        asset_class: Asset class ('equity', 'forex', 'crypto', 'futures')
    
    Returns:
        Strategy object with template logic
    """
    templates = {
        'mean_reversion': {
            'logic': 'RSI < 30 AND price < lower_bollinger',
            'parameters': {
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 2.0,
                'entry_threshold': 0.5,
                'exit_threshold': 0.0
            }
        },
        'trend_following': {
            'logic': 'MA_short > MA_long AND ADX > 25',
            'parameters': {
                'ma_short': 10,
                'ma_long': 50,
                'adx_period': 14,
                'adx_threshold': 25,
                'stop_loss': 0.02,
                'take_profit': 0.04
            }
        },
        'momentum': {
            'logic': 'returns_12m > returns_1m AND volume > avg_volume * 1.5',
            'parameters': {
                'lookback_short': 1,
                'lookback_long': 12,
                'volume_multiplier': 1.5,
                'rebalance_period': 21
            }
        },
        'breakout': {
            'logic': 'price > highest(high, lookback) AND volume > avg_volume',
            'parameters': {
                'lookback': 20,
                'volume_threshold': 1.0,
                'stop_loss_atr_mult': 2.0,
                'confirmation_periods': 2
            }
        },
        'pairs_trading': {
            'logic': 'abs(z_score) > entry_threshold AND exit when z_score < exit_threshold',
            'parameters': {
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'lookback_window': 60,
                'hedge_ratio_window': 60
            }
        }
    }
    
    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}. Available: {list(templates.keys())}")
    
    template = templates[template_type]
    
    return Strategy(
        id=f"{template_type}_{asset_class}",
        name=f"{template_type.replace('_', ' ').title()} ({asset_class.title()})",
        logic=template['logic'],
        parameters=template['parameters']
    )
