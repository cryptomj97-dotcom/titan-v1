"""
TITAN 3.0 - Genetic Algorithm Strategy Generator
Automatically discovers and evolves trading strategies using genetic algorithms.
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from copy import deepcopy

from ..core.strategy_base import BaseStrategy, TradeSignal, SignalType, Timeframe


@dataclass
class Gene:
    """Represents a single gene in a strategy chromosome."""
    name: str
    value: Any
    min_value: Any = None
    max_value: Any = None
    mutation_rate: float = 0.1
    gene_type: str = 'float'  # float, int, bool, categorical
    categorical_options: List[Any] = field(default_factory=list)
    
    def mutate(self) -> 'Gene':
        """Mutate the gene value."""
        if random.random() > self.mutation_rate:
            return deepcopy(self)
        
        new_gene = deepcopy(self)
        
        if self.gene_type == 'float':
            if self.min_value is not None and self.max_value is not None:
                # Gaussian mutation within bounds
                mutation = np.random.normal(0, (self.max_value - self.min_value) * 0.1)
                new_value = self.value + mutation
                new_gene.value = np.clip(new_value, self.min_value, self.max_value)
            else:
                new_gene.value = self.value * (1 + np.random.normal(0, 0.2))
                
        elif self.gene_type == 'int':
            if self.min_value is not None and self.max_value is not None:
                mutation = np.random.randint(-2, 3)
                new_value = int(self.value) + mutation
                new_gene.value = int(np.clip(new_value, self.min_value, self.max_value))
                
        elif self.gene_type == 'bool':
            new_gene.value = not self.value
            
        elif self.gene_type == 'categorical':
            if self.categorical_options:
                new_gene.value = random.choice(self.categorical_options)
        
        return new_gene


@dataclass
class Chromosome:
    """Represents a complete strategy configuration (genome)."""
    genes: Dict[str, Gene]
    fitness: float = 0.0
    generation: int = 0
    id: str = field(default_factory=lambda: f"chr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}")
    
    def to_parameters(self) -> Dict[str, Any]:
        """Convert chromosome to strategy parameters dictionary."""
        return {name: gene.value for name, gene in self.genes.items()}
    
    def crossover(self, other: 'Chromosome') -> Tuple['Chromosome', 'Chromosome']:
        """Perform uniform crossover with another chromosome."""
        child1_genes = {}
        child2_genes = {}
        
        for gene_name in self.genes.keys():
            if random.random() < 0.5:
                child1_genes[gene_name] = deepcopy(self.genes[gene_name])
                child2_genes[gene_name] = deepcopy(other.genes[gene_name])
            else:
                child1_genes[gene_name] = deepcopy(other.genes[gene_name])
                child2_genes[gene_name] = deepcopy(self.genes[gene_name])
        
        child1 = Chromosome(genes=child1_genes, generation=max(self.generation, other.generation) + 1)
        child2 = Chromosome(genes=child2_genes, generation=max(self.generation, other.generation) + 1)
        
        return child1, child2
    
    def mutate_all(self) -> 'Chosome':
        """Mutate all genes in the chromosome."""
        mutated_genes = {name: gene.mutate() for name, gene in self.genes.items()}
        return Chromosome(genes=mutated_genes, generation=self.generation + 1)


class EvolvableStrategy(BaseStrategy):
    """
    A strategy whose parameters are defined by a chromosome.
    Can be evolved through genetic algorithms.
    """
    
    def __init__(self,
                 chromosome: Chromosome,
                 strategy_id: str = None,
                 base_strategy_type: str = 'momentum',
                 timeframe: Timeframe = Timeframe.HOUR_1,
                 symbols: List[str] = None):
        
        self.chromosome = chromosome
        self.base_strategy_type = base_strategy_type
        params = chromosome.to_parameters()
        
        strategy_id = strategy_id or f"{base_strategy_type}_{chromosome.id}"
        
        super().__init__(
            strategy_id=strategy_id,
            name=f"Evolvable {base_strategy_type.title()} - {chromosome.id}",
            description=f"Genetically evolved {base_strategy_type} strategy",
            timeframe=timeframe,
            symbols=symbols,
            parameters=params
        )
        
        self._internal_strategy = self._build_internal_strategy()
    
    def _build_internal_strategy(self) -> BaseStrategy:
        """Build the internal strategy based on chromosome parameters."""
        from ..core.strategy_base import MomentumStrategy, MeanReversionStrategy
        
        if self.base_strategy_type == 'momentum':
            return MomentumStrategy(
                strategy_id=self.strategy_id,
                timeframe=self.timeframe,
                symbols=self.symbols,
                parameters=self.parameters
            )
        elif self.base_strategy_type == 'mean_reversion':
            return MeanReversionStrategy(
                strategy_id=self.strategy_id,
                timeframe=self.timeframe,
                symbols=self.symbols,
                parameters=self.parameters
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.base_strategy_type}")
    
    def generate_signal(self,
                       data: pd.DataFrame,
                       features: Dict[str, pd.Series],
                       regime: str,
                       current_position=None) -> Optional[TradeSignal]:
        """Delegate signal generation to internal strategy."""
        if not self.is_active or self._internal_strategy is None:
            return None
        
        # Update internal strategy parameters if chromosome changed
        if self._internal_strategy.parameters != self.parameters:
            self._internal_strategy.parameters = self.parameters
        
        return self._internal_strategy.generate_signal(data, features, regime, current_position)
    
    def calculate_position_size(self,
                               signal: TradeSignal,
                               account_balance: float,
                               risk_per_trade: float = 0.02,
                               volatility: float = 0.0) -> float:
        """Delegate position sizing to internal strategy."""
        if self._internal_strategy is None:
            return 0.0
        
        return self._internal_strategy.calculate_position_size(
            signal, account_balance, risk_per_trade, volatility
        )


class GeneticStrategyGenerator:
    """
    Genetic algorithm engine for evolving trading strategies.
    """
    
    def __init__(self,
                 population_size: int = 50,
                 elitism_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.2,
                 generations: int = 100,
                 base_strategy_type: str = 'momentum',
                 timeframe: Timeframe = Timeframe.HOUR_1,
                 symbols: List[str] = None):
        
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.base_strategy_type = base_strategy_type
        self.timeframe = timeframe
        self.symbols = symbols or []
        
        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None
        self.fitness_history: List[float] = []
        self.generation = 0
        
        self._define_gene_space()
    
    def _define_gene_space(self):
        """Define the gene space based on strategy type."""
        if self.base_strategy_type == 'momentum':
            self.gene_definitions = {
                'rsi_period': Gene('rsi_period', 14, 5, 30, gene_type='int'),
                'rsi_overbought': Gene('rsi_overbought', 70, 60, 80, gene_type='int'),
                'rsi_oversold': Gene('rsi_oversold', 30, 20, 40, gene_type='int'),
                'macd_fast': Gene('macd_fast', 12, 5, 20, gene_type='int'),
                'macd_slow': Gene('macd_slow', 26, 15, 40, gene_type='int'),
                'macd_signal': Gene('macd_signal', 9, 5, 15, gene_type='int'),
                'min_volume_ratio': Gene('min_volume_ratio', 1.5, 1.0, 3.0, gene_type='float'),
            }
        elif self.base_strategy_type == 'mean_reversion':
            self.gene_definitions = {
                'bb_period': Gene('bb_period', 20, 10, 50, gene_type='int'),
                'bb_std': Gene('bb_std', 2.0, 1.5, 3.0, gene_type='float'),
                'zscore_threshold': Gene('zscore_threshold', 2.0, 1.5, 3.0, gene_type='float'),
                'exit_zscore': Gene('exit_zscore', 0.5, 0.0, 1.0, gene_type='float'),
            }
        else:
            self.gene_definitions = {}
    
    def initialize_population(self) -> List[Chromosome]:
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            genes = {}
            for name, template_gene in self.gene_definitions.items():
                gene = deepcopy(template_gene)
                
                if gene.gene_type == 'float':
                    if gene.min_value is not None and gene.max_value is not None:
                        gene.value = random.uniform(gene.min_value, gene.max_value)
                    else:
                        gene.value = random.gauss(gene.value, gene.value * 0.2)
                        
                elif gene.gene_type == 'int':
                    if gene.min_value is not None and gene.max_value is not None:
                        gene.value = random.randint(gene.min_value, gene.max_value)
                    else:
                        gene.value = int(random.gauss(gene.value, gene.value * 0.2))
                        
                elif gene.gene_type == 'bool':
                    gene.value = random.choice([True, False])
                    
                elif gene.gene_type == 'categorical':
                    if gene.categorical_options:
                        gene.value = random.choice(gene.categorical_options)
                
                genes[name] = gene
            
            chromosome = Chromosome(genes=genes)
            self.population.append(chromosome)
        
        return self.population
    
    def evaluate_fitness(self, 
                        chromosome: Chromosome,
                        historical_data: pd.DataFrame,
                        features: Dict[str, pd.Series],
                        regimes: pd.Series,
                        initial_capital: float = 100000.0) -> float:
        """
        Evaluate fitness of a chromosome using backtesting.
        Returns Sharpe ratio as fitness metric.
        """
        strategy = EvolvableStrategy(
            chromosome=chromosome,
            base_strategy_type=self.base_strategy_type,
            timeframe=self.timeframe,
            symbols=self.symbols
        )
        
        strategy.activate()
        
        # Simple backtest simulation
        capital = initial_capital
        position = None
        returns = []
        trades = []
        
        for i in range(len(historical_data) - 1):
            idx = historical_data.index[i]
            
            # Get data slice up to current point
            data_slice = historical_data.iloc[:i+1]
            features_slice = {k: v.iloc[:i+1] if hasattr(v, 'iloc') else v for k, v in features.items()}
            regime = regimes.iloc[i] if hasattr(regimes, 'iloc') else regimes[i]
            
            # Generate signal
            signal = strategy.generate_signal(data_slice, features_slice, regime, position)
            
            if signal is None:
                continue
            
            # Execute signal
            if signal.signal_type == SignalType.LONG and position is None:
                size = strategy.calculate_position_size(signal, capital, 0.02, 0.02)
                if size > 0:
                    position = {
                        'entry_price': signal.price,
                        'size': size,
                        'side': 'long'
                    }
            
            elif signal.signal_type == SignalType.SHORT and position is None:
                size = strategy.calculate_position_size(signal, capital, 0.02, 0.02)
                if size > 0:
                    position = {
                        'entry_price': signal.price,
                        'size': size,
                        'side': 'short'
                    }
            
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT] and position is not None:
                if position['side'] == 'long':
                    pnl = (signal.price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - signal.price) * position['size']
                
                capital += pnl
                returns.append(pnl / initial_capital)
                trades.append({'entry': position['entry_price'], 'exit': signal.price, 'pnl': pnl})
                position = None
            
            # Update price for unrealized PnL tracking
            current_price = historical_data['close'].iloc[i]
            if position:
                if position['side'] == 'long':
                    unrealized = (current_price - position['entry_price']) * position['size']
                else:
                    unrealized = (position['entry_price'] - current_price) * position['size']
        
        # Calculate fitness (Sharpe ratio)
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            sharpe = 0.0
        else:
            sharpe = (mean_return / std_return) * np.sqrt(252)  # Annualized
        
        # Add penalty for low number of trades
        trade_penalty = max(0, (10 - len(trades)) * 0.1)
        
        # Add penalty for high drawdown (simplified)
        cumulative_returns = np.cumsum(returns_array)
        if len(cumulative_returns) > 0:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            drawdown_penalty = max_drawdown * 2
        else:
            drawdown_penalty = 0
        
        fitness = sharpe - trade_penalty - drawdown_penalty
        
        return fitness
    
    def select_parents(self, fitness_scores: List[float]) -> Tuple[Chromosome, Chromosome]:
        """Select two parents using tournament selection."""
        tournament_size = 5
        
        def tournament_select():
            indices = random.sample(range(len(self.population)), min(tournament_size, len(self.population)))
            best_idx = max(indices, key=lambda i: fitness_scores[i])
            return self.population[best_idx]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        while parent2.id == parent1.id and len(self.population) > 1:
            parent2 = tournament_select()
        
        return parent1, parent2
    
    def evolve_generation(self,
                         historical_data: pd.DataFrame,
                         features: Dict[str, pd.Series],
                         regimes: pd.Series) -> List[Chromosome]:
        """Evolve one generation."""
        # Evaluate fitness for all chromosomes
        fitness_scores = []
        for chromosome in self.population:
            fitness = self.evaluate_fitness(chromosome, historical_data, features, regimes)
            chromosome.fitness = fitness
            fitness_scores.append(fitness)
        
        # Track best chromosome
        best_idx = np.argmax(fitness_scores)
        if self.best_chromosome is None or fitness_scores[best_idx] > self.best_chromosome.fitness:
            self.best_chromosome = deepcopy(self.population[best_idx])
        
        self.fitness_history.append(max(fitness_scores))
        self.generation += 1
        
        # Create new population
        new_population = []
        
        # Elitism: keep top performers
        elite_count = int(self.population_size * self.elitism_rate)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        for i in range(elite_count):
            new_population.append(deepcopy(self.population[sorted_indices[i]]))
        
        # Crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1, parent2 = self.select_parents(fitness_scores)
                child1, child2 = parent1.crossover(parent2)
                
                if random.random() < self.mutation_rate:
                    child1 = child1.mutate_all()
                if random.random() < self.mutation_rate:
                    child2 = child2.mutate_all()
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                # Just mutate existing
                selected_idx = random.choice(range(len(self.population)))
                mutant = self.population[selected_idx].mutate_all()
                new_population.append(mutant)
        
        self.population = new_population[:self.population_size]
        return self.population
    
    def run_evolution(self,
                     historical_data: pd.DataFrame,
                     features: Dict[str, pd.Series],
                     regimes: pd.Series,
                     callback=None) -> Chromosome:
        """
        Run full evolution process.
        
        Args:
            historical_data: Historical OHLCV data
            features: Pre-computed features
            regimes: Market regime series
            callback: Optional callback function called each generation
            
        Returns:
            Best chromosome found
        """
        print(f"Starting genetic evolution for {self.base_strategy_type} strategy...")
        print(f"Population size: {self.population_size}, Generations: {self.generations}")
        
        self.initialize_population()
        
        for gen in range(self.generations):
            self.evolve_generation(historical_data, features, regimes)
            
            if callback:
                callback(self.generation, self.best_chromosome, self.fitness_history[-1])
            
            if self.generation % 10 == 0:
                print(f"Generation {self.generation}: Best fitness = {self.fitness_history[-1]:.4f}")
        
        print(f"\nEvolution complete!")
        print(f"Best fitness: {self.best_chromosome.fitness:.4f}")
        print(f"Best parameters: {self.best_chromosome.to_parameters()}")
        
        return self.best_chromosome
    
    def get_best_strategy(self) -> EvolvableStrategy:
        """Get the best evolved strategy."""
        if self.best_chromosome is None:
            raise ValueError("No evolution has been performed yet.")
        
        return EvolvableStrategy(
            chromosome=self.best_chromosome,
            base_strategy_type=self.base_strategy_type,
            timeframe=self.timeframe,
            symbols=self.symbols
        )
