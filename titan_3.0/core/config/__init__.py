"""
TITAN 3.0 Configuration Management
Handles all configuration loading, validation, and access
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    provider: str = "yahoo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% max loss
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    var_confidence: float = 0.95
    max_drawdown_limit: float = 0.20
    daily_loss_limit_pct: float = 0.02
    kelly_fraction: float = 0.5
    volatility_target: float = 0.15


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    lookback_period: int = 252
    rebalance_frequency: str = "daily"
    min_sharpe_ratio: float = 1.0
    min_sortino_ratio: float = 1.5
    walk_forward_ratio: float = 0.7
    monte_carlo_simulations: int = 1000


@dataclass
class MLConfig:
    """Machine learning configuration"""
    regime_detection_method: str = "tda"  # tda, hmm, clustering
    n_regimes: int = 3
    rl_algorithm: str = "ppo"
    training_episodes: int = 10000
    state_window: int = 60


@dataclass
class ExecutionConfig:
    """Execution configuration"""
    paper_trading: bool = True
    broker: str = "alpaca"
    order_type: str = "limit"
    slippage_model: str = "linear"
    commission_rate: float = 0.001


@dataclass
class TITANConfig:
    """Main configuration container"""
    environment: str = "development"
    log_level: str = "INFO"
    data_sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TITANConfig':
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            return cls()  # Return defaults
        
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls._from_dict(config_data or {})
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'TITANConfig':
        """Create config from dictionary"""
        config = cls()
        
        # Parse system config
        if 'system' in data:
            config.environment = data['system'].get('environment', 'development')
            config.log_level = data['system'].get('log_level', 'INFO')
        
        # Parse data sources
        if 'data' in data:
            data_section = data['data']
            if 'sources' in data_section:
                for source_data in data_section['sources']:
                    name = source_data.get('name', 'default')
                    config.data_sources[name] = DataSourceConfig(
                        provider=name,
                        timeout=source_data.get('config', {}).get('timeout', 30),
                        retry_attempts=source_data.get('config', {}).get('retries', 3),
                        cache_enabled=data_section.get('cache', {}).get('enabled', True),
                        cache_ttl=data_section.get('cache', {}).get('ttl_seconds', 3600)
                    )
        
        # Parse risk config
        if 'risk' in data:
            risk_data = data['risk']
            config.risk = RiskConfig(
                max_drawdown_limit=risk_data.get('max_drawdown_pct', 0.20),
                daily_loss_limit_pct=risk_data.get('daily_loss_limit_pct', 0.02),
                var_confidence=risk_data.get('var_confidence_level', 0.95),
                kelly_fraction=risk_data.get('kelly_criterion', {}).get('fraction', 0.5),
                volatility_target=risk_data.get('volatility_targeting', {}).get('target_volatility', 0.15)
            )
        
        # Parse strategy config
        if 'strategies' in data:
            strat_data = data['strategies']
            if 'walk_forward' in strat_data:
                wf = strat_data['walk_forward']
                config.strategy = StrategyConfig(
                    lookback_period=wf.get('train_window_days', 252),
                    min_sharpe_ratio=wf.get('min_sharpe', 1.0),
                    monte_carlo_simulations=data.get('backtest', {}).get('monte_carlo', {}).get('simulations', 1000)
                )
        
        # Parse ML config
        if 'rl' in data:
            rl_data = data['rl']
            config.ml = MLConfig(
                rl_algorithm=rl_data.get('algorithm', 'ppo'),
                training_episodes=rl_data.get('training', {}).get('total_timesteps', 10000),
                state_window=60
            )
        
        # Parse execution config
        if 'execution' in data:
            exec_data = data['execution']
            config.execution = ExecutionConfig(
                paper_trading=exec_data.get('broker', 'mock') == 'mock',
                broker=exec_data.get('broker', 'alpaca'),
                slippage_model=exec_data.get('slippage_model', 'linear'),
                commission_rate=exec_data.get('commission_per_share', 0.001)
            )
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'environment': self.environment,
            'log_level': self.log_level,
            'data_sources': {k: vars(v) for k, v in self.data_sources.items()},
            'risk': vars(self.risk),
            'strategy': vars(self.strategy),
            'ml': vars(self.ml),
            'execution': vars(self.execution)
        }


class ConfigManager:
    """Singleton configuration manager"""
    _instance: Optional['ConfigManager'] = None
    _config: Optional[TITANConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: Optional[str] = None) -> TITANConfig:
        """Load configuration from file or environment"""
        if config_path:
            self._config = TITANConfig.from_yaml(config_path)
        else:
            # Try default locations
            for path in ['config.yaml', 'titan_config.yaml', 'config/titan_config.yaml']:
                if Path(path).exists():
                    self._config = TITANConfig.from_yaml(path)
                    break
            
            if self._config is None:
                self._config = TITANConfig()  # Use defaults
        
        # Override with environment variables
        self._apply_env_overrides()
        
        return self._config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        if self._config is None:
            return
        
        if env := os.getenv('TITAN_ENV'):
            self._config.environment = env
        if log_level := os.getenv('TITAN_LOG_LEVEL'):
            self._config.log_level = log_level
        if paper_trading := os.getenv('TITAN_PAPER_TRADING'):
            self._config.execution.paper_trading = paper_trading.lower() == 'true'
    
    def get_config(self) -> TITANConfig:
        """Get current configuration"""
        if self._config is None:
            return self.load()
        return self._config
    
    def reload(self, config_path: str) -> TITANConfig:
        """Reload configuration from file"""
        self._config = TITANConfig.from_yaml(config_path)
        self._apply_env_overrides()
        return self._config


# Global config accessor
def get_config() -> TITANConfig:
    """Get global configuration instance"""
    return ConfigManager().get_config()
