"""Deployment manager for live trading strategies."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Strategy deployment status."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DeploymentConfig:
    """Configuration for strategy deployment."""
    strategy_id: str
    execution_algo: str  # 'vwap', 'almgren_chriss', 'market'
    max_position_size: float  # Maximum position size in dollars
    max_daily_trades: int  # Maximum trades per day
    risk_limits: Dict[str, float]  # Risk limit configuration
    trading_hours: tuple  # (start_hour, end_hour)
    allowed_assets: List[str]  # List of allowed assets
    auto_rebalance: bool = False
    rebalance_frequency: str = "daily"  # 'daily', 'weekly'


@dataclass
class ActiveDeployment:
    """Active strategy deployment."""
    config: DeploymentConfig
    status: DeploymentStatus
    deployed_at: datetime
    deployed_by: str
    total_trades: int = 0
    daily_trades: int = 0
    last_trade_at: Optional[datetime] = None
    last_error: Optional[str] = None
    pnl_today: float = 0.0
    pnl_total: float = 0.0


class DeploymentManager:
    """
    Manages live deployment of approved strategies.
    
    Handles execution algorithm selection, risk limit enforcement,
    trade scheduling, and automatic kill switches.
    """
    
    def __init__(self, risk_manager=None, execution_engines=None):
        """
        Initialize deployment manager.
        
        Args:
            risk_manager: RiskManager instance for limit enforcement
            execution_engines: Dictionary of execution algorithm instances
        """
        self.risk_manager = risk_manager
        self.execution_engines = execution_engines or {}
        
        self.active_deployments: Dict[str, ActiveDeployment] = {}
        self.deployment_history: List[Dict] = []
        
        self.market_open_hour = 9  # 9 AM
        self.market_close_hour = 16  # 4 PM
        
        logger.info("Deployment manager initialized")
    
    def deploy_strategy(self, 
                       strategy_id: str,
                       deployed_by: str,
                       execution_algo: str = 'vwap',
                       max_position_size: float = 100000,
                       max_daily_trades: int = 50,
                       risk_limits: Optional[Dict] = None,
                       allowed_assets: Optional[List[str]] = None) -> Dict:
        """
        Deploy an approved strategy.
        
        Args:
            strategy_id: Strategy to deploy
            deployed_by: User deploying the strategy
            execution_algo: Execution algorithm to use
            max_position_size: Maximum position size
            max_daily_trades: Maximum trades per day
            risk_limits: Custom risk limits
            allowed_assets: List of allowed assets
            
        Returns:
            Deployment result
        """
        # Check if strategy exists and is approved
        # (In production, would check with ApprovalGateway)
        
        # Create deployment config
        config = DeploymentConfig(
            strategy_id=strategy_id,
            execution_algo=execution_algo,
            max_position_size=max_position_size,
            max_daily_trades=max_daily_trades,
            risk_limits=risk_limits or {},
            trading_hours=(self.market_open_hour, self.market_close_hour),
            allowed_assets=allowed_assets or []
        )
        
        # Create active deployment
        deployment = ActiveDeployment(
            config=config,
            status=DeploymentStatus.ACTIVE,
            deployed_at=datetime.now(),
            deployed_by=deployed_by
        )
        
        self.active_deployments[strategy_id] = deployment
        
        # Record in history
        self.deployment_history.append({
            'action': 'deployed',
            'strategy_id': strategy_id,
            'timestamp': deployment.deployed_at.isoformat(),
            'deployed_by': deployed_by,
            'execution_algo': execution_algo
        })
        
        logger.info(f"Strategy deployed: {strategy_id} by {deployed_by}")
        
        return {
            'success': True,
            'strategy_id': strategy_id,
            'status': 'active',
            'deployed_at': deployment.deployed_at.isoformat()
        }
    
    def execute_trade(self, strategy_id: str, symbol: str, side: str,
                     quantity: float, price: float) -> Dict:
        """
        Execute a trade for a deployed strategy.
        
        Args:
            strategy_id: Strategy executing the trade
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Quantity to trade
            price: Current price
            
        Returns:
            Trade execution result
        """
        # Check if strategy is deployed and active
        if strategy_id not in self.active_deployments:
            return {'success': False, 'error': 'Strategy not deployed'}
        
        deployment = self.active_deployments[strategy_id]
        
        if deployment.status != DeploymentStatus.ACTIVE:
            return {
                'success': False,
                'error': f'Strategy not active (status: {deployment.status.value})'
            }
        
        # Check trading hours
        current_hour = datetime.now().hour
        start_hour, end_hour = deployment.config.trading_hours
        if not (start_hour <= current_hour < end_hour):
            return {
                'success': False,
                'error': 'Outside trading hours'
            }
        
        # Check daily trade limit
        if deployment.daily_trades >= deployment.config.max_daily_trades:
            return {
                'success': False,
                'error': f'Daily trade limit reached ({deployment.config.max_daily_trades})'
            }
        
        # Check allowed assets
        if deployment.config.allowed_assets and symbol not in deployment.config.allowed_assets:
            return {
                'success': False,
                'error': f'Asset {symbol} not in allowed list'
            }
        
        # Check position size
        notional = quantity * price
        if notional > deployment.config.max_position_size:
            return {
                'success': False,
                'error': f'Position size ${notional:,.2f} exceeds limit ${deployment.config.max_position_size:,.2f}'
            }
        
        # Select execution algorithm
        exec_algo = deployment.config.execution_algo
        if exec_algo == 'vwap' and 'vwap' in self.execution_engines:
            executor = self.execution_engines['vwap']
            # Generate schedule and execute first slice
            schedule = executor.generate_schedule(quantity)
            if schedule:
                result = executor.execute_slice(schedule[0], price, quantity * 100)
            else:
                result = {'filled_qty': 0, 'price': price, 'status': 'no_schedule'}
        elif exec_algo == 'almgren_chriss' and 'almgren_chriss' in self.execution_engines:
            executor = self.execution_engines['almgren_chriss']
            trajectory = executor.compute_optimal_trajectory(quantity, price)
            if trajectory:
                result = executor.execute_trade(trajectory[0], price)
            else:
                result = {'filled_qty': 0, 'price': price, 'status': 'no_trajectory'}
        else:
            # Simple market execution
            result = {
                'filled_qty': quantity,
                'price': price,
                'status': 'market_fill'
            }
        
        # Update deployment stats
        if result.get('filled_qty', 0) > 0:
            deployment.total_trades += 1
            deployment.daily_trades += 1
            deployment.last_trade_at = datetime.now()
            
            # Update P&L (simplified)
            if side.lower() == 'sell':
                deployment.pnl_today += result.get('filled_qty', 0) * (price - result.get('price', price))
        
        logger.debug(f"Trade executed for {strategy_id}: {side} {quantity} {symbol} @ {price}")
        
        return {
            'success': True,
            'strategy_id': strategy_id,
            'symbol': symbol,
            'side': side,
            'filled_qty': result.get('filled_qty', 0),
            'price': result.get('price', price),
            'execution_algo': exec_algo
        }
    
    def pause_deployment(self, strategy_id: str, reason: str) -> Dict:
        """Pause an active deployment."""
        if strategy_id not in self.active_deployments:
            return {'success': False, 'error': 'Strategy not deployed'}
        
        deployment = self.active_deployments[strategy_id]
        deployment.status = DeploymentStatus.PAUSED
        deployment.last_error = reason
        
        self.deployment_history.append({
            'action': 'paused',
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        })
        
        logger.warning(f"Strategy paused: {strategy_id} - {reason}")
        
        return {'success': True, 'strategy_id': strategy_id, 'status': 'paused'}
    
    def resume_deployment(self, strategy_id: str) -> Dict:
        """Resume a paused deployment."""
        if strategy_id not in self.active_deployments:
            return {'success': False, 'error': 'Strategy not deployed'}
        
        deployment = self.active_deployments[strategy_id]
        
        if deployment.status != DeploymentStatus.PAUSED:
            return {'success': False, 'error': f'Strategy not paused (status: {deployment.status.value})'}
        
        deployment.status = DeploymentStatus.ACTIVE
        deployment.last_error = None
        
        self.deployment_history.append({
            'action': 'resumed',
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Strategy resumed: {strategy_id}")
        
        return {'success': True, 'strategy_id': strategy_id, 'status': 'active'}
    
    def stop_deployment(self, strategy_id: str, reason: str) -> Dict:
        """Stop a deployment permanently."""
        if strategy_id not in self.active_deployments:
            return {'success': False, 'error': 'Strategy not deployed'}
        
        deployment = self.active_deployments[strategy_id]
        deployment.status = DeploymentStatus.STOPPED
        deployment.last_error = reason
        
        # Move to history
        self.deployment_history.append({
            'action': 'stopped',
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'total_trades': deployment.total_trades,
            'pnl_total': deployment.pnl_total
        })
        
        del self.active_deployments[strategy_id]
        
        logger.warning(f"Strategy stopped: {strategy_id} - {reason}")
        
        return {'success': True, 'strategy_id': strategy_id, 'status': 'stopped'}
    
    def reset_daily_counters(self):
        """Reset daily trade counters for all deployments."""
        for deployment in self.active_deployments.values():
            deployment.daily_trades = 0
            deployment.pnl_today = 0.0
        
        logger.info("Daily counters reset for all deployments")
    
    def get_deployment_status(self, strategy_id: str) -> Optional[Dict]:
        """Get status of a specific deployment."""
        if strategy_id not in self.active_deployments:
            return None
        
        deployment = self.active_deployments[strategy_id]
        
        return {
            'strategy_id': strategy_id,
            'status': deployment.status.value,
            'execution_algo': deployment.config.execution_algo,
            'deployed_at': deployment.deployed_at.isoformat(),
            'deployed_by': deployment.deployed_by,
            'total_trades': deployment.total_trades,
            'daily_trades': deployment.daily_trades,
            'max_daily_trades': deployment.config.max_daily_trades,
            'pnl_today': deployment.pnl_today,
            'pnl_total': deployment.pnl_total,
            'last_trade_at': deployment.last_trade_at.isoformat() if deployment.last_trade_at else None,
            'last_error': deployment.last_error
        }
    
    def get_all_deployments(self) -> List[Dict]:
        """Get status of all active deployments."""
        return [
            self.get_deployment_status(sid)
            for sid in self.active_deployments.keys()
        ]
    
    def check_kill_switches(self, risk_metrics: Dict) -> List[Dict]:
        """
        Check if any kill switches should be triggered.
        
        Args:
            risk_metrics: Current risk metrics from RiskManager
            
        Returns:
            List of actions taken
        """
        actions = []
        
        # Check for risk limit breaches
        if risk_metrics.get('kill_switch_active'):
            # Stop all deployments
            for strategy_id in list(self.active_deployments.keys()):
                result = self.stop_deployment(
                    strategy_id,
                    f"Risk kill switch triggered: {risk_metrics.get('kill_switch_reason', 'Unknown')}"
                )
                if result['success']:
                    actions.append({
                        'action': 'stop',
                        'strategy_id': strategy_id,
                        'reason': 'risk_kill_switch'
                    })
        
        # Check individual deployment metrics
        for strategy_id, deployment in list(self.active_deployments.items()):
            # Check daily loss limit
            if deployment.pnl_today < -deployment.config.risk_limits.get('daily_max_loss', float('inf')):
                result = self.pause_deployment(
                    strategy_id,
                    f"Daily loss limit breached: ${abs(deployment.pnl_today):,.2f}"
                )
                if result['success']:
                    actions.append({
                        'action': 'pause',
                        'strategy_id': strategy_id,
                        'reason': 'daily_loss_limit'
                    })
        
        return actions
