"""Approval gateway for strategy deployment."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Strategy approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    PAUSED = "paused"


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    sharpe_ratio: float
    probabilistic_sharpe: float
    deflated_sharpe: float
    max_drawdown: float
    total_return: float
    annualized_return: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_pnl: float
    calmar_ratio: float
    var_95: float
    cvar_95: float


@dataclass
class StrategyApproval:
    """Strategy awaiting approval."""
    strategy_id: str
    strategy_name: str
    strategy_type: str
    parameters: Dict[str, Any]
    metrics: StrategyMetrics
    assets: List[str]
    backtest_start: datetime
    backtest_end: datetime
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    notes: str = ""


class ApprovalGateway:
    """
    Gateway for reviewing and approving trading strategies.
    
    Presents top strategies with full metrics, equity curves,
    and trade logs for user approval before deployment.
    """
    
    def __init__(self, min_probabilistic_sharpe: float = 0.95,
                 min_deflated_sharpe: float = 0.5,
                 max_drawdown_threshold: float = 0.15,
                 min_num_trades: int = 100,
                 min_assets: int = 10):
        """
        Initialize approval gateway.
        
        Args:
            min_probabilistic_sharpe: Minimum PSR for auto-qualification
            min_deflated_sharpe: Minimum DSR for auto-qualification
            max_drawdown_threshold: Maximum acceptable drawdown
            min_num_trades: Minimum number of trades required
            min_assets: Minimum number of assets tested
        """
        self.min_probabilistic_sharpe = min_probabilistic_sharpe
        self.min_deflated_sharpe = min_deflated_sharpe
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_num_trades = min_num_trades
        self.min_assets = min_assets
        
        self.pending_strategies: Dict[str, StrategyApproval] = {}
        self.approved_strategies: Dict[str, StrategyApproval] = {}
        self.rejected_strategies: Dict[str, StrategyApproval] = {}
        self.deployed_strategies: Dict[str, StrategyApproval] = {}
        
        self.approval_history: List[Dict] = []
        
        logger.info("Approval gateway initialized")
        
    def submit_strategy(self, 
                       strategy_id: str,
                       strategy_name: str,
                       strategy_type: str,
                       parameters: Dict[str, Any],
                       metrics: Dict[str, float],
                       assets: List[str],
                       backtest_period: tuple,
                       equity_curve: List[float],
                       trade_log: List[Dict]) -> StrategyApproval:
        """
        Submit a strategy for approval.
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_name: Human-readable name
            strategy_type: Type of strategy (e.g., 'momentum', 'mean_reversion')
            parameters: Strategy parameters
            metrics: Performance metrics dictionary
            assets: List of assets tested
            backtest_period: (start_date, end_date) tuple
            equity_curve: List of cumulative returns
            trade_log: List of trade dictionaries
            
        Returns:
            Created StrategyApproval object
        """
        # Create metrics object
        metrics_obj = StrategyMetrics(
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            probabilistic_sharpe=metrics.get('probabilistic_sharpe', 0.0),
            deflated_sharpe=metrics.get('deflated_sharpe', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            total_return=metrics.get('total_return', 0.0),
            annualized_return=metrics.get('annualized_return', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            num_trades=metrics.get('num_trades', 0),
            avg_trade_pnl=metrics.get('avg_trade_pnl', 0.0),
            calmar_ratio=metrics.get('calmar_ratio', 0.0),
            var_95=metrics.get('var_95', 0.0),
            cvar_95=metrics.get('cvar_95', 0.0)
        )
        
        # Create approval object
        approval = StrategyApproval(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            parameters=parameters,
            metrics=metrics_obj,
            assets=assets,
            backtest_start=backtest_period[0] if isinstance(backtest_period[0], datetime) else datetime.fromisoformat(backtest_period[0]),
            backtest_end=backtest_period[1] if isinstance(backtest_period[1], datetime) else datetime.fromisoformat(backtest_period[1]),
            equity_curve=equity_curve,
            trade_log=trade_log
        )
        
        # Check if strategy meets minimum criteria
        qualification = self._check_qualification(approval)
        approval.notes = qualification['notes']
        
        self.pending_strategies[strategy_id] = approval
        
        logger.info(f"Strategy submitted for approval: {strategy_id} "
                   f"(PSR: {metrics_obj.probabilistic_sharpe:.3f}, "
                   f"DSR: {metrics_obj.deflated_sharpe:.3f})")
        
        return approval
    
    def _check_qualification(self, approval: StrategyApproval) -> Dict:
        """Check if strategy meets qualification criteria."""
        m = approval.metrics
        notes = []
        qualified = True
        
        if m.probabilistic_sharpe < self.min_probabilistic_sharpe:
            notes.append(f"PSR {m.probabilistic_sharpe:.3f} < {self.min_probabilistic_sharpe}")
            qualified = False
            
        if m.deflated_sharpe < self.min_deflated_sharpe:
            notes.append(f"DSR {m.deflated_sharpe:.3f} < {self.min_deflated_sharpe}")
            qualified = False
            
        if m.max_drawdown > self.max_drawdown_threshold:
            notes.append(f"MaxDD {m.max_drawdown:.2%} > {self.max_drawdown_threshold:.2%}")
            qualified = False
            
        if m.num_trades < self.min_num_trades:
            notes.append(f"Trades {m.num_trades} < {self.min_num_trades}")
            qualified = False
            
        if len(approval.assets) < self.min_assets:
            notes.append(f"Assets {len(approval.assets)} < {self.min_assets}")
            qualified = False
        
        if qualified:
            notes.append("✓ Meets all qualification criteria")
        
        return {'qualified': qualified, 'notes': '; '.join(notes)}
    
    def get_pending_strategies(self, sort_by: str = 'probabilistic_sharpe') -> List[StrategyApproval]:
        """Get all pending strategies sorted by metric."""
        strategies = list(self.pending_strategies.values())
        
        if sort_by == 'probabilistic_sharpe':
            strategies.sort(key=lambda s: s.metrics.probabilistic_sharpe, reverse=True)
        elif sort_by == 'deflated_sharpe':
            strategies.sort(key=lambda s: s.metrics.deflated_sharpe, reverse=True)
        elif sort_by == 'sharpe_ratio':
            strategies.sort(key=lambda s: s.metrics.sharpe_ratio, reverse=True)
        elif sort_by == 'total_return':
            strategies.sort(key=lambda s: s.metrics.total_return, reverse=True)
        
        return strategies
    
    def approve_strategy(self, strategy_id: str, approved_by: str,
                        notes: str = "") -> Dict:
        """
        Approve a strategy for deployment.
        
        Args:
            strategy_id: Strategy to approve
            approved_by: User approving the strategy
            notes: Optional approval notes
            
        Returns:
            Approval result
        """
        if strategy_id not in self.pending_strategies:
            return {'success': False, 'error': 'Strategy not found'}
        
        approval = self.pending_strategies[strategy_id]
        approval.status = ApprovalStatus.APPROVED
        approval.approved_at = datetime.now()
        approval.approved_by = approved_by
        approval.notes = notes if notes else approval.notes
        
        # Move to approved dict
        self.approved_strategies[strategy_id] = approval
        del self.pending_strategies[strategy_id]
        
        # Record in history
        self.approval_history.append({
            'action': 'approved',
            'strategy_id': strategy_id,
            'timestamp': approval.approved_at.isoformat(),
            'approved_by': approved_by
        })
        
        logger.info(f"Strategy approved: {strategy_id} by {approved_by}")
        
        return {
            'success': True,
            'strategy_id': strategy_id,
            'status': 'approved',
            'approved_at': approval.approved_at.isoformat()
        }
    
    def reject_strategy(self, strategy_id: str, rejected_by: str,
                       reason: str) -> Dict:
        """
        Reject a strategy.
        
        Args:
            strategy_id: Strategy to reject
            rejected_by: User rejecting the strategy
            reason: Reason for rejection
            
        Returns:
            Rejection result
        """
        if strategy_id not in self.pending_strategies:
            return {'success': False, 'error': 'Strategy not found'}
        
        approval = self.pending_strategies[strategy_id]
        approval.status = ApprovalStatus.REJECTED
        approval.rejection_reason = reason
        
        # Move to rejected dict
        self.rejected_strategies[strategy_id] = approval
        del self.pending_strategies[strategy_id]
        
        # Record in history
        self.approval_history.append({
            'action': 'rejected',
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'rejected_by': rejected_by,
            'reason': reason
        })
        
        logger.warning(f"Strategy rejected: {strategy_id} - {reason}")
        
        return {
            'success': True,
            'strategy_id': strategy_id,
            'status': 'rejected',
            'reason': reason
        }
    
    def get_strategy_details(self, strategy_id: str) -> Optional[Dict]:
        """Get detailed information about a strategy."""
        # Search all dictionaries
        for strategy_dict in [self.pending_strategies, self.approved_strategies, 
                             self.rejected_strategies, self.deployed_strategies]:
            if strategy_id in strategy_dict:
                approval = strategy_dict[strategy_id]
                
                return {
                    'strategy_id': approval.strategy_id,
                    'strategy_name': approval.strategy_name,
                    'strategy_type': approval.strategy_type,
                    'status': approval.status.value,
                    'parameters': approval.parameters,
                    'metrics': {
                        'sharpe_ratio': approval.metrics.sharpe_ratio,
                        'probabilistic_sharpe': approval.metrics.probabilistic_sharpe,
                        'deflated_sharpe': approval.metrics.deflated_sharpe,
                        'max_drawdown': approval.metrics.max_drawdown,
                        'total_return': approval.metrics.total_return,
                        'annualized_return': approval.metrics.annualized_return,
                        'win_rate': approval.metrics.win_rate,
                        'profit_factor': approval.metrics.profit_factor,
                        'num_trades': approval.metrics.num_trades,
                        'calmar_ratio': approval.metrics.calmar_ratio
                    },
                    'assets': approval.assets,
                    'backtest_period': {
                        'start': approval.backtest_start.isoformat(),
                        'end': approval.backtest_end.isoformat()
                    },
                    'equity_curve_summary': {
                        'start': approval.equity_curve[0] if approval.equity_curve else 0,
                        'end': approval.equity_curve[-1] if approval.equity_curve else 0,
                        'peak': max(approval.equity_curve) if approval.equity_curve else 0,
                        'trough': min(approval.equity_curve) if approval.equity_curve else 0
                    },
                    'recent_trades': approval.trade_log[-10:],  # Last 10 trades
                    'notes': approval.notes,
                    'rejection_reason': approval.rejection_reason,
                    'approved_by': approval.approved_by,
                    'created_at': approval.created_at.isoformat(),
                    'approved_at': approval.approved_at.isoformat() if approval.approved_at else None
                }
        
        return None
    
    def mark_deployed(self, strategy_id: str) -> Dict:
        """Mark an approved strategy as deployed."""
        if strategy_id not in self.approved_strategies:
            return {'success': False, 'error': 'Strategy not found or not approved'}
        
        approval = self.approved_strategies[strategy_id]
        approval.status = ApprovalStatus.DEPLOYED
        
        self.deployed_strategies[strategy_id] = approval
        del self.approved_strategies[strategy_id]
        
        self.approval_history.append({
            'action': 'deployed',
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Strategy deployed: {strategy_id}")
        
        return {'success': True, 'strategy_id': strategy_id, 'status': 'deployed'}
    
    def pause_strategy(self, strategy_id: str, reason: str) -> Dict:
        """Pause a deployed strategy."""
        if strategy_id not in self.deployed_strategies:
            return {'success': False, 'error': 'Strategy not found or not deployed'}
        
        approval = self.deployed_strategies[strategy_id]
        approval.status = ApprovalStatus.PAUSED
        approval.notes += f"\nPaused: {reason}"
        
        logger.warning(f"Strategy paused: {strategy_id} - {reason}")
        
        return {'success': True, 'strategy_id': strategy_id, 'status': 'paused'}
    
    def get_approval_dashboard(self) -> Dict:
        """Get dashboard summary for approval decisions."""
        pending = self.get_pending_strategies()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'pending_count': len(self.pending_strategies),
                'approved_count': len(self.approved_strategies),
                'deployed_count': len(self.deployed_strategies),
                'rejected_count': len(self.rejected_strategies)
            },
            'top_pending': [
                {
                    'strategy_id': s.strategy_id,
                    'strategy_name': s.strategy_name,
                    'psr': s.metrics.probabilistic_sharpe,
                    'dsr': s.metrics.deflated_sharpe,
                    'sharpe': s.metrics.sharpe_ratio,
                    'max_dd': s.metrics.max_drawdown,
                    'total_return': s.metrics.total_return,
                    'num_trades': s.metrics.num_trades,
                    'num_assets': len(s.assets)
                }
                for s in pending[:5]  # Top 5 pending
            ],
            'recent_approvals': [
                {
                    'strategy_id': s.strategy_id,
                    'approved_at': s.approved_at.isoformat() if s.approved_at else None,
                    'approved_by': s.approved_by
                }
                for s in list(self.approved_strategies.values())[-5:]
            ]
        }
