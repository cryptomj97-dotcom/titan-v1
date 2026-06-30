"""Gateway module for user approval and strategy deployment."""

from .approval_gateway import ApprovalGateway, StrategyApproval
from .paper_trader import PaperTrader
from .deployment_manager import DeploymentManager

__all__ = ['ApprovalGateway', 'StrategyApproval', 'PaperTrader', 'DeploymentManager']
