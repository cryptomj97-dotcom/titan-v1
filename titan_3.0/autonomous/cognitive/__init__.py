"""
Unified Cognitive Quantitative Organism - Core Module
Implements Multi-Agent Debate, Generative World Models, Neuro-Symbolic Reasoning,
Real-Time Pattern Scanning, HFT Execution, Adaptive UI, and Meta-Learning.
"""

from .multi_agent_debate import (
    SignalPool,
    Agent,
    TrendFollowingAgent,
    MeanReversionAgent,
    RiskManagementAgent,
    AgentCouncil,
    AgentDecision,
    ConsensusResult
)

from .generative_world_models import GenerativeWorldModel

from .neuro_symbolic_engine import NeuroSymbolicEngine

from .hunter_daemon import HunterDaemon

from .hft_executor import HFTExecutor

from .adaptive_lens_ui import AdaptiveLensUI

from .meta_learning_loop import MetaLearningLoop

from .unified_organism import UnifiedCognitiveQuantitativeOrganism

__all__ = [
    'SignalPool',
    'Agent',
    'TrendFollowingAgent',
    'MeanReversionAgent',
    'RiskManagementAgent',
    'AgentCouncil',
    'AgentDecision',
    'ConsensusResult',
    'GenerativeWorldModel',
    'NeuroSymbolicEngine',
    'HunterDaemon',
    'HFTExecutor',
    'AdaptiveLensUI',
    'MetaLearningLoop',
    'UnifiedCognitiveQuantitativeOrganism'
]
