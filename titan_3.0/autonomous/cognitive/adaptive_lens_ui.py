"""
6. Adaptive Lens UI - Dual-Mode Interface
Provides different interfaces for normal users (Oracle mode) and pro users (Architect mode).
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UserPreference:
    user_id: str
    experience_level: str  # 'beginner', 'intermediate', 'advanced'
    preferred_mode: Optional[str]  # 'oracle' or 'architect'
    risk_tolerance: float  # 0.0 to 1.0

class AdaptiveLensUI:
    """
    Adaptive User Interface System
    Dynamically adjusts complexity based on user expertise
    """
    
    def __init__(self):
        self.user_profiles: Dict[str, UserPreference] = {}
        self.session_data: Dict[str, Dict] = {}
        
    def register_user(self, user_id: str, experience_level: str = 'beginner') -> None:
        """Register a new user with preferences"""
        self.user_profiles[user_id] = UserPreference(
            user_id=user_id,
            experience_level=experience_level,
            preferred_mode=None,
            risk_tolerance=0.5 if experience_level == 'beginner' else 0.7
        )
        logger.info(f"Registered user {user_id} as {experience_level}")
    
    def get_user_mode(self, user_id: str) -> str:
        """Determine UI mode based on user profile"""
        if user_id not in self.user_profiles:
            return 'oracle'  # Default to simple mode
        
        profile = self.user_profiles[user_id]
        
        # Use explicit preference if set
        if profile.preferred_mode:
            return profile.preferred_mode
        
        # Auto-detect based on experience
        if profile.experience_level in ['advanced', 'intermediate']:
            return 'architect'
        else:
            return 'oracle'
    
    def generate_dashboard_data(
        self, 
        user_id: str, 
        organism_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate appropriate dashboard data based on user mode"""
        mode = self.get_user_mode(user_id)
        
        if mode == 'oracle':
            return self._generate_oracle_view(user_id, organism_state)
        else:
            return self._generate_architect_view(user_id, organism_state)
    
    def _generate_oracle_view(self, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple, clean interface for normal users"""
        consensus_action = state.get('consensus_action', 'HOLD')
        confidence = state.get('consensus_confidence', 0.0)
        
        # Convert to simple signal
        if consensus_action == 'BUY' and confidence > 0.7:
            signal = "STRONG BUY"
            color = "green"
        elif consensus_action == 'BUY' and confidence > 0.5:
            signal = "BUY"
            color = "lightgreen"
        elif consensus_action == 'SELL' and confidence > 0.7:
            signal = "STRONG SELL"
            color = "red"
        elif consensus_action == 'SELL' and confidence > 0.5:
            signal = "SELL"
            color = "orange"
        else:
            signal = "HOLD"
            color = "gray"
        
        return {
            'mode': 'oracle',
            'signal': signal,
            'color': color,
            'confidence_pct': round(confidence * 100, 1),
            'action': consensus_action,
            'top_assets': state.get('top_assets', []),
            'last_update': time.time(),
            'next_review': time.time() + 300  # 5 minutes
        }
    
    def _generate_architect_view(self, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed, complex interface for pro users"""
        return {
            'mode': 'architect',
            'agents_status': state.get('agent_decisions', {}),
            'debate_log': state.get('debate_log', []),
            'market_conditions': state.get('market_data', {}),
            'signals_pool_count': state.get('signals_count', 0),
            'recent_executions': state.get('recent_executions', []),
            'performance_metrics': state.get('performance', {}),
            'synthetic_scenarios': state.get('scenario_analysis', {}),
            'risk_metrics': state.get('risk_metrics', {}),
            'neuro_symbolic_rules': state.get('applied_rules', []),
            'hunter_stats': state.get('hunter_stats', {}),
            'last_update': time.time(),
            'refresh_rate_ms': 100
        }
    
    def set_user_mode(self, user_id: str, mode: str) -> bool:
        """Manually set user's preferred mode"""
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found")
            return False
        
        if mode not in ['oracle', 'architect']:
            logger.warning(f"Invalid mode: {mode}")
            return False
        
        self.user_profiles[user_id].preferred_mode = mode
        logger.info(f"Set user {user_id} mode to {mode}")
        return True
    
    def update_session(self, user_id: str, data: Dict[str, Any]) -> None:
        """Update session data for user"""
        if user_id not in self.session_data:
            self.session_data[user_id] = {}
        
        self.session_data[user_id].update(data)
        self.session_data[user_id]['last_activity'] = time.time()
    
    def get_session(self, user_id: str) -> Dict[str, Any]:
        """Get current session data for user"""
        return self.session_data.get(user_id, {})
    
    def cleanup_inactive_sessions(self, max_age_seconds: float = 3600) -> int:
        """Remove inactive sessions"""
        current_time = time.time()
        inactive_users = [
            uid for uid, session in self.session_data.items()
            if current_time - session.get('last_activity', 0) > max_age_seconds
        ]
        
        for uid in inactive_users:
            del self.session_data[uid]
        
        if inactive_users:
            logger.info(f"Cleaned up {len(inactive_users)} inactive sessions")
        
        return len(inactive_users)
