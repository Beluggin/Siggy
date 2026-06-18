# scheduler_UPDATED.py
"""
UPDATED SCHEDULER - Uses cognitive_state for initiative decisions

CHANGES:
- Now checks cognitive_state.should_initiate() instead of just cooldown
- More sophisticated initiative logic based on full state

LEARNING NOTE: Initiative now depends on:
- Curiosity (high = more initiative)
- Engagement (high = more initiative)
- Frustration (high = less initiative)
"""

import time
from typing import Optional, Dict, Any
from thread_registry import ThreadRegistry
from goal_engine_DAEMON import GoalEngine
from cognitive_state import get_cognitive_state

MIN_SECONDS_BETWEEN_INITIATIVE = 45
MIN_TURNS_BETWEEN_INITIATIVE = 3
CURIOSITY_MIN = 0.4

class InitiativeScheduler:
    def __init__(self):
        self._last_initiative_ts: float = 0.0
        self._turns_since_initiative: int = 0
    
    def update_state(self, threads: ThreadRegistry, goals: GoalEngine):
        """Called every turn."""
        self._turns_since_initiative += 1
    
    def evaluate(self, threads: ThreadRegistry, goals: GoalEngine) -> Optional[Dict[str, Any]]:
        """
        Decide if we should initiate.
        
        NEW: Now checks cognitive_state.should_initiate() FIRST
        This prevents initiative when frustrated or disengaged.
        """
        # Check cooldown
        if not self._cooldown_ok():
            return None
        
        # NEW: Check cognitive state
        # If state says "don't initiate", respect that
        if not get_cognitive_state().should_initiate():
            return None
        
        # Original logic for WHAT to say
        goal_id = self._pick_curiosity_goal(goals)
        if goal_id:
            self._mark_fired()
            return {"type": "curiosity_ping", "goal_id": goal_id}
        
        goal_id = self._pick_unresolved_goal(goals)
        if goal_id:
            self._mark_fired()
            return {"type": "goal_nudge", "goal_id": goal_id}
        
        thread_id = self._pick_stale_thread(threads)
        if thread_id:
            self._mark_fired()
            return {"type": "revive_thread", "thread_id": thread_id}
        
        return None
    
    def _cooldown_ok(self) -> bool:
        now = time.time()
        if (now - self._last_initiative_ts) < MIN_SECONDS_BETWEEN_INITIATIVE:
            return False
        if self._turns_since_initiative < MIN_TURNS_BETWEEN_INITIATIVE:
            return False
        return True
    
    def _pick_curiosity_goal(self, goals: GoalEngine) -> Optional[str]:
        top = goals.get_top_curiosity_goals(n=3)
        if not top:
            return None
        if top[0].curiosity < CURIOSITY_MIN:
            return None
        return top[0].id
    
    def _pick_stale_thread(self, threads: ThreadRegistry) -> Optional[str]:
        stale = threads.get_stale_threads(max_age_seconds=600)
        if not stale:
            return None
        stale.sort(key=lambda th: th.last_active)
        return stale[0].id
    
    def _pick_unresolved_goal(self, goals: GoalEngine) -> Optional[str]:
        unresolved = goals.get_unresolved_goals(max_age_seconds=1200)
        if not unresolved:
            return None
        unresolved.sort(key=lambda g: (g.importance, g.last_active), reverse=True)
        return unresolved[0].id
    
    def _mark_fired(self):
        self._last_initiative_ts = time.time()
        self._turns_since_initiative = 0
