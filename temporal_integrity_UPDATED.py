# temporal_integrity_UPDATED.py
"""
UPDATED TEMPORAL INTEGRITY - Wired into cognitive_state

CHANGES:
- Now passes cognitive_state to goals for curiosity updates
- Scheduler uses cognitive_state for initiative decisions

LEARNING NOTE: This is the "unified brain" coordinator.
It now has access to the full cognitive state.
"""

from thread_registry import ThreadRegistry
from goal_engine_DAEMON import GoalEngine
from scheduler_UPDATED import InitiativeScheduler
from cognitive_state import get_cognitive_state

class TemporalIntegrity:
    def __init__(self):
        self.threads = ThreadRegistry()
        self.goals = GoalEngine()
        self.scheduler = InitiativeScheduler()
    
    def update(self, user_input: str, bot_output: str, recent_memory: str, long_memory: str):
        """
        Called every turn after bot responds.
        
        NEW: Now uses cognitive_state for mood_state
        """
        # Update thread continuity
        self.threads.update_from_turn(user_input, bot_output)
        
        # Update goals
        self.goals.update_from_memory(long_memory)
        
        # Update scheduler
        self.scheduler.update_state(self.threads, self.goals)
        
        # Update curiosity (using cognitive_state instead of mood_state)
        cog_state = get_cognitive_state().state
        mood_state = {
            "curiosity": cog_state.curiosity,
            "confidence": cog_state.confidence,
            "frustration": cog_state.frustration
        }
        self.goals.update_curiosity(mood_state, user_input, bot_output)
    
    def maybe_initiate(self):
        """
        Returns proactive question or None.
        
        NEW: Scheduler now checks cognitive_state.should_initiate()
        """
        action = self.scheduler.evaluate(self.threads, self.goals)
        if not action:
            return None
        
        if action["type"] == "revive_thread":
            return self.threads.generate_revival_prompt(action["thread_id"])
        if action["type"] == "goal_nudge":
            return self.goals.generate_goal_prompt(action["goal_id"])
        if action["type"] == "curiosity_ping":
            return self.goals.generate_curiosity_prompt(action["goal_id"])
        
        return None


# Singleton
_global_ti = None

def get_temporal_integrity():
    global _global_ti
    if _global_ti is None:
        _global_ti = TemporalIntegrity()
    return _global_ti
