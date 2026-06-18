# goal_engine.py
"""
═══════════════════════════════════════════════════════════════════
GOAL ENGINE — Daemon-Compatible with Action Recommendation Queue
═══════════════════════════════════════════════════════════════════

DAEMON INTEGRATION VERSION:
  - ActionCandidate dataclass for the daemon's recommendation queue
  - Composite scoring (curiosity + identity + importance + engagement)
  - Crap-filtering support (items can be scored and pruned)
  - Thread-safe goal access (daemon reads, main loop writes)
  - Identity relevance scoring per goal
  - Backward compatible with existing GoalEngine API

LEARNING NOTE:
  The daemon continuously evaluates goals. This engine provides
  the raw material. The daemon handles the cycling/scoring/pruning.
"""

import ast
import time
import uuid
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

CURIO_DECAY_PER_SEC = 0.00012      # TUNED: 4x slower decay (was 0.0005)
CURIO_BOOST_AMBIGUOUS = 0.15       # Boost when input looks like a rabbit hole
CURIO_BOOST_MOOD = 0.25            # Scaled by mood_state["curiosity"]
CURIO_FLOOR = 0.10                 # NEW: Goals never drop below this curiosity
MAX_ACTIVE_RABBIT_HOLES = 5        # Increased for daemon (was 3)
MAX_GOALS = 50                     # Cap to prevent unbounded growth
STALE_PURGE_AGE = 3600             # Purge goals older than 1 hour with no activity


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Goal:
    """A tracked goal/thread/preference."""
    id: str
    type: str              # "project", "loop", "preference", "rabbit_hole"
    description: str
    last_active: float
    importance: float
    curiosity: float = 0.0
    unresolved: bool = False
    identity_relevance: float = 0.0   # NEW: how related to core identity
    access_count: int = 0             # NEW: how often this gets evaluated
    created_at: float = 0.0           # NEW: for age tracking


@dataclass
class ActionCandidate:
    """
    A scored action recommendation from the daemon.
    
    This is what flows through the daemon's 9-phase pipeline.
    The main loop reads the top candidates when responding.
    """
    goal_id: str
    description: str
    action_type: str           # "think", "ask_user", "revisit", "explore", "resolve"
    composite_score: float     # Overall priority (0.0–1.5)
    curiosity_score: float     # Raw curiosity component
    identity_score: float      # Identity relevance component
    needs_user: bool = False   # Requires user-in-the-loop?
    reasoning: str = ""        # Why this action was chosen
    
    def to_dict(self) -> Dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "action_type": self.action_type,
            "composite_score": self.composite_score,
            "curiosity_score": self.curiosity_score,
            "identity_score": self.identity_score,
            "needs_user": self.needs_user,
            "reasoning": self.reasoning,
        }


# ═══════════════════════════════════════════════════════════════════
# GOAL ENGINE
# ═══════════════════════════════════════════════════════════════════

class GoalEngine:
    """
    Thread-safe goal engine with daemon-compatible interface.
    
    The daemon reads goals continuously.
    The main loop writes goals on each turn.
    Lock protects concurrent access.
    """
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self._lock = threading.Lock()
        self._last_decay_ts: float = time.time()
    
    # ═══ EXTRACTION FROM MEMORY ═══
    
    def _extract_from_memory(self, long_memory: str) -> List[Goal]:
        """
        Parse goals from long_memory block.
        Adds identity_relevance scoring.
        """
        extracted: List[Goal] = []
        now = time.time()
        
        def _extract_list(label: str, gtype: str, base_importance: float):
            nonlocal extracted
            if f"{label}:" not in long_memory:
                return
            try:
                block = long_memory.split(f"{label}:")[1].split("\n")[0].strip()
                items = ast.literal_eval(block)  # safe: only parses literals, no code execution
                for item in items:
                    desc = str(item)
                    g = Goal(
                        id=str(uuid.uuid4())[:8],
                        type=gtype,
                        description=desc,
                        last_active=now,
                        importance=base_importance,
                        curiosity=0.0,
                        unresolved=False,
                        identity_relevance=0.0,
                        access_count=0,
                        created_at=now,
                    )
                    extracted.append(g)
            except Exception:
                return
        
        _extract_list("Projects", "project", 0.9)
        _extract_list("Open loops", "loop", 0.75)
        _extract_list("Preferences", "preference", 0.4)
        
        return extracted
    
    def update_from_memory(self, long_memory: str):
        """
        Refresh goal list from long-term memory.
        Thread-safe.
        """
        new_goals = self._extract_from_memory(long_memory)
        now = time.time()
        
        with self._lock:
            for g in new_goals:
                matched = False
                for existing in self.goals.values():
                    if g.description.lower() == existing.description.lower():
                        existing.last_active = now
                        existing.access_count += 1
                        matched = True
                        break
                if not matched:
                    self.goals[g.id] = g
            
            # Enforce max goals cap
            self._enforce_cap()
    
    def _enforce_cap(self):
        """Remove oldest low-importance goals if over cap."""
        if len(self.goals) <= MAX_GOALS:
            return
        
        # Sort by composite of recency + importance
        ranked = sorted(
            self.goals.values(),
            key=lambda g: (g.importance * 0.5 + g.curiosity * 0.3 + 
                          (1.0 / max(1, time.time() - g.last_active + 1)) * 0.2),
        )
        
        # Remove the lowest-ranked to get back under cap
        to_remove = len(self.goals) - MAX_GOALS
        for g in ranked[:to_remove]:
            del self.goals[g.id]
    
    # ═══ RABBIT HOLE DETECTION ═══
    
    def add_rabbit_hole(self, description: str, curiosity: float = 0.5) -> str:
        """
        Explicitly add a rabbit hole from conversation.
        Returns goal_id.
        
        Day 28 fix: strips all leading "Unaddressed urgent: " prefixes
        at the point of insertion so no goal can ever be born polluted,
        regardless of which caller supplies the description.
        """
        _PREFIX = "Unaddressed urgent: "
        while description.startswith(_PREFIX):
            description = description[len(_PREFIX):]
        if not description.strip():
            # After stripping, nothing left — don't create an empty goal
            return ""
        
        now = time.time()
        gid = str(uuid.uuid4())[:8]
        
        with self._lock:
            goal = Goal(
                id=gid,
                type="rabbit_hole",
                description=description,
                last_active=now,
                importance=0.6,  # Moderate base importance
                curiosity=curiosity,
                unresolved=True,
                identity_relevance=0.0,
                access_count=0,
                created_at=now,
            )
            self.goals[gid] = goal
            self._enforce_cap()
        
        return gid
    
    # ═══ CURIOSITY DYNAMICS ═══
    
    def _looks_like_rabbit_hole(self, text: str) -> bool:
        t = text.lower()
        return any(
            kw in t
            for kw in [
                "wonder", "curious", "what if", "maybe", "could we",
                "explore", "idea", "hypothesis", "rabbit hole",
                "interesting", "deep dive", "investigate",
            ]
        )
    
    def decay_curiosity(self):
        """Global curiosity decay. Thread-safe. Respects CURIO_FLOOR."""
        now = time.time()
        dt = now - self._last_decay_ts
        if dt <= 0:
            return
        self._last_decay_ts = now
        
        with self._lock:
            for g in self.goals.values():
                if g.curiosity <= CURIO_FLOOR:
                    continue
                g.curiosity = max(CURIO_FLOOR, g.curiosity - CURIO_DECAY_PER_SEC * dt)
    
    def update_curiosity(self, mood_state: Dict, user_input: str, bot_output: str):
        """
        Adjust curiosity scores based on mood and conversation signals.
        Thread-safe.
        """
        self.decay_curiosity()
        
        mood_curio = float(mood_state.get("curiosity", 0.0))
        mood_conf = float(mood_state.get("confidence", 0.0))
        mood_frust = float(mood_state.get("frustration", 0.0))
        
        # Don't boost curiosity when frustrated
        if mood_frust > 0.6:
            return
        
        ambiguous = (
            self._looks_like_rabbit_hole(user_input) or 
            self._looks_like_rabbit_hole(bot_output)
        )
        
        if not ambiguous and mood_curio < 0.3:
            return
        
        boost = 0.0
        if ambiguous:
            boost += CURIO_BOOST_AMBIGUOUS
        if mood_curio > 0.2 and mood_conf < 0.7:
            boost += CURIO_BOOST_MOOD * mood_curio
        
        if boost <= 0:
            return
        
        with self._lock:
            for g in self.goals.values():
                g.curiosity += boost * (0.5 + 0.5 * g.importance)
                g.curiosity = min(1.5, g.curiosity)  # Soft cap
    
    def update_identity_relevance(self, identity_keywords: List[str]):
        """
        Score each goal's relevance to core identity.
        Called by daemon periodically.
        """
        if not identity_keywords:
            return
        
        with self._lock:
            for g in self.goals.values():
                desc_lower = g.description.lower()
                matches = sum(1 for kw in identity_keywords if kw in desc_lower)
                g.identity_relevance = min(1.0, matches * 0.2)
    
    # ═══ SELECTION & PROMPTS (backward compatible) ═══
    
    def get_top_curiosity_goals(self, n: int = MAX_ACTIVE_RABBIT_HOLES) -> List[Goal]:
        with self._lock:
            goals = list(self.goals.values())
        goals.sort(key=lambda g: (g.curiosity, g.importance, g.last_active), reverse=True)
        return goals[:n]
    
    def get_unresolved_goals(self, max_age_seconds: float = 900) -> List[Goal]:
        now = time.time()
        with self._lock:
            return [
                g for g in self.goals.values()
                if g.unresolved or (now - g.last_active) > max_age_seconds
            ]
    
    def get_all_scored(self) -> List[Dict]:
        """
        Get all goals with their current scores.
        Used by daemon for evaluation phase.
        """
        with self._lock:
            return [
                {
                    "id": g.id,
                    "type": g.type,
                    "description": g.description,
                    "importance": g.importance,
                    "curiosity": g.curiosity,
                    "identity_relevance": g.identity_relevance,
                    "unresolved": g.unresolved,
                    "age": time.time() - g.last_active,
                    "access_count": g.access_count,
                }
                for g in self.goals.values()
            ]
    
    def generate_goal_prompt(self, goal_id: str) -> Optional[str]:
        with self._lock:
            g = self.goals.get(goal_id)
        if not g:
            return None
        
        if g.type == "project":
            return (
                f"I'm still thinking about your project: {g.description}.\n"
                f"Is now a good time to go deeper on that?"
            )
        if g.type == "loop":
            return (
                f"Earlier you left this a bit open: {g.description}.\n"
                f"I'm curious where you'd like to take that next."
            )
        if g.type == "rabbit_hole":
            return (
                f"I've been chewing on something: {g.description}.\n"
                f"What angle of that would you actually enjoy exploring?"
            )
        if g.type == "preference":
            return (
                f"You mentioned you prefer '{g.description}'.\n"
                f"Should that shape what we explore right now?"
            )
        return f"I'm still curious about: {g.description}. Want to explore that further?"
    
    def generate_curiosity_prompt(self, goal_id: str) -> Optional[str]:
        with self._lock:
            g = self.goals.get(goal_id)
        if not g:
            return None
        
        return (
            f"[DREAM]\n"
            f"I'm following a rabbit hole in my own head about:\n"
            f"  {g.description}\n"
            f"What angle of that would you actually enjoy exploring?"
        )
    
    # ═══ STALE PURGE ═══
    
    @staticmethod
    def _nuke_prefix(text: str) -> str:
        """Remove ALL occurrences of 'Unaddressed urgent: ' from text,
        whether leading, embedded, or double-nested."""
        _PREFIX = "Unaddressed urgent: "
        # First strip all leading prefixes
        while text.startswith(_PREFIX):
            text = text[len(_PREFIX):]
        # Then remove any remaining embedded occurrences
        # (e.g. "Explore connection between: Unaddressed urgent: X")
        if _PREFIX in text:
            text = text.replace(_PREFIX, "")
        # Collapse any resulting double-spaces
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    def clean_registry(self) -> int:
        """
        Sanitize goals in-place:
          1. Strip all "Unaddressed urgent: " substrings from ANYWHERE
             in the description (leading, embedded, double-nested).
          2. Deduplicate goals with identical *post-strip* descriptions,
             keeping the one with the highest curiosity score.
        Returns count of goals removed (strip-empties + dedup).

        Day 41 fix: use _nuke_prefix() which removes the pollution
        substring from anywhere in the description, not just as a
        leading prefix. This catches embedded pollution like
        "Explore connection between: Unaddressed urgent: X" that
        survived all previous leading-prefix-only strip passes.
        """
        with self._lock:
            # Single combined pass: normalize key, strip in-place, group.
            seen: Dict[str, List[str]] = {}  # normalized_desc -> [goal_ids]
            for gid, g in self.goals.items():
                # Compute the canonical key by nuking ALL prefix occurrences
                desc = self._nuke_prefix(g.description)
                key = desc.lower().strip()

                # Apply the strip to the live goal object
                g.description = desc

                if not key:
                    continue
                seen.setdefault(key, []).append(gid)

            removed = 0

            # Remove goals whose stripped description is empty
            empty_ids = [gid for gid, g in self.goals.items()
                         if not g.description.strip()]
            for gid in empty_ids:
                del self.goals[gid]
                removed += 1

            # Deduplicate groups with >1 member
            for key, gids in seen.items():
                if len(gids) <= 1:
                    continue
                # Filter out any gids already removed (empty-description pass)
                gids = [gid for gid in gids if gid in self.goals]
                if len(gids) <= 1:
                    continue
                # Keep the goal with highest curiosity; break ties by most recent last_active
                best_gid = max(gids, key=lambda gid: (
                    self.goals[gid].curiosity, self.goals[gid].last_active
                ))
                for gid in gids:
                    if gid != best_gid:
                        del self.goals[gid]
                        removed += 1

            return removed

    def purge_stale(self) -> int:
        """
        Remove goals that are old, low-curiosity, and low-importance.
        Returns count of purged goals.
        """
        now = time.time()
        purged = 0
        
        with self._lock:
            to_remove = []
            for gid, g in self.goals.items():
                age = now - g.last_active
                if (
                    age > STALE_PURGE_AGE and
                    g.curiosity < 0.1 and
                    g.importance < 0.5 and
                    not g.unresolved
                ):
                    to_remove.append(gid)
            
            for gid in to_remove:
                del self.goals[gid]
                purged += 1
        
        return purged
    
    # ═══ DIAGNOSTICS ═══
    
    def get_status(self) -> str:
        with self._lock:
            n = len(self.goals)
            top = sorted(
                self.goals.values(),
                key=lambda g: (g.curiosity, g.importance, g.last_active),
                reverse=True,
            )[:3]
        
        lines = [f"[GOALS] {n} tracked"]
        for g in top:
            lines.append(
                f"  [{g.curiosity:.2f}c|{g.importance:.2f}i|{g.identity_relevance:.2f}id] "
                f"{g.type}: {g.description[:40]}"
            )
        return "\n".join(lines)
