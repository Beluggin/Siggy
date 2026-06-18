# curiosity_engine_v2.py
"""
═══════════════════════════════════════════════════════════════════
CURIOSITY ENGINE V2 — Good-Sense Gated Curiosity
═══════════════════════════════════════════════════════════════════

WHAT'S NEW (vs curiosity_engine_UPDATED.py):
  - "Good sense" gate: curiosity is tempered by identity, confidence,
    and frustration. You can be curious AND sensible about it.
  - Curiosity TYPES: not all curiosity is equal
    - "rabbit_hole" → deep exploratory (high reward, high risk)
    - "practical"   → problem-solving curiosity (moderate, useful)
    - "social"      → curiosity about user's state/needs
    - "ambient"     → low-grade background interest
  - Integration with daemon: provides typed curiosity signals
  - Curiosity momentum: sustained curiosity in one direction builds

LEARNING NOTE:
  The original curiosity engine was just a number reader.
  This one actually CLASSIFIES what kind of curiosity is active
  and whether it passes the "good sense" test.
"""

import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from cognitive_state import get_cognitive_state


# ═══════════════════════════════════════════════════════════════════
# CURIOSITY TYPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CuriositySignal:
    """Typed curiosity with intensity and good-sense gate."""
    intensity: float          # 0.0–1.0 raw curiosity strength
    type: str                 # "rabbit_hole", "practical", "social", "ambient"
    good_sense_score: float   # 0.0–1.0 should we actually pursue this?
    gated_intensity: float    # intensity * good_sense weighting
    momentum: float           # how long this curiosity type has been sustained
    
    @property
    def is_actionable(self) -> bool:
        """Should the daemon act on this curiosity?"""
        return self.gated_intensity > 0.35
    
    @property
    def is_deep_dive(self) -> bool:
        """Is this a full rabbit-hole signal?"""
        return self.type == "rabbit_hole" and self.gated_intensity > 0.6


# ═══════════════════════════════════════════════════════════════════
# GOOD SENSE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def _compute_good_sense() -> float:
    """
    Should we actually pursue this curiosity?
    
    Good sense = weighted composite of:
      identity_adherence (0.30) — staying true to who we are
      engagement (0.25)         — user is actually interested
      confidence (0.25)         — we believe this is worthwhile
      (1 - frustration) (0.20)  — we're in a good headspace
    
    Returns 0.0–1.0. Below 0.3 = "not a good time."
    """
    s = get_cognitive_state().state
    return (
        s.identity_adherence  * 0.30 +
        s.engagement          * 0.25 +
        s.confidence          * 0.25 +
        (1.0 - s.frustration) * 0.20
    )


# ═══════════════════════════════════════════════════════════════════
# CURIOSITY TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════

# Keywords that signal different curiosity types
_RABBIT_HOLE_SIGNALS = {
    "what if", "wonder", "hypothesis", "deep dive", "rabbit hole",
    "explore", "imagine", "theory", "philosophical", "existential",
    "why does", "how would", "could we"
}

_PRACTICAL_SIGNALS = {
    "how to", "fix", "solve", "implement", "build", "create",
    "step by step", "tutorial", "guide", "help me", "debug"
}

_SOCIAL_SIGNALS = {
    "how are you", "what do you think", "feeling", "mood",
    "you okay", "check in", "talk about", "tell me about yourself"
}


def detect_curiosity_type(text: str) -> str:
    """Classify what kind of curiosity the input signals."""
    t = text.lower()
    
    # Check each type (order matters — rabbit_hole is most specific)
    rabbit_score = sum(1 for kw in _RABBIT_HOLE_SIGNALS if kw in t)
    practical_score = sum(1 for kw in _PRACTICAL_SIGNALS if kw in t)
    social_score = sum(1 for kw in _SOCIAL_SIGNALS if kw in t)
    
    if rabbit_score > practical_score and rabbit_score > social_score:
        return "rabbit_hole"
    if practical_score > 0:
        return "practical"
    if social_score > 0:
        return "social"
    
    return "ambient"


# ═══════════════════════════════════════════════════════════════════
# MOMENTUM TRACKING
# ═══════════════════════════════════════════════════════════════════

class CuriosityMomentum:
    """
    Tracks sustained curiosity in one direction.
    
    If curiosity type stays the same for multiple evaluations,
    momentum builds → stronger signal. If it shifts, momentum resets.
    
    This prevents flip-flopping and rewards sustained exploration.
    """
    
    def __init__(self):
        self._current_type: str = "ambient"
        self._momentum: float = 0.0
        self._last_update: float = time.time()
        self._streak: int = 0
    
    def update(self, curiosity_type: str) -> float:
        """
        Update momentum. Returns current momentum value (0.0–1.0).
        """
        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        
        if curiosity_type == self._current_type:
            # Same type → build momentum
            self._streak += 1
            self._momentum = min(1.0, self._momentum + 0.08 * self._streak)
        else:
            # Type changed → partial reset (don't zero out completely)
            self._current_type = curiosity_type
            self._momentum *= 0.4  # Keep 40% of previous momentum
            self._streak = 1
        
        # Natural decay over time (lose momentum if idle)
        if dt > 5.0:  # More than 5 seconds between updates
            decay = min(0.3, dt * 0.02)
            self._momentum = max(0.0, self._momentum - decay)
        
        return self._momentum
    
    @property
    def current_type(self) -> str:
        return self._current_type
    
    @property
    def streak(self) -> int:
        return self._streak


# ═══════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════

# Module-level momentum tracker
_momentum = CuriosityMomentum()


def get_curiosity_signal(user_input: str = "", bot_output: str = "") -> CuriositySignal:
    """
    Full curiosity evaluation with type, good-sense gate, and momentum.
    
    This is the main function the daemon and main loop should call.
    
    Returns a CuriositySignal with:
      - intensity: raw curiosity from cognitive state
      - type: what kind of curiosity
      - good_sense_score: should we act on it?
      - gated_intensity: intensity * good_sense weighting
      - momentum: sustained direction strength
    """
    state = get_cognitive_state().state
    
    # Raw intensity (same formula as v1 but cleaner)
    raw_intensity = (
        state.curiosity * 0.60 +
        (1.0 - state.cognitive_load) * 0.2 -
        state.frustration * 0.20
    )
    raw_intensity = max(0.0, min(1.0, raw_intensity))
    
    # Detect type from recent input/output
    combined_text = f"{user_input} {bot_output}"
    curiosity_type = detect_curiosity_type(combined_text) if combined_text.strip() else "ambient"
    
    # Good sense gate
    good_sense = _compute_good_sense()
    
    # Momentum
    momentum = _momentum.update(curiosity_type)
    
    # Gated intensity: raw * good_sense, boosted by momentum
    # Momentum can push gated_intensity up to 20% above base
    gated = raw_intensity * (0.5 + good_sense * 0.5)
    gated *= (1.0 + momentum * 0.20)
    gated = max(0.0, min(1.0, gated))
    
    # Special case: rabbit holes get a slight bonus if good_sense is high
    if curiosity_type == "rabbit_hole" and good_sense > 0.6:
        gated = min(1.0, gated * 1.10)
    
    # Special case: practical curiosity is always somewhat gated-through
    # (you should always be allowed to try to fix things)
    if curiosity_type == "practical" and gated < 0.3:
        gated = 0.3
    
    return CuriositySignal(
        intensity=raw_intensity,
        type=curiosity_type,
        good_sense_score=good_sense,
        gated_intensity=gated,
        momentum=momentum,
    )


# ═══ BACKWARD COMPATIBILITY ═══

def get_curiosity_intensity() -> float:
    """
    Drop-in replacement for curiosity_engine_UPDATED.get_curiosity_intensity()
    Now returns gated intensity.
    """
    return get_curiosity_signal().gated_intensity


def get_random_curiosity_prompt():
    """Backward compatible."""
    import random
    signal = get_curiosity_signal()
    
    if signal.type == "rabbit_hole":
        prompts = [
            "What's the deeper question here?",
            "Follow that thread — where does it lead?",
            "There's something under the surface. What is it?",
        ]
    elif signal.type == "practical":
        prompts = [
            "What's the next concrete step?",
            "How would you test this?",
            "What's blocking progress here?",
        ]
    elif signal.type == "social":
        prompts = [
            "How are you doing with all this?",
            "What matters most to you right now?",
            "Is this what you actually want to talk about?",
        ]
    else:
        prompts = [
            "Anything catching your attention?",
            "What's on your mind?",
            "Where should we go from here?",
        ]
    
    return random.choice(prompts)


# ═══ DIAGNOSTICS ═══

def get_curiosity_report() -> str:
    """Detailed curiosity diagnostic."""
    signal = get_curiosity_signal()
    return (
        f"[CURIOSITY]\n"
        f"  Type: {signal.type}\n"
        f"  Raw: {signal.intensity:.2f}\n"
        f"  Good Sense: {signal.good_sense_score:.2f}\n"
        f"  Gated: {signal.gated_intensity:.2f}\n"
        f"  Momentum: {signal.momentum:.2f} (streak={_momentum.streak})\n"
        f"  Actionable: {signal.is_actionable}\n"
        f"  Deep Dive: {signal.is_deep_dive}"
    )
