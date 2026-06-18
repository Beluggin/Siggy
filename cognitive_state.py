# cognitive_state.py
"""
Unified Cognitive State System
-------------------------------
Integrates state vectors across ALL cognitive subsystems:
- Memory retrieval (TWDC)
- Goal prioritization
- Thread management
- Initiative timing
- Response tone

This is the "nervous system" that connects everything.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

STATE_PATH = Path(__file__).parent / "cognitive_state.json"


def _mentions(text: str, triggers) -> bool:
    # Whole-word match, not substring — "no" must NOT fire on "know"/"now"/"nothing".
    # Substring matching here was the bug that pinned frustration at 1.0 and
    # silently killed the playful tone for months. Phrases ("what if") work too.
    return any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in triggers)

@dataclass
class CognitiveState:
    """
    State vectors that modulate ALL cognitive processes.
    These get updated every turn based on interaction dynamics.
    """
    # Emotional state
    frustration: float = 0.2
    curiosity: float = 0.8
    confidence: float = 0.6
    engagement: float = 0.9
    
    # Identity & grounding
    identity_adherence: float = 0.7  # How strongly to anchor to core identity facts
    context: float = 0.9              # How much recent context to use
    
    # Communication style
    tone_playful: float = 0.6
    tone_formal: float = 0.3
    tone_concise: float = 0.5
    tone_warm: float = 0.7
    
    # Processing capacity
    cognitive_load: float = 0.7       # Higher = simpler responses needed
    recursion_tolerance: float = 0.5  # Tolerance for deep/abstract thinking
    affect_matching: float = 0.6      # Match user's emotional tone
    
    # Metadata
    last_update: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dict with tone sub-dict"""
        d = asdict(self)
        # Restructure tone fields
        d["tone"] = {
            "playful": d.pop("tone_playful"),
            "formal": d.pop("tone_formal"),
            "concise": d.pop("tone_concise"),
            "warm": d.pop("tone_warm")
        }
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CognitiveState':
        """Load from nested dict"""
        tone = d.get("tone", {})
        return cls(
            frustration=d.get("frustration", 0.2),
            curiosity=d.get("curiosity", 0.8),
            confidence=d.get("confidence", 0.6),
            engagement=d.get("engagement", 0.9),
            identity_adherence=d.get("identity_adherence", 0.7),
            context=d.get("context", 0.9),
            tone_playful=tone.get("playful", 0.6),
            tone_formal=tone.get("formal", 0.3),
            tone_concise=tone.get("concise", 0.5),
            tone_warm=tone.get("warm", 0.7),
            cognitive_load=d.get("cognitive_load", 0.7),
            recursion_tolerance=d.get("recursion_tolerance", 0.5),
            affect_matching=d.get("affect_matching", 0.6),
            last_update=d.get("last_update", time.time())
        )


class CognitiveStateEngine:
    """
    Central state management system.
    All cognitive subsystems read from this.
    """

    def __init__(self, data_dir=None):
        # data_dir: per-user Path for web app; None = use module-level STATE_PATH (CLI)
        self._state_path = Path(data_dir) / "cognitive_state.json" if data_dir else STATE_PATH
        self.state = self._load_state()

    def _load_state(self) -> CognitiveState:
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                return CognitiveState.from_dict(data)
            except Exception:
                pass
        return CognitiveState()

    def _save_state(self):
        self._state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2),
            encoding="utf-8"
        )
    
    # ============ STATE UPDATES ============
    
    def update_from_interaction(
        self,
        user_input: str,
        bot_output: str,
        intent_label: str = "GENERAL",
        latency_ms: float = 0.0
    ):
        """
        Update cognitive state based on the interaction.
        This is the main feedback loop.
        """
        u = user_input.lower()
        b = bot_output.lower()
        
        # 1. FRUSTRATION
        # Increase if: user shows irritation, bot took too long, corrections
        if _mentions(u, ["wrong", "no", "fix", "broken", "again", "stop"]):
            self.state.frustration = min(1.0, self.state.frustration + 0.25)
        elif latency_ms > 300000:
            self.state.frustration = min(1.0, self.state.frustration + 0.15)
        else:
            # Natural decay toward baseline
            self.state.frustration = self.state.frustration * 0.9 + 0.2 * 0.1
        
        # 2. CURIOSITY
        # Increase if: questions, "what if", exploring, rabbit holes
        if _mentions(u, ["why", "how", "what if", "wonder", "curious", "explore"]):
            self.state.curiosity = min(1.0, self.state.curiosity + 0.2)
        elif _mentions(u, ["ok", "thanks", "got it", "done"]):
            # Closure reduces curiosity
            self.state.curiosity = max(0.3, self.state.curiosity - 0.15)
        else:
            # Slow decay to mid-range
            self.state.curiosity = self.state.curiosity * 0.95 + 0.5 * 0.05
        
        # 3. CONFIDENCE
        # Increase if: positive feedback, successful completions
        # Decrease if: errors, uncertainty, user corrections
        if intent_label == "SUCCESS_SIGNAL" or _mentions(u, ["perfect", "great", "exactly", "yes"]):
            self.state.confidence = min(1.0, self.state.confidence + 0.15)
        elif intent_label == "CRITICISM" or _mentions(u, ["wrong"]):
            self.state.confidence = max(0.2, self.state.confidence - 0.2)
        
        # 4. ENGAGEMENT
        # High when: long messages, emoji, enthusiasm
        # Low when: short/dismissive responses
        # Punctuation/emoji stay substring checks (\b can't bound them); words get _mentions
        if any(c in user_input for c in ["!", "😊", "🎉"]) or _mentions(u, ["awesome", "cool", "amazing", "interesting"]):
            self.state.engagement = min(1.0, self.state.engagement + 0.15)
        elif len(user_input.split()) < 5:
            self.state.engagement = max(0.3, self.state.engagement - 0.1)
        
        # 5. COGNITIVE LOAD
        # High when: complex questions, long inputs, technical depth
        # Low when: simple queries, greetings
        word_count = len(user_input.split())
        if word_count > 50 or _mentions(u, ["architecture", "implement", "design", "algorithm"]):
            self.state.cognitive_load = 0.85
        elif word_count < 10:
            self.state.cognitive_load = 0.3
        else:
            # Gradual return to baseline
            self.state.cognitive_load = self.state.cognitive_load * 0.8 + 0.5 * 0.2
        
        # 6. IDENTITY ADHERENCE
        # Boost when: asking about personal facts, relationships, core identity
        if _mentions(u, ["adam", "griffin", "sophie", "mason", "your name", "my name", "remember"]):
            self.state.identity_adherence = min(1.0, self.state.identity_adherence + 0.2)
        else:
            # Slow drift toward baseline
            self.state.identity_adherence = self.state.identity_adherence * 0.98 + 0.7 * 0.02
        
        # 7. TONE ADJUSTMENTS
        # Each slider mean-reverts toward its baseline when not triggered —
        # without this they're a one-way ratchet (all four were pinned at 1.0
        # for months, which is what flattened the personality).
        # Playful: detected through humor, casual language
        if _mentions(u, ["haha", "lol", "funny", "joke"]):
            self.state.tone_playful = min(1.0, self.state.tone_playful + 0.15)
        else:
            self.state.tone_playful = self.state.tone_playful * 0.9 + 0.6 * 0.1

        # Formal: detected through technical language, professional context
        if _mentions(u, ["please", "kindly", "sir", "madam"]) or intent_label == "FORMAL_REQUEST":
            self.state.tone_formal = min(1.0, self.state.tone_formal + 0.2)
        else:
            self.state.tone_formal = self.state.tone_formal * 0.9 + 0.3 * 0.1

        # Concise: user wants brevity
        if _mentions(u, ["brief", "short", "tldr", "quick"]):
            self.state.tone_concise = min(1.0, self.state.tone_concise + 0.3)
        else:
            self.state.tone_concise = self.state.tone_concise * 0.9 + 0.5 * 0.1

        # Warm: positive emotional tone
        if self.state.engagement > 0.7 and self.state.frustration < 0.4:
            self.state.tone_warm = min(1.0, self.state.tone_warm + 0.1)
        else:
            self.state.tone_warm = self.state.tone_warm * 0.9 + 0.7 * 0.1

        self.state.last_update = time.time()
        self._clamp_all()  # clamp BEFORE saving so the persisted state is always clean
        self._save_state()
    
    def _clamp_all(self):
        """Ensure all values stay in [0, 1]"""
        for field in self.state.__dataclass_fields__:
            if field != "last_update":
                val = getattr(self.state, field)
                if isinstance(val, (int, float)):
                    setattr(self.state, field, max(0.0, min(1.0, val)))
    
    # ============ MEMORY MODULATION ============
    
    def get_memory_retrieval_params(self) -> Dict[str, Any]:
        """
        Returns parameters for memory retrieval based on current state.
        Used by TWDC system to adjust what gets surfaced.
        """
        return {
            "identity_boost": self.state.identity_adherence * 0.5,  # Boost identity facts
            "recency_weight": 1.0 - self.state.context * 0.3,       # Less context = more recency
            "complexity_tolerance": 1.0 - self.state.cognitive_load,  # Low load = complex OK
            "curiosity_filter": self.state.curiosity,  # High curiosity = surface rabbit holes
            "frustration_penalty": self.state.frustration * 0.4,  # Penalize abstract when frustrated
        }
    
    def get_response_constraints(self) -> Dict[str, Any]:
        """
        Returns constraints for response generation.
        Used to modify prompt instructions.
        """
        max_length = 200  # baseline
        
        # Cognitive load: higher load = shorter responses
        if self.state.cognitive_load > 0.7:
            max_length = 120
        elif self.state.cognitive_load < 0.4:
            max_length = 300
        
        # Conciseness preference
        if self.state.tone_concise > 0.7:
            max_length = int(max_length * 0.6)
        
        return {
            "max_length": max_length,
            "use_examples": self.state.cognitive_load < 0.5,  # Examples when load is low
            "structured_format": self.state.cognitive_load > 0.6,  # Bullets when overloaded
        }
    
    def get_tone_instructions(self) -> str:
        """
        Returns natural language tone instructions for the prompt.
        """
        parts = []
        
        # Frustration → direct and practical
        if self.state.frustration > 0.6:
            parts.append("Be direct and solution-focused. Skip philosophy.")
        
        # Curiosity → exploratory
        if self.state.curiosity > 0.7:
            parts.append("Follow rabbit holes. Ask deeper questions.")
        
        # Cognitive load → simplify
        if self.state.cognitive_load > 0.7:
            parts.append("Keep it simple and structured. Use numbered steps.")
        
        # Tone modifiers
        if self.state.tone_playful > 0.6 and self.state.frustration < 0.4:
            parts.append("Be playful and creative.")
        
        if self.state.tone_warm > 0.6:
            parts.append("Be warm and encouraging.")
        
        if self.state.tone_formal > 0.6:
            parts.append("Maintain a professional tone.")
        
        if self.state.tone_concise > 0.7:
            parts.append("Be extremely concise.")
        
        if not parts:
            return "Be candid, practical, and slightly irreverent."
        
        return " ".join(parts)
    
    def should_initiate(self) -> bool:
        """
        Decide if bot should proactively speak.
        Based on curiosity + engagement - frustration.
        """
        initiative_signal = (
            self.state.curiosity * 0.6 +
            self.state.engagement * 0.3 -
            self.state.frustration * 0.5
        )
        
        # Threshold: only initiate when signal is strong
        return initiative_signal > 0.75
    
    def get_vitals_report(self) -> str:
        """
        Internal state summary for debugging / prompt injection.
        """
        s = self.state
        report = "[COGNITIVE_STATE]\n"
        
        if s.frustration > 0.7:
            report += f"- Frustration HIGH ({s.frustration:.2f}): Prioritize practical solutions\n"
        
        if s.curiosity > 0.75:
            report += f"- Curiosity PEAK ({s.curiosity:.2f}): Deep dive mode active\n"
        
        if s.cognitive_load > 0.7:
            report += f"- Cognitive load HIGH ({s.cognitive_load:.2f}): Simplify responses\n"
        
        if s.identity_adherence > 0.8:
            report += f"- Identity adherence HIGH ({s.identity_adherence:.2f}): Ground in core facts\n"
        
        if s.engagement < 0.4:
            report += f"- Engagement LOW ({s.engagement:.2f}): User may be disengaging\n"
        
        return report if len(report) > 20 else ""


# Singleton instance (CLI / root-dir usage)
_engine: Optional[CognitiveStateEngine] = None

def get_cognitive_state(data_dir=None) -> CognitiveStateEngine:
    global _engine
    if data_dir is not None:
        return CognitiveStateEngine(data_dir)   # per-user: always fresh
    if _engine is None:
        _engine = CognitiveStateEngine()        # CLI: lazy-init global
    return _engine

# Convenience functions
def update_from_interaction(user_input: str, bot_output: str, intent_label: str = "GENERAL", latency_ms: float = 0.0):
    engine = get_cognitive_state()
    engine.update_from_interaction(user_input, bot_output, intent_label, latency_ms)

def get_state() -> CognitiveState:
    return get_cognitive_state().state

def get_tone_instructions() -> str:
    return get_cognitive_state().get_tone_instructions()

def get_memory_retrieval_params() -> Dict[str, Any]:
    return get_cognitive_state().get_memory_retrieval_params()

def should_initiate() -> bool:
    return get_cognitive_state().should_initiate()
