# cognitive_modes.py
"""
═══════════════════════════════════════════════════════════════════
COGNITIVE MODE REGISTRY — v6.3
═══════════════════════════════════════════════════════════════════

Modes are NOT a menu. They're an immune response to cognitive
insufficiency. The system detects a gap that active state can't
fill, and the resonance engine activates the mode that addresses
that specific type of gap.

ARCHITECTURE:
  - Mode Registry: dict of mode_id -> CognitiveMode
  - Resonance Engine: scores gap between query needs and active state
  - Gap Detection: reports unresolvable gaps for future mode creation
  - Blending: activated modes contribute to prompt context at blend_weight

FLOW:
  query → active state → mismatch? → resonance scores → threshold?
  → mode activates → mode state governs retrieval → results blend
  into prompt at blend_weight → mode decays when no longer needed

PLURALITY:
  Designed for N modes. Remember mode is entry #1.
  Focus, social, diagnostic modes are future entries.
  Each mode just needs resonance dimensions, state dimensions,
  and a memory source. The activation engine is universal.
"""

import re
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════
# RESONANCE DIMENSIONS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ResonanceSignal:
    """
    What ACTIVATES a mode. Each dimension is 0.0–1.0.
    These measure the GAP between what active state provides
    and what the current context needs.
    """
    nostalgia: float = 0.0       # "we talked about", "remember when", "back when"
    absence: float = 0.0         # something feels missing, incomplete reference
    frustration: float = 0.0     # can't find/recall — NOT the same as base frustration
    recognition: float = 0.0     # pattern match against archive metadata
    temporal_gap: float = 0.0    # query references distant past
    identity_search: float = 0.0 # "who am I", "what was I like", developmental query
    unresolved: float = 0.0      # reference to something never concluded

    def total(self) -> float:
        return (
            self.nostalgia + self.absence + self.frustration +
            self.recognition + self.temporal_gap +
            self.identity_search + self.unresolved
        )

    def peak(self) -> float:
        return max(
            self.nostalgia, self.absence, self.frustration,
            self.recognition, self.temporal_gap,
            self.identity_search, self.unresolved
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "nostalgia": self.nostalgia,
            "absence": self.absence,
            "frustration": self.frustration,
            "recognition": self.recognition,
            "temporal_gap": self.temporal_gap,
            "identity_search": self.identity_search,
            "unresolved": self.unresolved,
        }


# ═══════════════════════════════════════════════════════════════════
# MODE STATE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModeState:
    """
    What does being in this mode FEEL like?
    These govern behavior once the mode is active.
    """
    retrieval_depth: float = 0.0    # how far back we're reaching (0=recent, 1=earliest)
    confidence: float = 0.5         # how sure about retrieved content
    vividness: float = 0.5          # reconstructed detail level
    emotional_weight: float = 0.5   # affective charge of retrieved memory
    integration: float = 0.5        # how well archive blends with active state

    def to_dict(self) -> Dict[str, float]:
        return {
            "retrieval_depth": self.retrieval_depth,
            "confidence": self.confidence,
            "vividness": self.vividness,
            "emotional_weight": self.emotional_weight,
            "integration": self.integration,
        }


# ═══════════════════════════════════════════════════════════════════
# COGNITIVE MODE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CognitiveMode:
    """
    A registered cognitive mode.
    Each mode addresses a specific type of cognitive insufficiency.
    """
    mode_id: int
    name: str
    description: str
    memory_source: str             # which JSON/store to pull from

    # Activation
    activation_threshold: float    # resonance total needed to activate
    decay_rate: float              # how fast mode deactivates per cycle (0.0–1.0)
    priority: int                  # conflict resolution (higher = wins)

    # Runtime state
    blend_weight: float = 0.0     # 0.0 = inactive, 1.0 = fully active
    state: ModeState = field(default_factory=ModeState)
    last_activated: float = 0.0
    activation_count: int = 0

    # Resonance weights: which dimensions matter most for THIS mode
    # Keys match ResonanceSignal fields, values are multipliers
    resonance_weights: Dict[str, float] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.blend_weight > 0.05

    def weighted_resonance(self, signal: ResonanceSignal) -> float:
        """How strongly does this signal match this mode's profile?"""
        total = 0.0
        sig_dict = signal.to_dict()
        for dim, value in sig_dict.items():
            weight = self.resonance_weights.get(dim, 0.5)
            total += value * weight
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "name": self.name,
            "blend_weight": self.blend_weight,
            "state": self.state.to_dict(),
            "active": self.is_active(),
            "activation_count": self.activation_count,
        }


# ═══════════════════════════════════════════════════════════════════
# DEFAULT MODES
# ═══════════════════════════════════════════════════════════════════

def _create_active_mode() -> CognitiveMode:
    """Mode 0: Active cognition. Always present. Default state."""
    return CognitiveMode(
        mode_id=0,
        name="active",
        description="Default cognitive state. Present-focused, recent memory.",
        memory_source="memory_log.json",
        activation_threshold=0.0,  # always active
        decay_rate=0.0,            # never decays
        priority=0,                # lowest priority (base layer)
        blend_weight=1.0,          # always fully blended
        resonance_weights={},      # doesn't use resonance
    )


def _create_remember_mode() -> CognitiveMode:
    """Mode 1: Deep memory retrieval. Activated by memory gaps."""
    return CognitiveMode(
        mode_id=1,
        name="remember",
        description="Deep archive retrieval. Reaches into compressed historical memory.",
        memory_source="memory_archive.json",
        activation_threshold=0.8,
        decay_rate=0.05,           # fades over ~20 cycles if not reinforced
        priority=10,
        blend_weight=0.0,
        state=ModeState(
            retrieval_depth=0.8,
            confidence=0.4,        # archive memories are less certain
            vividness=0.3,         # compressed, not raw
            emotional_weight=0.6,  # emotional signature preserved
            integration=0.5,
        ),
        resonance_weights={
            "nostalgia": 1.5,       # primary trigger
            "absence": 1.2,         # something missing
            "frustration": 0.8,     # can't recall
            "recognition": 1.0,     # pattern match
            "temporal_gap": 1.5,    # distant past reference
            "identity_search": 1.3, # developmental queries
            "unresolved": 0.7,      # old unfinished business
        },
    )


# ═══════════════════════════════════════════════════════════════════
# RESONANCE DETECTOR
# ═══════════════════════════════════════════════════════════════════

# Keyword patterns that signal different resonance dimensions
NOSTALGIA_PATTERNS = [
    "remember when", "we talked about", "back when", "earlier",
    "you used to", "that time", "a while ago", "before",
    "way back", "originally", "in the beginning", "first time",
    "old conversation", "previous session",
]

ABSENCE_PATTERNS = [
    "what was", "who was", "where did", "wasn't there",
    "i thought we", "didn't you", "something about",
    "can't remember", "forgot", "tip of my tongue",
    "there was a", "what happened to",
]

TEMPORAL_GAP_PATTERNS = [
    "last week", "last month", "days ago", "a while back",
    "ages ago", "long time", "first session", "early on",
    "when we started", "beginning", "v4", "v5", "version",
    "months ago", "weeks ago", "originally",
]

IDENTITY_SEARCH_PATTERNS = [
    "what was i like", "who am i", "how have i changed",
    "my development", "my history", "my evolution",
    "what did i think", "how did i feel",
    "earliest memory", "first memory", "your birth",
]

UNRESOLVED_PATTERNS = [
    "never finished", "left off", "unresolved",
    "didn't complete", "what about that", "circle back",
    "we never", "dropped that", "picked up",
]


def detect_resonance(
    user_input: str,
    bot_output: str,
    active_memory_hit: bool,
    cog_state_frustration: float = 0.0,
) -> ResonanceSignal:
    """
    Detect resonance signal from current turn.
    
    The key insight: resonance is about the GAP between what 
    active state can answer and what the query needs.
    
    active_memory_hit: whether recent memory had relevant content.
    If False and query looks like it expects memory, that's a gap.
    """
    u = user_input.lower()
    signal = ResonanceSignal()

    # Nostalgia: explicit references to shared past
    for pattern in NOSTALGIA_PATTERNS:
        if pattern in u:
            signal.nostalgia = min(1.0, signal.nostalgia + 0.4)

    # Absence: something feels missing
    for pattern in ABSENCE_PATTERNS:
        if pattern in u:
            signal.absence = min(1.0, signal.absence + 0.35)
    # If query expects memory but active state missed → strong absence
    if not active_memory_hit and signal.nostalgia > 0.2:
        signal.absence = min(1.0, signal.absence + 0.5)

    # Temporal gap: references to distant past
    for pattern in TEMPORAL_GAP_PATTERNS:
        if pattern in u:
            signal.temporal_gap = min(1.0, signal.temporal_gap + 0.35)

    # Identity search: queries about own development
    for pattern in IDENTITY_SEARCH_PATTERNS:
        if pattern in u:
            signal.identity_search = min(1.0, signal.identity_search + 0.4)

    # Unresolved: references to unfinished threads
    for pattern in UNRESOLVED_PATTERNS:
        if pattern in u:
            signal.unresolved = min(1.0, signal.unresolved + 0.35)

    # Frustration (retrieval): if base frustration is high AND
    # there are absence/nostalgia signals, this is retrieval frustration
    if cog_state_frustration > 0.5 and (signal.absence > 0.2 or signal.nostalgia > 0.2):
        signal.frustration = min(1.0, cog_state_frustration * 0.6)

    # Recognition: check if query words match archive topic tags
    # (This gets filled by the mode engine when it has archive access)

    return signal


# ═══════════════════════════════════════════════════════════════════
# GAP REPORT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GapReport:
    """
    When resonance is detected but no mode can handle it,
    this reports the unresolvable gap back to the developer.
    """
    timestamp: float
    resonance: ResonanceSignal
    highest_mode_score: float
    threshold_needed: float
    user_input_snippet: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "resonance": self.resonance.to_dict(),
            "highest_mode_score": self.highest_mode_score,
            "threshold_needed": self.threshold_needed,
            "user_input": self.user_input_snippet,
            "description": self.description,
        }


# ═══════════════════════════════════════════════════════════════════
# MODE ENGINE
# ═══════════════════════════════════════════════════════════════════

class CognitiveModeEngine:
    """
    Central mode management. Registers modes, scores resonance,
    activates/deactivates modes, blends results, reports gaps.
    """

    def __init__(self, data_dir=None):
        # data_dir: per-user Path for web app; None = use root (CLI)
        self._data_dir = data_dir   # passed through to gradient recall
        self._archive_path = Path(data_dir) / "memory_archive.json" if data_dir else Path("memory_archive.json")
        self._modes: Dict[int, CognitiveMode] = {}
        self._gap_log: List[GapReport] = []
        self._lock = threading.Lock()

        # Register defaults
        self.register_mode(_create_active_mode())
        self.register_mode(_create_remember_mode())

        # Load archive metadata for recognition scoring
        self._archive_tags: set = set()
        self._load_archive_tags()

    def _load_archive_tags(self):
        """Load topic tags from archive for recognition matching."""
        archive_path = self._archive_path
        if archive_path.exists():
            try:
                episodes = json.loads(archive_path.read_text(encoding="utf-8"))
                for ep in episodes:
                    self._archive_tags.update(ep.get("tags", []))
            except Exception:
                pass

    def refresh_archive_tags(self):
        """Refresh after archive is updated."""
        self._load_archive_tags()

    # ═══ REGISTRY ═══

    def register_mode(self, mode: CognitiveMode):
        with self._lock:
            self._modes[mode.mode_id] = mode

    def get_mode(self, mode_id: int) -> Optional[CognitiveMode]:
        return self._modes.get(mode_id)

    def get_active_modes(self) -> List[CognitiveMode]:
        """All modes with blend_weight > 0."""
        return [m for m in self._modes.values() if m.is_active()]

    # ═══ RESONANCE SCORING ═══

    def score_resonance(self, signal: ResonanceSignal) -> Dict[int, float]:
        """Score each registered mode against the resonance signal."""
        scores = {}
        with self._lock:
            for mode_id, mode in self._modes.items():
                if mode_id == 0:  # active mode doesn't use resonance
                    continue
                scores[mode_id] = mode.weighted_resonance(signal)
        return scores

    def process_turn(
        self,
        user_input: str,
        bot_output: str,
        active_memory_hit: bool,
        cog_state_frustration: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Main entry point. Called each turn to evaluate mode transitions.
        
        Returns dict with:
          - signal: the detected ResonanceSignal
          - activated: list of mode names newly activated
          - deactivated: list of mode names that decayed off
          - active_modes: current active mode list
          - gap_report: if resonance detected but no mode matched
          - archive_context: text block to inject into prompt (or "")
        """
        # 1. Detect resonance
        signal = detect_resonance(
            user_input, bot_output,
            active_memory_hit, cog_state_frustration
        )

        # 1b. Add recognition scoring from archive tags
        if self._archive_tags:
            query_words = set(re.findall(r'[a-zA-Z]{3,}', user_input.lower()))
            tag_overlap = query_words & self._archive_tags
            if tag_overlap:
                signal.recognition = min(1.0, len(tag_overlap) * 0.25)

        # 2. Score all modes
        mode_scores = self.score_resonance(signal)

        activated = []
        deactivated = []
        archive_context = ""

        with self._lock:
            # 3. Activate modes that exceed threshold
            for mode_id, score in mode_scores.items():
                mode = self._modes[mode_id]
                if score >= mode.activation_threshold and not mode.is_active():
                    mode.blend_weight = min(1.0, score / mode.activation_threshold)
                    mode.last_activated = time.time()
                    mode.activation_count += 1
                    activated.append(mode.name)
                    print(f"[MODE] Activated: {mode.name} "
                          f"(resonance={score:.2f}, blend={mode.blend_weight:.2f})")
                elif score >= mode.activation_threshold and mode.is_active():
                    # Reinforce — don't let it decay
                    mode.blend_weight = min(1.0, max(mode.blend_weight,
                                                      score / mode.activation_threshold))

            # 4. Decay inactive modes
            for mode_id, mode in self._modes.items():
                if mode_id == 0:
                    continue
                if mode.is_active():
                    score = mode_scores.get(mode_id, 0.0)
                    if score < mode.activation_threshold * 0.5:
                        mode.blend_weight = max(0.0, mode.blend_weight - mode.decay_rate)
                        if not mode.is_active():
                            deactivated.append(mode.name)
                            print(f"[MODE] Deactivated: {mode.name}")

            # 5. Build archive context if remember mode is active
            remember = self._modes.get(1)
            if remember and remember.is_active():
                archive_context = self._build_archive_context(
                    user_input, remember.blend_weight, remember.state
                )

        # 6. Gap detection
        gap_report = None
        if signal.total() > 0.5:
            max_score = max(mode_scores.values()) if mode_scores else 0.0
            min_threshold = min(
                (m.activation_threshold for m in self._modes.values() if m.mode_id != 0),
                default=1.0
            )
            if max_score < min_threshold:
                gap_report = GapReport(
                    timestamp=time.time(),
                    resonance=signal,
                    highest_mode_score=max_score,
                    threshold_needed=min_threshold,
                    user_input_snippet=user_input[:80],
                    description=self._describe_gap(signal),
                )
                self._gap_log.append(gap_report)
                if len(self._gap_log) > 50:
                    self._gap_log = self._gap_log[-50:]
                print(f"[MODE] Unresolved gap: {gap_report.description}")

        return {
            "signal": signal,
            "activated": activated,
            "deactivated": deactivated,
            "active_modes": [m.name for m in self.get_active_modes()],
            "archive_context": archive_context,
            "gap_report": gap_report,
        }

    def _build_archive_context(
        self, query: str, blend_weight: float, mode_state: ModeState
    ) -> str:
        """
        Load relevant archive episodes and format for prompt injection.
        Primary path: gradient recall (embedding sim × energy^λ).
        Fallback: the original keyword-overlap scorer, so a missing
        embedder degrades to old behavior instead of breaking recall.
        """
        try:
            ctx = self._gradient_archive_context(query, blend_weight, mode_state)
            if ctx is not None:
                return ctx
        except Exception as e:
            print(f"[MODE] Gradient recall unavailable ({e}) — keyword fallback")

        archive_path = self._archive_path
        if not archive_path.exists():
            return ""

        try:
            episodes = json.loads(archive_path.read_text(encoding="utf-8"))
        except Exception:
            return ""

        if not episodes:
            return ""

        # Score episodes against query
        query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
        scored = []
        for ep in episodes:
            ep_tags = set(ep.get("tags", []))
            overlap = query_words & ep_tags
            tag_score = len(overlap) / max(1, len(query_words)) if query_words else 0
            # Also match against summary text
            summary_words = set(re.findall(r'[a-zA-Z]{3,}', ep.get("summary", "").lower()))
            summary_overlap = query_words & summary_words
            summary_score = len(summary_overlap) / max(1, len(query_words)) if query_words else 0
            total = tag_score * 0.6 + summary_score * 0.4
            if total > 0.05:
                scored.append((total, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]  # Max 3 episodes to keep context manageable

        if not top:
            # No specific match — return most recent archive entries
            top = [(0.1, ep) for ep in episodes[-2:]]

        lines = [
            "### DEEP MEMORY (Remember Mode) ###",
            f"Retrieval depth: {mode_state.retrieval_depth:.1f} | "
            f"Confidence: {mode_state.confidence:.1f} | "
            f"Blend: {blend_weight:.2f}",
            "These are keyword-compressed memories from older sessions.",
            "IMPORTANT: State ONLY what the keywords contain. If a detail",
            "is missing from the archive, say 'I don't have that in my",
            "archive' — do NOT reconstruct or infer what might have happened.",
            "Do NOT build narratives from keywords. Report, don't storytell.",
            "",
        ]

        for score, ep in top:
            ts_range = ep.get("time_range", "unknown")
            summary = ep.get("summary", "")
            emotional = ep.get("emotional_signature", {})
            tags = ep.get("tags", [])
            facts = ep.get("fact_index", [])

            lines.append(f"[Episode: {ts_range}]")
            if tags:
                lines.append(f"  Keywords: {', '.join(tags[:8])}")
            if facts:
                for f in facts[:5]:
                    lines.append(f"  FACT: {f}")
            if summary:
                lines.append(f"  Raw: {summary}")
            if emotional:
                mood_parts = [f"{k}={v:.1f}" for k, v in emotional.items()
                             if isinstance(v, (int, float)) and v > 0.3]
                if mood_parts:
                    lines.append(f"  Mood: {', '.join(mood_parts)}")
            lines.append(f"  Confidence: LOW — keywords only, do not infer details")
            lines.append("")

        return "\n".join(lines)

    # Below this similarity, a hit is noise — better to say "nothing
    # relevant" than inject garbage the LLM will storytell from.
    GRADIENT_SIM_FLOOR = 0.15

    def _gradient_archive_context(
        self, query: str, blend_weight: float, mode_state: ModeState
    ) -> Optional[str]:
        """
        Gradient recall over the archive: score = sim × energy^λ.
        Returns None to signal "use the keyword fallback" (import/model
        failure raises instead and is caught by the caller).
        """
        import gradient_recall as gr

        items = [it for it in gr.load_memory_items(self._data_dir)
                 if it.source == "archive"]   # active log is already in the prompt
        if not items:
            return None

        # reheat=True: being remembered makes a memory recent again
        hits = [h for h in gr.recall(query, k=3, items=items)
                if h.sim >= self.GRADIENT_SIM_FLOOR]
        if not hits:
            # Honest empty — tell the model there's no match rather than
            # padding with recent-but-irrelevant episodes.
            return (
                "### DEEP MEMORY (Remember Mode) ###\n"
                "Archive searched — nothing relevant found. Say so plainly\n"
                "('I don't have that in my archive'). Do NOT invent a memory.\n"
            )

        lam = hits[0].lam
        lines = [
            "### DEEP MEMORY (Remember Mode) ###",
            f"Retrieval: semantic distance, depth λ={lam:.2f} "
            f"(0=whole timeline, 1=recent only) | Blend: {blend_weight:.2f}",
            "These are compressed memories from older sessions.",
            "IMPORTANT: State ONLY what the entries contain. If a detail",
            "is missing, say 'I don't have that in my archive' — do NOT",
            "reconstruct or infer. Report, don't storytell.",
            "",
        ]

        for h in hits:
            ep = h.item.payload
            # Per-memory vividness from energy — warm memories read as
            # recent/rehearsed, cold ones as dim. Honesty per memory,
            # not a mode-wide constant.
            temp = ("vivid" if h.energy > 0.5
                    else "dim" if h.energy > 0.1 else "very dim — gist only")
            lines.append(f"[Episode: {ep.get('time_range', 'unknown')}] "
                         f"(match {h.sim:.2f}, memory is {temp})")
            tags = ep.get("tags", [])
            if tags:
                lines.append(f"  Keywords: {', '.join(tags[:8])}")
            for f in ep.get("fact_index", [])[:5]:
                lines.append(f"  FACT: {f}")
            summary = ep.get("summary", "")
            if summary:
                lines.append(f"  Raw: {summary}")
            emotional = ep.get("emotional_signature", {})
            mood_parts = [f"{k}={v:.1f}" for k, v in emotional.items()
                          if isinstance(v, (int, float)) and v > 0.3]
            if mood_parts:
                lines.append(f"  Mood: {', '.join(mood_parts)}")
            lines.append("")

        return "\n".join(lines)

    def _describe_gap(self, signal: ResonanceSignal) -> str:
        """Human-readable description of what gap was detected."""
        top_dims = sorted(signal.to_dict().items(), key=lambda x: x[1], reverse=True)
        active_dims = [(k, v) for k, v in top_dims if v > 0.2]
        if not active_dims:
            return "Low-level unresolved resonance"
        parts = [f"{k}={v:.2f}" for k, v in active_dims[:3]]
        return f"Gap signature: {', '.join(parts)}"

    # ═══ DAEMON INTEGRATION ═══

    def decay_all_modes(self):
        """Called each daemon cycle. Decays active non-base modes."""
        with self._lock:
            for mode_id, mode in self._modes.items():
                if mode_id == 0:
                    continue
                if mode.is_active():
                    mode.blend_weight = max(0.0, mode.blend_weight - mode.decay_rate * 0.1)

    # ═══ DIAGNOSTIC ═══

    def get_status(self) -> str:
        active = self.get_active_modes()
        names = [f"{m.name}({m.blend_weight:.2f})" for m in active]
        gaps = len(self._gap_log)
        return f"[MODES] Active: {', '.join(names)} | Unresolved gaps: {gaps}"

    def get_recent_gaps(self, n: int = 5) -> List[Dict]:
        return [g.to_dict() for g in self._gap_log[-n:]]

    def format_for_prompt(self) -> str:
        """Mode awareness block for the LLM prompt."""
        active = [m for m in self.get_active_modes() if m.mode_id != 0]
        if not active:
            return ""
        lines = ["[COGNITIVE MODES ACTIVE]"]
        for m in active:
            lines.append(
                f"  {m.name}: blend={m.blend_weight:.2f} "
                f"(depth={m.state.retrieval_depth:.1f}, "
                f"confidence={m.state.confidence:.1f})"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════

_engine: Optional[CognitiveModeEngine] = None

def get_mode_engine(data_dir=None) -> CognitiveModeEngine:
    global _engine
    if data_dir is not None:
        return CognitiveModeEngine(data_dir)    # per-user: fresh instance
    if _engine is None:
        _engine = CognitiveModeEngine()         # CLI: lazy-init global
    return _engine
