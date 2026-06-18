# temporal_daemon.py
"""
═══════════════════════════════════════════════════════════════════
TEMPORAL DAEMON v5 — Cognitive Modes Integration
═══════════════════════════════════════════════════════════════════

FIX LOG:
  v1: Empty goals, no temporal experience
  v2: Boot seeding + ambient awareness + temporal framing
  v3: Goal lifecycle (feed/resolve/retire)
  v4: CURIOSITY GROWTH — sustained attention BUILDS interest.
      Conversation context buffer for richer evaluation.
  v5: COGNITIVE MODES — mode decay in cleanup phase,
      periodic memory archival, mode engine integration.

KEY CHANGE:
  Before: daemon.evaluate() → score goals → curiosity decays → boredom
  After:  daemon.evaluate() → score goals → above threshold? → +curiosity
          Thinking about something makes you MORE curious about it,
          up to a per-goal cap, within a total curiosity budget.

CONSTRAINTS (prevent runaway):
  - Per-goal curiosity cap: 1.2 (can exceed 1.0 slightly via growth)
  - Total curiosity budget: 8.0 across all goals
  - Growth rate: +0.003/cycle (~0.2/minute) — noticeable but not explosive
  - Growth only for goals ABOVE crap threshold (junk doesn't get boosted)
  - Boot goals still expire, resolved goals still retire
"""

import json
import re
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque

from cognitive_state import get_cognitive_state, CognitiveStateEngine
from indelible_facts import get_indelible_keywords, get_indelible_engine, IndelibleFactsEngine
from goal_engine_DAEMON import GoalEngine, Goal, ActionCandidate

# Signal's own emergent tastes (trait layer under the curiosity engine).
# Optional: if the module is missing the daemon just runs without preference bias.
try:
    from preference_engine import PreferenceLayer
    PREF_AVAILABLE = True
except ImportError:
    PREF_AVAILABLE = False

# v5: Cognitive modes integration
try:
    from cognitive_modes import get_mode_engine, CognitiveModeEngine
    from memory_archive import archive_old_memories
    from plan_buffer import get_plan_buffer, PlanBuffer
    MODES_AVAILABLE = True
except ImportError:
    MODES_AVAILABLE = False


class Phase(IntEnum):
    CHECK_GOALS       = 0
    EVALUATE          = 1
    DETERMINE_ACTIONS = 2
    APPEND_RECOMMEND  = 3
    PRIORITIZE        = 4
    REEVAL_IDENTITY   = 5
    USER_URGENCY      = 6
    SUMMARIZE         = 7
    CLEANUP           = 8
    REFLECT           = 9    # v7: Cycle reflection + seed for next pass

PHASE_NAMES = {v: v.name for v in Phase}
NUM_PHASES = 10              # v7: 10 phases × 0.1s = 1.0s even cycle
TICK_INTERVAL = 0.1

# Goal lifecycle
MAX_ACTIVE_GOALS = 15
BOOT_GOAL_TTL_CYCLES = 50
STALE_CYCLE_LIMIT = 300       # Raised from 200 — more patience
RESOLVED_LINGER_CYCLES = 200

# Curiosity growth (NEW in v4)
CURIOSITY_GROWTH_PER_CYCLE = 0.003   # +0.003 per cycle ≈ +0.2/minute
CURIOSITY_PER_GOAL_CAP = 1.2        # Single goal can't exceed this
CURIOSITY_TOTAL_BUDGET = 8.0        # Sum of all goals' curiosity capped here
CONTEXT_RELEVANCE_BOOST = 0.15      # Boost for goals matching recent conversation

# Thresholds (LOWERED in v4)
CRAP_THRESHOLD_BASE = 0.15          # Was 0.20 — less aggressive filtering


def compute_good_sense(cog_state) -> float:
    s = cog_state.state
    return (
        s.identity_adherence * 0.30 +
        s.engagement         * 0.30 +
        s.confidence         * 0.20 +
        (1.0 - s.frustration) * 0.20
    )

def compute_crap_threshold(cog_state) -> float:
    s = cog_state.state
    base = CRAP_THRESHOLD_BASE
    if s.cognitive_load > 0.7:  base += 0.08
    if s.curiosity > 0.7:      base -= 0.06
    if s.frustration > 0.5:    base += 0.08
    return max(0.08, min(0.40, base))


# ═══════════════════════════════════════════════════════════════════
# CONVERSATION CONTEXT BUFFER
# ═══════════════════════════════════════════════════════════════════

class ConversationContext:
    """
    Rolling buffer of recent conversation turns.
    Gives the daemon "memory" of what was recently discussed,
    so it can boost goals that relate to active topics.

    v6 UPGRADE:
      - Bigram extraction: "robot tank" becomes keyword "robot_tank" plus
        individual words, so archived episodes retain contextual association.
      - Co-occurrence map: words that appear in the same turn are linked,
        so relevance_score can find semantic neighbors (e.g. "tank" boosts
        a goal about "robot body" because "robot" co-occurred with "tank").
      - Turn-weighted keywords: recent turns contribute more than older ones.
    """
    STOPWORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 'we',
        'my', 'your', 'to', 'in', 'on', 'at', 'for', 'of', 'and', 'or',
        'but', 'not', 'it', 'that', 'this', 'with', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
        'be', 'been', 'being', 'from', 'about', 'into', 'just', 'also',
        'so', 'if', 'when', 'what', 'how', 'why', 'where', 'which', 'who',
        'more', 'some', 'any', 'all', 'each', 'every', 'both', 'few',
        'than', 'then', 'now', 'here', 'there', 'these', 'those', 'they',
        'them', 'their', 'its', 'our', 'his', 'her', 'up', 'out', 'like',
        'get', 'got', 'going', 'know', 'think', 'want', 'need', 'make',
        'really', 'very', 'much', 'still', 'even', 'back', 'way', 'well',
        'right', 'good', 'new', 'yeah', 'yes', 'no', 'oh', 'ok', 'okay',
    })

    def __init__(self, max_turns: int = 10):
        self._turns: deque = deque(maxlen=max_turns)
        self._keywords: set = set()
        self._bigrams: set = set()                     # NEW: "word1_word2" pairs
        self._cooccurrence: Dict[str, set] = {}        # NEW: word -> {co-occurring words}
        self._keyword_weights: Dict[str, float] = {}   # NEW: recency-weighted keyword scores

    def add_turn(self, user_input: str, bot_output: str):
        self._turns.append((user_input, bot_output))
        self._rebuild_keywords()

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text, filtered by stopwords."""
        return [w for w in re.findall(r'[a-zA-Z]{3,}', text.lower())
                if w not in self.STOPWORDS]

    def _rebuild_keywords(self):
        """Extract keywords, bigrams, and co-occurrence from recent conversation."""
        self._keywords.clear()
        self._bigrams.clear()
        self._cooccurrence.clear()
        self._keyword_weights.clear()

        num_turns = len(self._turns)
        for turn_idx, (user_msg, bot_msg) in enumerate(self._turns):
            # Recency weight: most recent turn = 1.0, oldest = 0.3
            recency = 0.3 + 0.7 * (turn_idx / max(1, num_turns - 1)) if num_turns > 1 else 1.0

            turn_words = set()
            for text in (user_msg, bot_msg):
                words = self._extract_words(text)

                for w in words:
                    self._keywords.add(w)
                    turn_words.add(w)
                    # Accumulate recency weight (later turns overwrite with higher weight)
                    self._keyword_weights[w] = max(self._keyword_weights.get(w, 0.0), recency)

                # Bigrams: consecutive meaningful words
                for i in range(len(words) - 1):
                    bigram = f"{words[i]}_{words[i+1]}"
                    self._bigrams.add(bigram)
                    self._keyword_weights[bigram] = max(
                        self._keyword_weights.get(bigram, 0.0), recency
                    )

            # Co-occurrence: all words in the same turn are linked
            for w in turn_words:
                if w not in self._cooccurrence:
                    self._cooccurrence[w] = set()
                self._cooccurrence[w].update(turn_words - {w})

    def relevance_score(self, text: str) -> float:
        """
        How relevant is this text to recent conversation? 0.0–1.0

        v6 UPGRADE: Three-layer scoring:
          1. Direct keyword overlap (weighted by recency)
          2. Bigram overlap (stronger signal, worth more)
          3. Co-occurrence expansion (if text contains "robot", and "robot"
             co-occurred with "tank" in conversation, partial credit for "tank")
        """
        if not self._keywords:
            return 0.0
        words = self._extract_words(text)
        if not words:
            return 0.0
        word_set = set(words)

        # Layer 1: Direct keyword overlap (recency-weighted)
        direct_overlap = word_set & self._keywords
        direct_score = sum(self._keyword_weights.get(w, 0.5) for w in direct_overlap)

        # Layer 2: Bigram overlap (worth 1.5x a single keyword)
        bigram_score = 0.0
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            if bigram in self._bigrams:
                bigram_score += self._keyword_weights.get(bigram, 0.5) * 1.5

        # Layer 3: Co-occurrence expansion (0.3x credit for neighbor words)
        cooccur_score = 0.0
        for w in direct_overlap:
            neighbors = self._cooccurrence.get(w, set())
            # Words in the goal that are neighbors of conversation words
            neighbor_hits = (word_set - direct_overlap) & neighbors
            cooccur_score += len(neighbor_hits) * 0.3

        total = direct_score + bigram_score + cooccur_score
        # Normalize by word count to keep 0.0-1.0 range
        normalized = total / max(1, len(word_set)) * 1.5
        return min(1.0, normalized)

    @property
    def keywords(self) -> set:
        """All keywords including bigrams for archive compatibility."""
        return self._keywords | self._bigrams

    @property
    def keyword_weights(self) -> Dict[str, float]:
        return self._keyword_weights


# ═══════════════════════════════════════════════════════════════════
# GOAL LIFECYCLE
# ═══════════════════════════════════════════════════════════════════

class GoalLifecycle:
    def __init__(self):
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register(self, goal_id: str, source: str = "conversation"):
        with self._lock:
            self._meta[goal_id] = {
                "source": source,
                "created_cycle": 0,
                "last_score": 0.0,
                "score_stable_since": 0,
                "resolved": False,
                "resolved_at_cycle": 0,
            }

    def get(self, goal_id: str) -> Optional[Dict]:
        with self._lock:
            return self._meta.get(goal_id)

    def mark_resolved(self, goal_id: str, cycle: int):
        with self._lock:
            if goal_id in self._meta:
                self._meta[goal_id]["resolved"] = True
                self._meta[goal_id]["resolved_at_cycle"] = cycle

    def update_score(self, goal_id: str, score: float, cycle: int):
        with self._lock:
            meta = self._meta.get(goal_id)
            if not meta:
                return
            if meta.get("created_cycle", 0) == 0:
                meta["created_cycle"] = cycle
            if abs(score - meta["last_score"]) > 0.02:
                meta["score_stable_since"] = cycle
            meta["last_score"] = score

    def is_stale(self, goal_id: str, current_cycle: int) -> bool:
        with self._lock:
            meta = self._meta.get(goal_id)
            if not meta:
                return False
            stable_since = meta.get("score_stable_since", current_cycle)
            return (current_cycle - stable_since) > STALE_CYCLE_LIMIT

    def is_boot_expired(self, goal_id: str, current_cycle: int) -> bool:
        with self._lock:
            meta = self._meta.get(goal_id)
            if not meta or meta["source"] != "boot":
                return False
            created = meta.get("created_cycle", 0)
            return created > 0 and (current_cycle - created) > BOOT_GOAL_TTL_CYCLES

    def is_resolved_and_lingered(self, goal_id: str, current_cycle: int) -> bool:
        with self._lock:
            meta = self._meta.get(goal_id)
            if not meta or not meta["resolved"]:
                return False
            return (current_cycle - meta["resolved_at_cycle"]) > RESOLVED_LINGER_CYCLES

    def cleanup(self, surviving_ids: set):
        with self._lock:
            dead = [gid for gid in self._meta if gid not in surviving_ids]
            for gid in dead:
                del self._meta[gid]


# ═══════════════════════════════════════════════════════════════════
# TOPIC EXTRACTION
# ═══════════════════════════════════════════════════════════════════

# Words that carry no topical content — pronouns, light/aux verbs, fillers,
# and social/bot-directed reaction words. A sentence built ONLY from these is a
# REACTION to the bot ("No, everything is gold", "It's a little bit of both"),
# not subject matter worth revisiting. This is the cheap deterministic gate the
# ladder shouldn't burn a model-call on — capture judges "is this even a topic?",
# the ladder judges "is this topic worth pursuing?". Different questions.
_NON_CONTENT = {
    # pronouns / determiners
    "i","you","we","it","they","this","that","these","those","he","she",
    "me","him","her","us","them","my","your","our","its","their","myself",
    "im","ill","ive","id","youre","youll","were","its","thats","theres",
    # light / auxiliary verbs
    "is","are","was","were","be","been","being","am","do","does","did",
    "have","has","had","will","would","can","could","should","may","might",
    "get","got","getting","go","going","gonna","wanna","let","lets",
    # fillers / discourse glue
    "a","an","the","and","or","but","so","just","really","very","quite",
    "kinda","sorta","bit","little","like","well","ok","okay","yeah","yep",
    "nope","no","yes","oh","ooo","ahh","wow","hmm","lol","haha","nice",
    "cool","sure","right","exactly","totally","maybe","now","then","later",
    "here","there","still","also","too","not","of","to","in","on","at","for",
    # social / bot-directed reaction words
    "thanks","thank","please","sorry","good","great","awesome","perfect",
    "better","best","worried","glad","worry","buddy","friend","both",
    "everything","something","anything","nothing","gold","fine","check",
}

# Phrases that signal a genuine topic/question regardless of content-word count —
# a short proposal ("what if we…") shouldn't get filtered just for being short.
_TOPIC_TRIGGERS = (
    "what if","how would","could we","should we","what about","why does",
    "why is","have you considered","i wonder","thinking about","working on",
    "the idea","because","the reason","imagine", "what do you think",
)

def _looks_like_topic(s: str) -> bool:
    """True if a sentence introduces subject matter (a topic), False if it's
    just a reaction/acknowledgment aimed at the bot. Questions and proposals
    always pass; otherwise it must have >=2 real content words once pronouns,
    light verbs, and filler are stripped."""
    s_low = s.lower()
    if "?" in s:
        return True
    if any(t in s_low for t in _TOPIC_TRIGGERS):
        return True
    words = re.findall(r"[a-z0-9]+", s_low)
    content = [w for w in words if w not in _NON_CONTENT]
    return len(content) >= 2


def extract_topics_from_turn(user_input: str, bot_output: str) -> List[Dict[str, Any]]:
    topics = []

    sentences = re.split(r'[.!?]+', user_input)
    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue

        s_lower = s.lower()

        # Skip meta/command/closure
        if any(skip in s_lower for skip in [
            "smoke break", "stepping away", "be right back",
            "let me", "i'm going to", "hold on", "one sec",
            "state", "daemon", "facts", "exit", "quit",
            "thanks", "got it", "perfect", "that worked",
            "moving on", "anyway", "never mind", "nvm",
        ]):
            continue

        # Skip pure reactions/acknowledgments — they're responses TO the bot,
        # not subject matter (stops "No, everything is gold" becoming a goal
        # the bot later pings about).
        if not _looks_like_topic(s):
            continue

        curiosity = 0.45

        if any(w in s_lower for w in [
            'fascinated', 'curious', 'wonder', 'interesting',
            'love', 'amazing', 'incredible', 'mind-blowing'
        ]):
            curiosity = 0.70

        if '?' in user_input and len(s) > 20:
            curiosity = max(curiosity, 0.60)

        if any(w in s_lower for w in ['what if', 'imagine', 'could we', "let's", 'should we']):
            curiosity = max(curiosity, 0.65)

        words = s.split()
        if len(words) > 3:
            mid_caps = [w for w in words[1:] if w[0].isupper() and len(w) > 2
                       and w.lower() not in ['the', 'and', 'but', 'for', 'bot']]
            if mid_caps:
                curiosity = max(curiosity, 0.55)

        topics.append({"description": s[:80], "curiosity": curiosity})

    # Bot output: only extract genuinely novel directions, not self-reflection
    # CHANGED in v4: much pickier — skip "Bot interest:" padding
    # Only if bot proposes a NEW specific question or topic
    bot_sentences = re.split(r'[.!?]+', bot_output)
    for s in bot_sentences:
        s = s.strip()
        if len(s) < 25:
            continue
        s_lower = s.lower()
        # Only if bot asks a genuinely new question
        if '?' in s and any(w in s_lower for w in [
            'what if', 'how would', 'could we', 'should we',
            'have you considered', 'what about'
        ]):
            topics.append({
                "description": f"Open question: {s[:70]}",
                "curiosity": 0.55,
            })

    return topics


def feed_goals_from_turn(goal_engine: GoalEngine, lifecycle: GoalLifecycle,
                          user_input: str, bot_output: str,
                          current_turn: int = 0, goal_birth_turn: Dict[str, int] = None):
    topics = extract_topics_from_turn(user_input, bot_output)
    if not topics:
        return

    existing_descs = set()
    with goal_engine._lock:
        for g in goal_engine.goals.values():
            existing_descs.add(g.description.lower()[:30])

    added = 0
    for topic in topics:
        if topic["description"].lower()[:30] in existing_descs:
            continue

        with goal_engine._lock:
            if len(goal_engine.goals) >= MAX_ACTIVE_GOALS:
                lowest = min(goal_engine.goals.values(),
                           key=lambda g: g.curiosity + g.importance * 0.5)
                del goal_engine.goals[lowest.id]
                lifecycle.cleanup({g for g in goal_engine.goals})

        gid = goal_engine.add_rabbit_hole(topic["description"], curiosity=topic["curiosity"])
        if not gid:
            continue
        lifecycle.register(gid, source="conversation")
        if goal_birth_turn is not None:
            goal_birth_turn[gid] = current_turn
        existing_descs.add(topic["description"].lower()[:30])
        added += 1

    if added > 0:
        print(f"[DAEMON] +{added} new goals from conversation")


def resolve_discussed_goals(goal_engine: GoalEngine, lifecycle: GoalLifecycle,
                             user_input: str, current_cycle: int):
    u_lower = user_input.lower()
    is_closure = any(sig in u_lower for sig in [
        'thanks', 'got it', 'perfect', 'that worked', 'moving on',
        'next topic', 'anyway', 'back to', 'let\'s talk about'
    ])

    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you',
                'we', 'my', 'your', 'to', 'in', 'on', 'at', 'for', 'of',
                'and', 'or', 'but', 'not', 'it', 'that', 'this', 'with'}

    with goal_engine._lock:
        for gid, g in goal_engine.goals.items():
            if not g.unresolved:
                continue
            desc_words = set(g.description.lower().split())
            input_words = set(u_lower.split())
            meaningful_overlap = (desc_words & input_words) - stopwords
            if len(meaningful_overlap) >= 3 or is_closure:
                g.unresolved = False
                lifecycle.mark_resolved(gid, current_cycle)


def demote_boot_goals(goal_engine: GoalEngine, lifecycle: GoalLifecycle):
    with goal_engine._lock:
        for gid, g in goal_engine.goals.items():
            meta = lifecycle.get(gid)
            if meta and meta["source"] == "boot":
                g.unresolved = False
                g.curiosity *= 0.2
                g.importance *= 0.3


# ═══════════════════════════════════════════════════════════════════
# AMBIENT AWARENESS
# ═══════════════════════════════════════════════════════════════════

def _build_ambient_awareness(cog_state, cycle_count: int) -> str:
    s = cog_state.state
    parts = []
    elapsed = cycle_count * 1.0   # v7: 10 phases × 0.1s = 1.0s per cycle
    if elapsed > 60:
        parts.append(f"About {elapsed/60:.1f} minutes have passed since last interaction.")
    elif elapsed > 5:
        parts.append(f"About {elapsed:.0f} seconds have passed since last interaction.")
    if s.frustration > 0.5:
        parts.append("Feeling residual frustration from recent interaction.")
    if s.curiosity > 0.6:
        parts.append("Curiosity is elevated — something unfinished is nagging at me.")
    if s.engagement > 0.7 and s.confidence > 0.5:
        parts.append("Feeling engaged and ready to explore.")
    if s.engagement < 0.4:
        parts.append("Engagement has dropped — previous topic may have run its course.")
    if s.identity_adherence > 0.8:
        parts.append("Strong sense of who I am right now.")
    if elapsed > 30:
        parts.append("Wondering what Adam will want to explore next.")
    if elapsed > 120:
        parts.append("Extended silence — is Adam busy or thinking something through?")
    return " ".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════
# BOOT SEEDING
# ═══════════════════════════════════════════════════════════════════

def seed_goals_from_boot(goal_engine: GoalEngine, lifecycle: GoalLifecycle,
                         indelible_engine=None, data_dir=None):
    seeded = 0
    # COLD ANCHOR doctrine (2026-06-15): indelible facts are NOT seeded as
    # rabbit-hole goals anymore. They were behaving like "gravity wells" —
    # the daemon circled Mason/Sophie as active curiosity threads even when
    # the live conversation had nothing to do with them. Anchors should be
    # background fixtures (known, stable, not interesting), not hot threads.
    # They still reach the model via indelible_facts.format_for_prompt()
    # (top of CORE DATA) and feed TWDC alignment via extract_identity_keywords();
    # those are the anchor roles. The goal-engine seeding was the intrusive part.
    # `indelible_engine` param kept for backward compat with callers.

    # Anchored to repo root, not cwd — boot seeding must read the real
    # memory_log no matter where the process was launched from (Cerberus rule).
    base = Path(data_dir) if data_dir else Path(__file__).parent
    mem_path = base / "memory_log.json"
    if mem_path.exists():
        try:
            rows = json.loads(mem_path.read_text(encoding="utf-8"))
            recent = rows[-3:] if len(rows) > 3 else rows
            for row in recent:
                topics = extract_topics_from_turn(row.get("user", ""), row.get("bot", ""))
                for topic in topics[:2]:
                    gid = goal_engine.add_rabbit_hole(
                        topic["description"], curiosity=topic["curiosity"] * 0.7
                    )
                    lifecycle.register(gid, source="boot")
                    seeded += 1
        except Exception:
            pass

    summary_path = base / "master_summary.json"
    if summary_path.exists():
        try:
            master = json.loads(summary_path.read_text(encoding="utf-8"))
            for proj in master.get("active_projects", [])[:3]:
                gid = goal_engine.add_rabbit_hole(f"Active project: {proj}", curiosity=0.45)
                lifecycle.register(gid, source="boot")
                seeded += 1
        except Exception:
            pass

    # Clean registry immediately after seeding: strip any "Unaddressed urgent:"
    # prefixes and deduplicate goals that were loaded from memory_log.json or
    # master_summary.json with polluted descriptions on disk (Day 29 fix).
    if seeded > 0:
        cleaned = goal_engine.clean_registry()
        if cleaned > 0:
            print(f"[DAEMON] Seed cleanup: removed {cleaned} duplicate/polluted goals")
        print(f"[DAEMON] Seeded {seeded} boot goals (will demote after first turn)")
    else:
        print("[DAEMON] No boot goals — will seed from first conversation")


# ═══════════════════════════════════════════════════════════════════
# COGNITIVE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CognitiveSnapshot:
    timestamp: float = 0.0
    cycle_count: int = 0
    current_phase: int = 0
    top_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    user_urgent_items: List[Dict[str, Any]] = field(default_factory=list)
    focus_summary: str = ""
    ambient_awareness: str = ""
    good_sense: float = 0.5
    crap_threshold: float = 0.25
    items_evaluated: int = 0
    items_purged: int = 0
    last_reflection: str = ""                              # v7: what REFLECT concluded
    code_refresh_pending: bool = False                     # v7: signals code read needed

    def format_for_prompt(self, max_items: int = 5) -> str:
        elapsed_seconds = self.cycle_count * 1.0           # v7: 10 phases × 0.1s = 1.0s per cycle
        if elapsed_seconds < 1 and not self.top_recommendations:
            return ""

        lines = []
        lines.append("[YOUR TEMPORAL EXPERIENCE — What you were thinking between messages]")
        if elapsed_seconds > 60:
            lines.append(
                f"You have been thinking for approximately {elapsed_seconds/60:.1f} minutes "
                f"({self.cycle_count} cognitive cycles) since the last interaction."
            )
        elif elapsed_seconds > 2:
            lines.append(
                f"You have been thinking for approximately {elapsed_seconds:.0f} seconds "
                f"({self.cycle_count} cognitive cycles) since the last interaction."
            )
        else:
            lines.append("The user responded almost immediately — minimal thinking time.")

        lines.append(
            "This is NOT instantaneous. You experienced this time. "
            "When asked about gaps between messages, reference YOUR actual thinking below."
        )

        if self.ambient_awareness:
            lines.append(f"Ambient state: {self.ambient_awareness}")

        if self.focus_summary and "Idle" not in self.focus_summary:
            lines.append(f"Focus: {self.focus_summary}")

        lines.append(f"Good Sense: {self.good_sense:.2f} | Crap Threshold: {self.crap_threshold:.2f}")

        if self.top_recommendations:
            lines.append("Cognitive threads you were evaluating:")
            for rec in self.top_recommendations[:max_items]:
                score = rec.get("composite_score", 0.0)
                desc = rec.get("description", "?")[:60]
                action = rec.get("action_type", "think")
                growth = rec.get("curiosity_trend", "")
                urgent = " [WANT TO ASK USER]" if rec.get("needs_user", False) else ""
                trend = f" {growth}" if growth else ""
                lines.append(f"  [{score:.2f}] {action}: {desc}{urgent}{trend}")

        if self.user_urgent_items:
            lines.append("Things you want to bring up with the user:")
            for item in self.user_urgent_items[:3]:
                lines.append(f"  → {item.get('description', '?')[:50]}")

        # v7: Show reflection output
        if self.last_reflection:
            lines.append(f"Last cycle reflection: {self.last_reflection}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# THE DAEMON
# ═══════════════════════════════════════════════════════════════════

class TemporalDaemon:
    def __init__(self, goal_engine: Optional[GoalEngine] = None, data_dir=None):
        # data_dir: per-user Path for web app; None = use global singletons (CLI)
        self._data_dir = Path(data_dir) if data_dir else None

        if data_dir is not None:
            # Per-user: create fresh isolated instances
            self._cog_state = CognitiveStateEngine(data_dir)
            self._indelible_engine = IndelibleFactsEngine(data_dir)
            self._mode_engine = CognitiveModeEngine(data_dir) if MODES_AVAILABLE else None
            self._plan_buffer = PlanBuffer(data_dir) if MODES_AVAILABLE else None
        else:
            # CLI: use global singletons (existing behavior)
            self._cog_state = get_cognitive_state()
            self._indelible_engine = get_indelible_engine()
            self._mode_engine = get_mode_engine() if MODES_AVAILABLE else None
            self._plan_buffer = get_plan_buffer() if MODES_AVAILABLE else None

        self._goals = goal_engine or GoalEngine()
        self._lifecycle = GoalLifecycle()
        self._context = ConversationContext(max_turns=5)

        # Preference layer: Signal's own tastes. Per-user it persists to
        # affinity.json; CLI runs memory-only (seeded, no file). The daemon is
        # the SOLE writer of affinity.json — it learns by READING the
        # dispatcher's initiative_log, so there's no cross-process write race.
        if PREF_AVAILABLE:
            pref_path = (self._data_dir / "affinity.json") if self._data_dir else None
            self._pref = PreferenceLayer(path=pref_path)
        else:
            self._pref = None
        self._identity_keywords = self._indelible_engine.extract_identity_keywords()

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._paused = threading.Event()
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

        self._phase: int = 0
        self._cycle_count: int = 0
        self._real_turns: int = 0
        self._goal_birth_turn: Dict[str, int] = {}    # NEW: track which turn each goal was born on

        self._candidate_scores: Dict[str, float] = {}
        self._action_queue: List[ActionCandidate] = []
        self._recommendations: List[ActionCandidate] = []
        self._user_urgent: List[ActionCandidate] = []
        self._focus_summary: str = ""
        self._ambient_awareness: str = ""
        self._items_purged: int = 0
        self._on_urgent_callback = None

        # v7: REFLECT phase state
        self._reflection_seeds: List[Dict[str, Any]] = []   # Seeds for next cycle's phase 0
        self._last_cycle_summary: str = ""                   # What REFLECT concluded
        self._code_refresh_pending: bool = False             # Triggers read_code injection
        self._cycles_since_code_read: int = 0                # Counter for 100-cycle code refresh
        self._pending_tasks: deque = deque()                 # v7ys: queued evolution tasks

    @staticmethod
    def _strip_esc(text: str) -> str:
        """Strip ALL 'Unaddressed urgent: ' from text (leading + embedded).
        Delegates to GoalEngine._nuke_prefix() — single source of truth."""
        return GoalEngine._nuke_prefix(text)

    @property
    def lifecycle(self) -> GoalLifecycle:
        return self._lifecycle

    # ═══ LIFECYCLE ═══

    def _migrate_goals_registry_on_disk(self):
        """
        One-shot migration: load goals_registry.json, strip all
        'Unaddressed urgent:' prefixes, deduplicate by normalized
        description, and write the cleaned version back.

        goals_registry.json is a write-only snapshot consumed by the
        evolve loop — the goal engine never reads it back. So polluted
        entries from pre-Day-28 sessions survive on disk indefinitely
        unless we explicitly clean the file itself.
        """
        path = (self._data_dir / "goals_registry.json"
                if self._data_dir
                else Path(__file__).parent / "goals_registry.json")
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return

            _PREFIX = "Unaddressed urgent: "
            cleaned = 0
            seen = {}  # normalized_desc -> index of best entry

            for i, entry in enumerate(data):
                desc = entry.get("description", "")
                original = desc
                while desc.startswith(_PREFIX):
                    desc = desc[len(_PREFIX):]
                if desc != original:
                    entry["description"] = desc
                    cleaned += 1
                key = desc.lower().strip()
                if not key:
                    continue
                if key in seen:
                    # Keep the one with higher curiosity
                    prev = data[seen[key]]
                    if entry.get("curiosity", 0) > prev.get("curiosity", 0):
                        seen[key] = i
                else:
                    seen[key] = i

            # Rebuild list from deduplicated entries (preserving order)
            surviving_indices = set(seen.values())
            new_data = [entry for i, entry in enumerate(data)
                        if i in surviving_indices and entry.get("description", "").strip()]
            removed = len(data) - len(new_data)

            if cleaned > 0 or removed > 0:
                path.write_text(
                    json.dumps(new_data, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                print(f"[DAEMON] Disk migration: stripped {cleaned} prefixes, "
                      f"removed {removed} duplicates/empties from goals_registry.json")
        except Exception as e:
            print(f"[DAEMON] goals_registry.json migration failed (non-fatal): {e}")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        # One-shot: clean the persisted goals_registry.json on disk
        # BEFORE seeding or starting the daemon thread.
        self._migrate_goals_registry_on_disk()

        # Day 35 fix: clean any goals already in-memory BEFORE seeding.
        # If the goal engine was reused or pre-populated (e.g. web app),
        # polluted descriptions from a prior session could survive in the
        # dict. clean_registry() strips all "Unaddressed urgent:" prefixes
        # and deduplicates in a single pass (Day 32 unified implementation).
        pre_clean = self._goals.clean_registry()
        if pre_clean > 0:
            print(f"[DAEMON] Pre-seed cleanup: removed {pre_clean} polluted/duplicate goals")

        seed_goals_from_boot(self._goals, self._lifecycle,
                             indelible_engine=self._indelible_engine,
                             data_dir=self._data_dir)

        # Day 35 consolidated post-seed cleanup: clean_registry() handles
        # prefix stripping AND deduplication in one pass (Day 32 fix).
        # This replaces the old manual strip loop + separate dedup call,
        # which had ordering edge cases where double-nested prefixes like
        # "Unaddressed urgent: Unaddressed urgent: ..." could survive
        # if the manual loop and clean_registry() disagreed on state.
        post_clean = self._goals.clean_registry()
        if post_clean > 0:
            self._save_goals_registry()
            print(f"[DAEMON] Boot cleanup: removed {post_clean} polluted/duplicate goals")
        self._stop_flag.clear()
        self._paused.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, name="TemporalDaemon", daemon=True)
        self._thread.start()
        print("[DAEMON] Temporal daemon started — cognitive heartbeat active")

    def stop(self):
        self._stop_flag.set()
        self._running.set()
        self._paused.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._save_goals_registry()   # Final flush on shutdown
        print("[DAEMON] Temporal daemon stopped")

    def pause(self):
        self._paused.set()
        self._running.clear()

    def resume(self):
        self._identity_keywords = self._indelible_engine.extract_identity_keywords()
        self._real_turns += 1
        with self._lock:
            self._cycle_count = 0
        self._paused.clear()
        self._running.set()

    @property
    def is_running(self) -> bool:
        return self._running.is_set() and not self._paused.is_set()

    # ═══ CONVERSATION INTERFACE ═══

    def on_turn_complete(self, user_input: str, bot_output: str):
        """Called after each turn. Feeds goals + context."""
        if self._real_turns <= 1:
            demote_boot_goals(self._goals, self._lifecycle)

        # Resolve OLD goals first, THEN feed new ones
        resolve_discussed_goals(self._goals, self._lifecycle, user_input, self._cycle_count)
        feed_goals_from_turn(self._goals, self._lifecycle, user_input, bot_output,
                             current_turn=self._real_turns,
                             goal_birth_turn=self._goal_birth_turn)

        # Update conversation context buffer
        self._context.add_turn(user_input, bot_output)

        # Persist goals after each turn — this is when they change most
        self._save_goals_registry()

    # ═══ SNAPSHOT ═══

    def get_snapshot(self) -> CognitiveSnapshot:
        with self._lock:
            return CognitiveSnapshot(
                timestamp=time.time(),
                cycle_count=self._cycle_count,
                current_phase=self._phase,
                top_recommendations=[
                    {"goal_id": r.goal_id, "description": r.description,
                     "action_type": r.action_type, "composite_score": r.composite_score,
                     "needs_user": r.needs_user, "reasoning": r.reasoning,
                     "curiosity_trend": self._curiosity_trend(r.goal_id)}
                    for r in self._recommendations[:10]
                ],
                user_urgent_items=[
                    {"goal_id": u.goal_id, "description": u.description,
                     "action_type": u.action_type, "reasoning": u.reasoning}
                    for u in self._user_urgent[:5]
                ],
                focus_summary=self._focus_summary,
                ambient_awareness=self._ambient_awareness,
                good_sense=compute_good_sense(self._cog_state),
                crap_threshold=compute_crap_threshold(self._cog_state),
                items_evaluated=len(self._candidate_scores),
                items_purged=self._items_purged,
                last_reflection=self._last_cycle_summary,
                code_refresh_pending=self._code_refresh_pending,
            )

    def _curiosity_trend(self, goal_id: str) -> str:
        """Show if curiosity is growing or shrinking for a goal."""
        goal = self._goals.goals.get(goal_id)
        if not goal:
            return ""
        if goal.curiosity > 0.8:
            return "(↑ high interest)"
        if goal.curiosity > 0.5:
            return "(↑ growing)"
        if goal.curiosity > 0.2:
            return "(→ stable)"
        return "(↓ fading)"

    def set_urgent_callback(self, callback):
        self._on_urgent_callback = callback

    # ═══ THE LOOP ═══

    def _run_loop(self):
        while not self._stop_flag.is_set():
            self._running.wait(timeout=0.5)
            if self._stop_flag.is_set():
                break
            if self._paused.is_set():
                continue
            try:
                self._execute_phase(self._phase)
            except Exception as e:
                print(f"[DAEMON] Phase {self._phase} error: {e}")
            self._phase = (self._phase + 1) % NUM_PHASES
            if self._phase == 0:
                self._cycle_count += 1
            remaining = TICK_INTERVAL
            while remaining > 0 and not self._paused.is_set() and not self._stop_flag.is_set():
                chunk = min(0.02, remaining)
                time.sleep(chunk)
                remaining -= chunk

    def _execute_phase(self, phase: int):
        {
            Phase.CHECK_GOALS:       self._phase_check_goals,
            Phase.EVALUATE:          self._phase_evaluate,
            Phase.DETERMINE_ACTIONS: self._phase_determine_actions,
            Phase.APPEND_RECOMMEND:  self._phase_append_recommend,
            Phase.PRIORITIZE:        self._phase_prioritize,
            Phase.REEVAL_IDENTITY:   self._phase_reeval_identity,
            Phase.USER_URGENCY:      self._phase_user_urgency,
            Phase.SUMMARIZE:         self._phase_summarize,
            Phase.CLEANUP:           self._phase_cleanup,
            Phase.REFLECT:           self._phase_reflect,
        }[phase]()

    # ─── PHASE 0: CHECK GOALS ───
    def _phase_check_goals(self):
        # Pre-clean: strip "Unaddressed urgent:" prefixes from all goals
        # BEFORE seed dedup runs. This closes the ordering gap where
        # _phase_reflect's cleanup runs AFTER _phase_check_goals in the
        # same cycle, so a polluted goal could bypass the 30-char prefix
        # dedup comparison against incoming seeds (Day 30 fix).
        _ESC = "Unaddressed urgent: "
        with self._goals._lock:
            for g in self._goals.goals.values():
                if g.description.startswith(_ESC):
                    cleaned = g.description
                    while cleaned.startswith(_ESC):
                        cleaned = cleaned[len(_ESC):]
                    if cleaned.strip():
                        g.description = cleaned

        # decay_curiosity() acquires goal_engine._lock internally.
        # Do NOT call it under self._lock — that creates a lock-ordering
        # inversion with the main thread's on_turn_complete() →
        # feed_goals_from_turn() which acquires goal_engine._lock first,
        # then may contend on self._lock. Deadlock: daemon holds self._lock
        # waiting for goals._lock, main thread holds goals._lock waiting
        # for self._lock.
        self._goals.decay_curiosity()

        # Preference layer ticks once per cycle, right alongside curiosity decay
        # (curiosity = fast state; preference = slow trait). Learn new
        # engagement from the dispatcher's log, fade tastes slightly, persist.
        # Fail gracefully — a taste hiccup must never stall the cognitive cycle.
        if self._pref is not None and self._data_dir is not None:
            try:
                self._pref.learn_from_initiative_log(
                    self._data_dir / "initiative_log.jsonl")
                self._pref.decay_tick()
                self._pref.save()   # affinity.json is tiny (two small dicts)
            except Exception as e:
                print(f"[DAEMON] preference tick failed (non-fatal): {e}")

        if self._cycle_count % 10 == 0:
            self._identity_keywords = self._indelible_engine.extract_identity_keywords()
        if self._cycle_count % 100 == 0 and self._cycle_count > 0:
            # purge_stale() also acquires goal_engine._lock internally.
            # We need self._lock only to safely read the surviving keys
            # for lifecycle cleanup, so split the operations.
            purged = self._goals.purge_stale()
            if purged > 0:
                with self._lock:
                    self._lifecycle.cleanup(set(self._goals.goals.keys()))

        # Every-cycle in-memory cleanup: strip prefixes + dedup.
        # clean_registry() is O(n) where n≤50 goals — cheap enough to run
        # every cycle. This eliminates the 10-cycle window (Day 37) where
        # pollution introduced by phases 1-9 could accumulate and reach
        # disk via _save_goals_registry() before cleanup ran (Day 38 fix).
        cleaned = self._goals.clean_registry()
        if cleaned > 0:
            print(f"[DAEMON] Registry cleanup: removed {cleaned} polluted/duplicate goals")

        # v7: Pick up seeds from previous cycle's REFLECT phase
        _ESC_PREFIX = "Unaddressed urgent: "
        seeds = self.consume_reflection_seeds()
        for seed in seeds:
            with self._lock:
                # Don't duplicate existing goals with similar descriptions.
                # Strip all "Unaddressed urgent: " prefixes from BOTH sides
                # before comparing, so escalation seeds can't bypass dedup
                # by having a prefix the existing goal lacks (Day 22 fix).
                seed_desc = seed["description"]
                while seed_desc.startswith(_ESC_PREFIX):
                    seed_desc = seed_desc[len(_ESC_PREFIX):]
                desc_prefix = seed_desc.lower()[:30]
                already_exists = any(
                    self._strip_esc(g.description).lower()[:30] == desc_prefix
                    for g in self._goals.goals.values()
                )
                if not already_exists:
                    gid = self._goals.add_rabbit_hole(
                        seed["description"],
                        curiosity=seed.get("curiosity", 0.45)
                    )
                    self._lifecycle.register(gid, source=seed.get("source", "reflect"))

    # ─── PHASE 1: EVALUATE (with CURIOSITY GROWTH + TURN-BASED DECAY) ───
    def _phase_evaluate(self):
        good_sense = compute_good_sense(self._cog_state)
        state = self._cog_state.state
        threshold = compute_crap_threshold(self._cog_state)

        # Snapshot goals under goal_engine's lock to prevent RuntimeError
        # from concurrent mutation by on_turn_complete() on the main thread.
        # NOTE: self._lock protects daemon internals (_candidate_scores etc.),
        # but self._goals._lock protects the goals dict. We must use the
        # latter for the snapshot, then self._lock for writing scores.
        with self._goals._lock:
            goals_snapshot = [(gid, Goal(id=g.id, type=g.type, description=g.description,
                                         last_active=g.last_active, importance=g.importance,
                                         curiosity=g.curiosity, unresolved=g.unresolved,
                                         identity_relevance=g.identity_relevance,
                                         access_count=g.access_count, created_at=g.created_at))
                              for gid, g in self._goals.goals.items()]

        # Calculate total curiosity budget usage
        total_curiosity = sum(g.curiosity for _, g in goals_snapshot)

        # Batch curiosity updates — applied after releasing self._lock
        # to avoid lock-ordering inversion with goal_engine._lock.
        curiosity_updates = []

        with self._lock:
            self._candidate_scores.clear()
            for gid, goal in goals_snapshot:
                # Base score
                score = (
                    goal.curiosity * 0.30 + good_sense * 0.25 +
                    goal.importance * 0.25 + state.engagement * 0.20
                )

                # Context relevance boost (v4 base + v6 turn-decay upgrade)
                # Goals related to recent conversation get a boost
                relevance = self._context.relevance_score(goal.description)
                if relevance > 0.1:
                    # TURN-BASED DECAY: goals born many turns ago get
                    # diminishing context boost. This replaces pure wall-clock
                    # decay for the middle TWDC layer.
                    # Half-life of ~8 turns: a goal from 8 turns ago gets 50% boost,
                    # 16 turns ago gets 25%, etc.
                    birth_turn = self._goal_birth_turn.get(gid, 0)
                    turn_age = max(0, self._real_turns - birth_turn)
                    turn_decay = 1.0 / (1.0 + turn_age / 8.0)  # smooth hyperbolic decay
                    score += relevance * CONTEXT_RELEVANCE_BOOST * turn_decay

                self._candidate_scores[gid] = score
                self._lifecycle.update_score(gid, score, self._cycle_count)

                # ═══ CURIOSITY GROWTH (THE KEY FIX) ═══
                # If this goal scored above threshold, it's worth thinking about.
                # Thinking about it makes you MORE curious, not less.
                # Growth is constrained by per-goal cap and total budget.
                meta = self._lifecycle.get(gid)
                is_resolved = meta.get("resolved", False) if meta else False
                is_boot = (meta.get("source") == "boot") if meta else False

                if (score >= threshold and
                    not is_resolved and
                    not is_boot and
                    goal.curiosity < CURIOSITY_PER_GOAL_CAP and
                    total_curiosity < CURIOSITY_TOTAL_BUDGET):

                    # Growth scales with context relevance
                    growth = CURIOSITY_GROWTH_PER_CYCLE
                    if relevance > 0.2:
                        growth *= 1.5  # 50% faster growth for contextually relevant goals

                    # Collect growth updates — do NOT mutate live goals here.
                    # self._goals.goals is protected by self._goals._lock,
                    # and acquiring it under self._lock would create a
                    # lock-ordering inversion with on_turn_complete().
                    # We batch updates and apply them after releasing self._lock.
                    curiosity_updates.append((gid, growth))
                    total_curiosity += growth

        # Apply batched curiosity growth outside self._lock to avoid
        # lock-ordering inversion. Acquire only goal_engine._lock.
        if curiosity_updates:
            with self._goals._lock:
                for gid, growth in curiosity_updates:
                    goal = self._goals.goals.get(gid)
                    if goal is not None:
                        goal.curiosity = min(CURIOSITY_PER_GOAL_CAP, goal.curiosity + growth)

    # ─── PHASE 2: DETERMINE ACTIONS ───
    def _phase_determine_actions(self):
        threshold = compute_crap_threshold(self._cog_state)
        state = self._cog_state.state
        new_actions = []

        # Step 1: Snapshot scores under daemon lock (self._lock protects
        # _candidate_scores which is written by _phase_evaluate).
        with self._lock:
            scored = sorted(self._candidate_scores.items(), key=lambda x: x[1], reverse=True)
            above_threshold = [(gid, score) for gid, score in scored if score >= threshold]

        # Step 2: Snapshot goal data under goal engine lock (self._goals._lock
        # protects the goals dict). This is separate from self._lock to avoid
        # lock-ordering inversion with on_turn_complete() which acquires
        # self._goals._lock first, then may contend on self._lock.
        goal_snapshots = {}
        with self._goals._lock:
            for gid, score in above_threshold:
                goal = self._goals.goals.get(gid)
                if goal is not None:
                    goal_snapshots[gid] = {
                        "id": goal.id,
                        "description": goal.description,
                        "curiosity": goal.curiosity,
                        "importance": goal.importance,
                        "unresolved": goal.unresolved,
                        "last_active": goal.last_active,
                        "score": score,
                    }

        # Process snapshots outside both locks — no live goal access needed
        for gid, gsnap in goal_snapshots.items():
            atype = self._pick_action_type_from_snap(gsnap, state)
            # Strip pollution at ActionCandidate creation — this is the read
            # path from goal snapshots into _recommendations → _user_urgent →
            # _phase_reflect escalation seeds. Without this, a goal polluted
            # between _phase_check_goals cleanup and this snapshot survives
            # into the recommendation pipeline (Day 39 fix).
            clean_desc = self._strip_esc(gsnap["description"])
            identity_score = self._compute_identity_relevance_from_desc(clean_desc)
            # Preference bias: nudge priority toward Signal's own tastes. Small
            # bounded additive term (±BIAS_CLAMP), same altitude as
            # identity_relevance. Biases priority among survivors — never vetoes
            # (good-sense threshold already gated survival above).
            pref_bias = (self._pref.bias_for_text(clean_desc, atype)
                         if self._pref is not None else 0.0)
            composite = max(0.0, min(1.5, gsnap["score"] + pref_bias))
            new_actions.append(ActionCandidate(
                goal_id=gid, description=clean_desc, action_type=atype,
                composite_score=composite, curiosity_score=gsnap["curiosity"],
                identity_score=identity_score,
                needs_user=(atype == "ask_user"),
                reasoning=self._generate_reasoning_from_snap(gsnap, atype, gsnap["score"]),
            ))
        with self._lock:
            self._action_queue = new_actions

    def _pick_action_type(self, goal, state) -> str:
        """
        CHANGED in v4: Rabbit holes default to "explore", not "resolve".
        "resolve" was making every goal appear to need user input.
        Now only truly stale unresolved items get "resolve".
        """
        meta = self._lifecycle.get(goal.id)
        is_resolved = (meta and meta.get("resolved"))
        if is_resolved:
            return "think"

        # High curiosity + capacity → explore (the happy path)
        if goal.curiosity > 0.5 and state.cognitive_load < 0.7:
            return "explore"

        # Old unresolved goals that might need user input
        age = time.time() - goal.last_active
        if goal.unresolved and age > 600:
            return "ask_user"

        # Very high importance but low confidence → ask
        if goal.importance > 0.8 and state.confidence < 0.4:
            return "ask_user"

        # Stale → revisit
        if age > 900:
            return "revisit"

        return "think"

    def _pick_action_type_from_snap(self, gsnap: Dict[str, Any], state) -> str:
        """Snapshot-safe version of _pick_action_type (works on dicts, not live Goals)."""
        meta = self._lifecycle.get(gsnap["id"])
        is_resolved = (meta and meta.get("resolved"))
        if is_resolved:
            return "think"
        if gsnap["curiosity"] > 0.5 and state.cognitive_load < 0.7:
            return "explore"
        age = time.time() - gsnap["last_active"]
        if gsnap["unresolved"] and age > 600:
            return "ask_user"
        if gsnap["importance"] > 0.8 and state.confidence < 0.4:
            return "ask_user"
        if age > 900:
            return "revisit"
        return "think"

    def _compute_identity_relevance(self, goal) -> float:
        if not self._identity_keywords: return 0.5
        desc = goal.description.lower()
        return min(1.0, sum(1 for kw in self._identity_keywords if kw in desc) * 0.2)

    def _compute_identity_relevance_from_desc(self, description: str) -> float:
        """Snapshot-safe version of _compute_identity_relevance."""
        if not self._identity_keywords: return 0.5
        desc = description.lower()
        return min(1.0, sum(1 for kw in self._identity_keywords if kw in desc) * 0.2)

    def _generate_reasoning(self, goal, atype, score) -> str:
        parts = [f"score={score:.2f}"]
        if goal.curiosity > 0.5: parts.append(f"curio={goal.curiosity:.2f}")
        if goal.unresolved: parts.append("unresolved")
        meta = self._lifecycle.get(goal.id)
        if meta:
            if meta.get("resolved"): parts.append("resolved")
            if meta.get("source") == "boot": parts.append("boot")
        # Show context relevance
        rel = self._context.relevance_score(goal.description)
        if rel > 0.1: parts.append(f"ctx={rel:.2f}")
        return f"{atype}({', '.join(parts)})"

    def _generate_reasoning_from_snap(self, gsnap: Dict[str, Any], atype: str, score: float) -> str:
        """Snapshot-safe version of _generate_reasoning."""
        parts = [f"score={score:.2f}"]
        if gsnap["curiosity"] > 0.5: parts.append(f"curio={gsnap['curiosity']:.2f}")
        if gsnap["unresolved"]: parts.append("unresolved")
        meta = self._lifecycle.get(gsnap["id"])
        if meta:
            if meta.get("resolved"): parts.append("resolved")
            if meta.get("source") == "boot": parts.append("boot")
        rel = self._context.relevance_score(gsnap["description"])
        if rel > 0.1: parts.append(f"ctx={rel:.2f}")
        return f"{atype}({', '.join(parts)})"

    # ─── PHASE 3: APPEND RECOMMENDATIONS ───
    def _phase_append_recommend(self):
        with self._lock:
            existing = {r.goal_id: r for r in self._recommendations}
            for a in self._action_queue:
                if a.goal_id not in existing or a.composite_score > existing[a.goal_id].composite_score:
                    existing[a.goal_id] = a
            self._recommendations = list(existing.values())

    # ─── PHASE 4: PRIORITIZE ───
    def _phase_prioritize(self):
        with self._lock:
            self._recommendations.sort(key=lambda r: r.composite_score, reverse=True)

    # ─── PHASE 5: RE-EVALUATE IDENTITY ───
    def _phase_reeval_identity(self):
        with self._lock:
            for r in self._recommendations:
                if r.identity_score < 0.3 and r.curiosity_score < 0.6:
                    r.composite_score *= 0.70
                elif r.identity_score > 0.7:
                    r.composite_score = min(1.5, r.composite_score * 1.15)
            self._recommendations.sort(key=lambda r: r.composite_score, reverse=True)

    # ─── PHASE 6: USER URGENCY ───
    def _phase_user_urgency(self):
        now = time.time()

        # Step 1: Collect which goal IDs we need, under self._lock
        # (self._lock protects _recommendations).
        with self._lock:
            needed_goal_ids = [
                r.goal_id for r in self._recommendations if r.needs_user
            ]

        # Step 2: Snapshot goal data under goal engine lock ONLY.
        # This avoids lock-ordering inversion: on_turn_complete() on the
        # main thread acquires self._goals._lock first, then may contend
        # on self._lock. If we held self._lock here while accessing
        # self._goals.goals, we'd have the inverse ordering → deadlock.
        goal_snapshots = {}
        with self._goals._lock:
            for gid in needed_goal_ids:
                g = self._goals.goals.get(gid)
                if g is not None:
                    goal_snapshots[gid] = {
                        "importance": g.importance,
                        "unresolved": g.unresolved,
                        "last_active": g.last_active,
                    }

        # Step 3: Process under self._lock (no goal engine access needed).
        with self._lock:
            # Day 40 fix: strip ALL recommendation descriptions, not just those
            # passing the urgency threshold. Recommendations persist across cycles
            # via _phase_append_recommend's merge logic, so a polluted description
            # that doesn't pass the threshold THIS cycle can pass it in a FUTURE
            # cycle and enter _user_urgent → _phase_reflect escalation seeds with
            # pollution intact. Stripping everything here closes that vector.
            for r in self._recommendations:
                r.description = self._strip_esc(r.description)

            self._user_urgent.clear()
            for r in self._recommendations:
                if not r.needs_user:
                    continue
                gsnap = goal_snapshots.get(r.goal_id)
                if not gsnap:
                    continue
                age = now - gsnap["last_active"]
                if (gsnap["importance"] > 0.7 and age > 300) or (gsnap["unresolved"] and age > 900) or r.composite_score > 0.7:
                    self._user_urgent.append(r)
            if self._user_urgent and self._on_urgent_callback:
                try: self._on_urgent_callback(len(self._user_urgent))
                except: pass

    # ─── PHASE 7: SUMMARIZE + AMBIENT ───
    def _phase_summarize(self):
        with self._lock:
            self._ambient_awareness = _build_ambient_awareness(
                self._cog_state, self._cycle_count
            )
            if not self._recommendations:
                self._focus_summary = "No active goals — resting in ambient awareness."
                return
            active = [r for r in self._recommendations
                     if not (self._lifecycle.get(r.goal_id) or {}).get("resolved")]
            if not active:
                active = self._recommendations[:3]
            topics = [r.description[:30] for r in active[:3]]
            gs = compute_good_sense(self._cog_state)
            prefix = "Actively exploring" if gs > 0.7 else ("Considering" if gs > 0.4 else "Low-confidence on")
            self._focus_summary = f"{prefix}: {' | '.join(topics)}"

    # ─── PHASE 8: CLEANUP ───
    def _phase_cleanup(self):
        threshold = compute_crap_threshold(self._cog_state)
        state = self._cog_state.state
        cycle = self._cycle_count

        # Step 1: Prune recommendations under self._lock (no goal engine access).
        with self._lock:
            before = len(self._recommendations)
            survivors = []
            for r in self._recommendations:
                should_purge = False

                is_crap = (r.composite_score < threshold and
                          r.curiosity_score < 0.3 and r.identity_score < 0.3)
                if is_crap and state.engagement > 0.6:
                    is_crap = False
                if is_crap:
                    should_purge = True

                if self._lifecycle.is_boot_expired(r.goal_id, cycle):
                    should_purge = True
                if self._lifecycle.is_resolved_and_lingered(r.goal_id, cycle):
                    should_purge = True
                if self._lifecycle.is_stale(r.goal_id, cycle):
                    r.composite_score *= 0.5
                    if r.composite_score < threshold:
                        should_purge = True

                if not should_purge:
                    survivors.append(r)

            purged = before - len(survivors)
            self._items_purged += purged
            self._recommendations = survivors
            sids = {r.goal_id for r in survivors}
            self._user_urgent = [u for u in self._user_urgent if u.goal_id in sids]

        # Step 2: Purge expired/lingered goals under goal engine lock ONLY.
        # This avoids lock-ordering inversion: on_turn_complete() on the main
        # thread acquires self._goals._lock first, then may contend on
        # self._lock. If we held self._lock here while accessing
        # self._goals.goals, we'd have the inverse ordering → deadlock.
        with self._goals._lock:
            all_goal_ids = set(self._goals.goals.keys())
            to_delete = []
            for gid in all_goal_ids:
                meta = self._lifecycle.get(gid)
                if not meta:
                    continue
                if (self._lifecycle.is_boot_expired(gid, cycle) or
                    self._lifecycle.is_resolved_and_lingered(gid, cycle)):
                    g = self._goals.goals.get(gid)
                    if g and "Identity anchor" not in g.description:
                        to_delete.append(gid)
            for gid in to_delete:
                del self._goals.goals[gid]
            surviving_goal_ids = set(self._goals.goals.keys())

        # Step 3: Clean lifecycle metadata (no locks on goal engine needed).
        self._lifecycle.cleanup(surviving_goal_ids)

        # v5: Mode decay — let inactive modes fade each cycle
        if MODES_AVAILABLE:
            try:
                self._plan_buffer.daemon_check()
            except Exception as e:
                if self._cycle_count % 100 == 0:
                    print(f"[DAEMON] plan_buffer.daemon_check() error: {e}")
            try:
                self._mode_engine.decay_all_modes()
            except Exception as e:
                if self._cycle_count % 100 == 0:
                    print(f"[DAEMON] decay_all_modes() error: {e}")
        
        # v5: Periodic archival — every 500 cycles (~50 seconds),
        # check if old memories need compressing to archive
        if MODES_AVAILABLE and self._cycle_count > 0 and self._cycle_count % 500 == 0:
            try:
                archived = archive_old_memories(data_dir=self._data_dir)
                if archived > 0:
                    self._mode_engine.refresh_archive_tags()
                    print(f"[DAEMON] Auto-archived {archived} episodes")
            except Exception as e:
                print(f"[DAEMON] Archive error: {e}")

    # ─── PHASE 9: REFLECT (v7) ───
    def _phase_reflect(self):
        """
        REFLECT: End-of-cycle introspection.

        Does three things:
        1. EVALUATE what this cycle accomplished — what goals moved,
           what got purged, what's stagnant.
        2. SEED goals/notes for the next cycle's phase 0 (CHECK_GOALS)
           to pick up. This closes the cognitive loop so the daemon
           learns from its own processing rather than just filtering.
           Seeds are only generated every REFLECT_SEED_INTERVAL cycles
           to prevent goal pollution during long idle periods.
        3. CODE REFRESH: Every 100 cycles (~100 seconds), flag that
           the next prompt should include a fresh code read for
           contextual self-model maintenance. For v7ys autonomous
           evolution, this keeps the self-model current.

        The reflection is lightweight — no API calls, no heavy compute.
        Just bookkeeping that makes the next cycle smarter.
        """
        # Step 1: Snapshot goal data under goal engine lock ONLY.
        # This avoids the lock-ordering inversion fixed in Days 30-32
        # for other phases: on_turn_complete() on the main thread acquires
        # self._goals._lock first, then may contend on self._lock. If we
        # held self._lock here while iterating self._goals.goals, we'd
        # have the inverse ordering → deadlock or RuntimeError.
        with self._goals._lock:
            goals_snapshot = [
                (gid, g.description[:40], g.curiosity)
                for gid, g in self._goals.goals.items()
            ]
            num_goals = len(self._goals.goals)
            # Snapshot existing goal description prefixes for escalation
            # seed dedup inside the self._lock block below. Must be built
            # here (under goals._lock only) to avoid lock-ordering inversion
            # — self._lock is acquired AFTER goals._lock, never the reverse.
            existing_descs = set(
                self._strip_esc(g.description).lower()[:30]
                for g in self._goals.goals.values()
            )

        with self._lock:
            # ─── 1. CYCLE EVALUATION ───
            num_recs = len(self._recommendations)
            num_urgent = len(self._user_urgent)

            # Identify goals that grew in curiosity this cycle (thriving)
            thriving = []
            stagnant = []
            threshold = compute_crap_threshold(self._cog_state)
            for gid, desc, curiosity in goals_snapshot:
                meta = self._lifecycle.get(gid)
                if not meta:
                    continue
                if meta.get("resolved"):
                    continue
                score = self._candidate_scores.get(gid, 0.0)
                if score >= threshold and curiosity > 0.5:
                    thriving.append((gid, desc, curiosity))
                elif score < threshold and curiosity < 0.2:
                    stagnant.append((gid, desc))

            # ─── 2. SEED FOR NEXT CYCLE ───
            # Only generate seeds every 50 cycles (~50 seconds) to prevent
            # goal pollution during long idle periods. Without this, each
            # 1-second cycle creates near-duplicate "Explore connection..."
            # goals that bypass the 30-char prefix dedup in CHECK_GOALS
            # because thriving goal descriptions shift slightly each cycle.
            should_seed = (self._cycle_count % 50 == 0) or (self._cycle_count < 3)
            self._reflection_seeds.clear()

            if should_seed:
                # If we have thriving goals, seed a meta-goal to explore connections
                if len(thriving) >= 2:
                    descriptions = [t[1] for t in thriving[:3]]
                    self._reflection_seeds.append({
                        "type": "connection",
                        "description": f"Explore connection between: {' & '.join(descriptions)}",
                        "curiosity": 0.55,
                        "source": "reflect",
                    })

                # If everything is stagnant, seed a novelty-seeking goal
                if num_recs > 0 and len(stagnant) > num_recs * 0.7:
                    self._reflection_seeds.append({
                        "type": "novelty",
                        "description": "Seek novel direction — current goals are stagnating",
                        "curiosity": 0.60,
                        "source": "reflect",
                    })

                # If urgent items exist but haven't been addressed, escalate.
                # Day 24 fix: Do NOT prepend "Unaddressed urgent:" — that was
                # the root cause of registry pollution. Instead, re-seed the
                # core description with boosted curiosity and source="escalation"
                # so the daemon treats it with higher priority without polluting
                # the description text. Also check the full goal registry to
                # avoid re-seeding goals that already exist.
                # Day 25 fix: existing_descs was snapshotted BEFORE self._lock
                # to avoid lock-ordering inversion (see step 0 above).
                if num_urgent > 0:
                    _ESC_PREFIX = "Unaddressed urgent: "
                    for u in self._user_urgent[:2]:
                        core_desc = u.description
                        while core_desc.startswith(_ESC_PREFIX):
                            core_desc = core_desc[len(_ESC_PREFIX):]
                        if not core_desc.strip():
                            continue
                        # Skip if this core description already exists in registry
                        if core_desc.lower()[:30] in existing_descs:
                            continue
                        self._reflection_seeds.append({
                            "type": "escalation",
                            "description": core_desc[:50],
                            "curiosity": 0.60,
                            "source": "escalation",
                        })

        # ─── 1b. GOAL REGISTRY PREFIX CLEANUP ───
        # Strip cascaded "Unaddressed urgent: " prefixes from existing goals
        # to prevent pollution that accumulated before the Day 17 seed fix.
        _ESCALATION_PREFIX = "Unaddressed urgent: "
        with self._goals._lock:
            for g in self._goals.goals.values():
                if g.description.startswith(_ESCALATION_PREFIX):
                    cleaned = g.description
                    while cleaned.startswith(_ESCALATION_PREFIX):
                        cleaned = cleaned[len(_ESCALATION_PREFIX):]
                    if cleaned.strip():
                        g.description = cleaned

            # Build cycle summary for snapshot
            self._last_cycle_summary = (
                f"cycle={self._cycle_count} | "
                f"goals={num_goals} | recs={num_recs} | "
                f"thriving={len(thriving)} | stagnant={len(stagnant)} | "
                f"urgent={num_urgent} | seeds={len(self._reflection_seeds)}"
            )

        # ─── 3. CODE REFRESH TRACKING (v7ys) ───
        with self._lock:
            self._cycles_since_code_read += 1
            if self._cycles_since_code_read >= 100:
                self._code_refresh_pending = True
                self._cycles_since_code_read = 0

        # ─── 4. GOAL REGISTRY PERSIST (every 100 cycles) ───
        # Complements the per-turn write — keeps registry fresh during idle periods
        if self._cycle_count % 100 == 0 and self._cycle_count > 0:
            self._save_goals_registry()

    def consume_reflection_seeds(self) -> List[Dict[str, Any]]:
        """
        Called by phase 0 (CHECK_GOALS) on the next cycle to pick up
        seeds that REFLECT planted. Returns and clears the seed list.
        """
        with self._lock:
            seeds = list(self._reflection_seeds)
            self._reflection_seeds.clear()
        return seeds

    def consume_code_refresh(self) -> bool:
        """
        Check if a code refresh is pending (every 100 cycles).
        Returns True once, then resets the flag.
        Used by signalbot.py or the v7ys evolution loop to inject
        fresh code context into the next prompt.
        """
        if self._code_refresh_pending:
            self._code_refresh_pending = False
            return True
        return False

    def _save_goals_registry(self):
        """
        Write current goal state to goals_registry.json so the evolve loop
        can read active goals and use them to guide what to improve.
        Per-user: writes to data_dir. CLI singleton: writes next to this file.

        Day 27 fix: clean_registry() is called before every write so that
        any "Unaddressed urgent:" prefix pollution or duplicates that
        accumulated in-memory are stripped before they reach disk.

        Day 36 fix: defense-in-depth write-time guard. After get_all_scored()
        returns the serialized list, strip all "Unaddressed urgent:" prefixes
        from each entry's description field. This closes the race window
        between clean_registry() and get_all_scored() where another thread
        could re-pollute a description in between the two calls.
        """
        try:
            self._goals.clean_registry()
            goals_data = self._goals.get_all_scored()
            # Write-time sanitization: strip pollution from serialized output
            # regardless of in-memory state, closing the race window.
            _PREFIX = "Unaddressed urgent: "
            for entry in goals_data:
                desc = entry.get("description", "")
                while desc.startswith(_PREFIX):
                    desc = desc[len(_PREFIX):]
                entry["description"] = desc
            # Stamp the lane classifier (action_type) onto each row so the
            # thought-quality ladder reads the REAL lane instead of re-deriving
            # it. The daemon stays the single source of truth for the classifier;
            # the registry is just the transport. get_all_scored() gives `age`,
            # the classifier wants `last_active` — synthesize it. Fail-graceful:
            # a state hiccup must never break the registry write (the ladder
            # falls back to goal.type when action_type is absent).
            try:
                state = self._cog_state.state
                _now = time.time()
                for entry in goals_data:
                    snap = {
                        "id": entry["id"],
                        "curiosity": entry.get("curiosity", 0.0),
                        "importance": entry.get("importance", 0.0),
                        "unresolved": entry.get("unresolved", False),
                        "last_active": _now - entry.get("age", 0.0),
                    }
                    entry["action_type"] = self._pick_action_type_from_snap(snap, state)
            except Exception:
                pass
            path = (self._data_dir / "goals_registry.json"
                    if self._data_dir
                    else Path(__file__).parent / "goals_registry.json")
            path.write_text(
                json.dumps(goals_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            # Only log occasionally to avoid spam
            if self._cycle_count % 100 == 0:
                print(f"[DAEMON] goals_registry write failed: {e}")

    # ═══ DIAGNOSTIC ═══

    def get_status(self) -> str:
        st = "RUNNING" if self.is_running else "PAUSED"
        with self._lock:
            nr, nu = len(self._recommendations), len(self._user_urgent)
            ng = len(self._goals.goals)
            total_curio = sum(g.curiosity for g in self._goals.goals.values())
        mode_info = ""
        if MODES_AVAILABLE and self._mode_engine is not None:
            try:
                mode_info = f" | {self._mode_engine.get_status()}"
            except Exception:
                pass
        return (f"[DAEMON] {st} | cycle={self._cycle_count} | "
                f"phase={Phase(self._phase).name} | goals={ng} | "
                f"recs={nr} | urgent={nu} | "
                f"good_sense={compute_good_sense(self._cog_state):.2f} | "
                f"total_curio={total_curio:.2f}/{CURIOSITY_TOTAL_BUDGET:.0f}"
                f"{mode_info}")


# SINGLETON
_daemon: Optional[TemporalDaemon] = None

def get_daemon(goal_engine=None, data_dir=None) -> TemporalDaemon:
    global _daemon
    if data_dir is not None:
        return TemporalDaemon(goal_engine=goal_engine, data_dir=data_dir)  # per-user: fresh
    if _daemon is None:
        _daemon = TemporalDaemon(goal_engine=goal_engine)  # CLI: lazy-init global
    return _daemon

def start_daemon(): get_daemon().start()
def stop_daemon(): get_daemon().stop()
def pause_daemon(): get_daemon().pause()
def resume_daemon(): get_daemon().resume()
def get_daemon_snapshot() -> CognitiveSnapshot: return get_daemon().get_snapshot()
