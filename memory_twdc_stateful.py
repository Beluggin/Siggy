# memory_twdc_stateful.py
"""
STATE-AWARE TWDC MEMORY WRAPPER

Wraps your existing TWDC memory system and modulates it with cognitive state.

KEY INNOVATION:
Memory scores are DYNAMIC - they change based on current state.

When frustrated → practical memories score higher
When curious → rabbit hole memories score higher
When cognitively overloaded → simple memories score higher
When identity_adherence high → identity facts get MASSIVE boost

HOW IT WORKS:
1. TWDC provides base scores (time/alignment/context decay)
2. This wrapper RE-SCORES based on current cognitive state
3. Indelible facts get injected at the top
4. Final block goes into the prompt
"""

from pathlib import Path
import json
from typing import Dict, Any, List

# Import the original TWDC system
try:
    from memory_summarizer_twdc import get_memory_engine
except ImportError:
    # Fallback if TWDC not available
    def get_memory_engine():
        class Stub:
            master_summary = {}
            top_memories = {}
            def load_existing(self): pass
            def notify_new_message(self): pass
        return Stub()

# Import our new state systems
from cognitive_state import get_cognitive_state
from indelible_facts import get_indelible_engine, get_indelible_prompt_section, get_indelible_keywords

class StatefulTWDCWrapper:
    """
    Wraps TWDC with state-aware re-scoring.

    v6 UPGRADE:
      - Contextual relevance now uses co-occurrence expansion from the daemon's
        ConversationContext (if available), so memories about "robot" get boosted
        when you're talking about "tank" because those words co-occurred.
      - Turn-based decay: memories from many turns ago get less state boost,
        independent of wall-clock time.
    """
    
    def __init__(self, data_dir=None):
        # data_dir: per-user Path for web app; None = repo root (CLI).
        # Anchored to __file__, not cwd — cwd-relative state forks (Cerberus rule).
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).parent
        self.twdc = get_memory_engine()
        self.cog_state = get_cognitive_state(data_dir)
        # Per-user indelible engine so keywords/facts are user-specific
        self._indelible = get_indelible_engine(data_dir)
        self._conversation_context = None  # Set by daemon if available
        self._current_turn: int = 0

        # Hydrate TWDC
        if hasattr(self.twdc, 'load_existing'):
            self.twdc.load_existing()

    def set_conversation_context(self, context, current_turn: int = 0):
        """Allow the daemon to inject its ConversationContext for richer scoring."""
        self._conversation_context = context
        self._current_turn = current_turn
    
    def _compute_contextual_relevance(self, text_blob: str) -> float:
        """
        Compute how relevant a memory is to the current conversation.
        Uses the daemon's ConversationContext if available (bigrams + co-occurrence),
        otherwise falls back to basic keyword matching.
        """
        if self._conversation_context is not None:
            return self._conversation_context.relevance_score(text_blob)

        # Fallback: basic keyword check against indelible facts
        identity_keywords = self._indelible.extract_identity_keywords()
        if not identity_keywords:
            return 0.0
        text_lower = text_blob.lower()
        hits = sum(1 for kw in identity_keywords if kw in text_lower)
        return min(1.0, hits * 0.2)

    def _apply_state_modulation(self, memory_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-score memories based on current cognitive state.
        
        v6 UPGRADE:
        - Uses expanded contextual relevance (bigrams + co-occurrence)
        - Applies turn-based decay so older memories fade by conversation
          progress, not wall-clock time
        """
        state = self.cog_state.state
        params = self.cog_state.get_memory_retrieval_params()
        
        for item in memory_items:
            base_score = item.get("score", 0.0)
            modulated_score = base_score
            
            # Build text blob for analysis
            tags = item.get("tags", [])
            facts = item.get("facts", [])
            text_blob = " ".join(str(f) for f in facts) + " " + " ".join(str(t) for t in tags)
            text_blob_lower = text_blob.lower()
            
            # ─── CONTEXTUAL RELEVANCE BOOST (v6: expanded scope) ───
            ctx_relevance = self._compute_contextual_relevance(text_blob)
            if ctx_relevance > 0.1:
                # Turn-based decay: memories scored relative to conversation recency
                item_turn = item.get("birth_turn", 0)
                turn_age = max(0, self._current_turn - item_turn)
                turn_decay = 1.0 / (1.0 + turn_age / 10.0)
                modulated_score *= (1.0 + ctx_relevance * turn_decay * 0.5)
            
            # ─── IDENTITY BOOST ───
            if state.identity_adherence > 0.7:
                identity_keywords = self._indelible.extract_identity_keywords()
                if any(kw in text_blob_lower for kw in identity_keywords):
                    modulated_score *= (1.0 + params["identity_boost"] * 3.0)
            
            # ─── FRUSTRATION FILTERING ───
            if state.frustration > 0.6:
                if any(w in text_blob_lower for w in ["wonder", "philosophy", "existential"]):
                    modulated_score *= (1.0 - params["frustration_penalty"])
                if any(w in text_blob_lower for w in ["fix", "solution", "step"]):
                    modulated_score *= (1.0 + params["frustration_penalty"])
            
            # ─── CURIOSITY AMPLIFICATION ───
            if state.curiosity > 0.7:
                open_loops = item.get("open_loops", [])
                if open_loops:
                    modulated_score *= (1.0 + params["curiosity_filter"] * 0.5)
            
            # ─── COGNITIVE LOAD FILTERING ───
            if state.cognitive_load > 0.7:
                complexity = len(text_blob)
                if complexity > 200:
                    modulated_score *= (1.0 - state.cognitive_load * 0.3)
            
            item["state_modulated_score"] = max(0.0, modulated_score)
        
        return memory_items
    
    def get_top_memories_stateful(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get top K memories with state-aware re-scoring."""
        index_path = self._data_dir / "memory_index.json"
        if not index_path.exists():
            return []
        
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            items = index_data.get("items", [])
        except Exception:
            return []
        
        # Apply state modulation
        items = self._apply_state_modulation(items)
        
        # Sort by modulated score
        items.sort(key=lambda x: x.get("state_modulated_score", 0.0), reverse=True)
        
        return items[:k]
    
    def build_long_memory_block_stateful(self, max_bullets: int = 10) -> str:
        """
        Build long memory block with state-aware prioritization.
        
        STRUCTURE:
        1. Indelible facts (learned, not hardcoded)
        2. Master summary
        3. Top memories (state-weighted)
        """
        state = self.cog_state.state
        
        # Get master summary
        master_path = self._data_dir / "master_summary.json"
        if master_path.exists():
            try:
                master = json.loads(master_path.read_text(encoding="utf-8"))
            except Exception:
                master = {}
        else:
            master = {}
        
        # Get state-aware top memories
        top_items = self.get_top_memories_stateful(k=max_bullets)
        
        # Build output
        lines = []
        
        # ─── INDELIBLE FACTS (Learned, not hardcoded) ───
        indelible_section = self._indelible.format_for_prompt(max_facts=20)
        if indelible_section:
            lines.append(indelible_section)
            lines.append("")
        
        # ─── MASTER SUMMARY ───
        lines.append("[MASTER SUMMARY]")
        facts = master.get("facts", [])[:5]
        projects = master.get("active_projects", [])[:3]
        if facts:
            lines.append(f"Facts: {facts}")
        if projects:
            lines.append(f"Projects: {projects}")
        lines.append("")
        
        # ─── TOP MEMORIES (State-Weighted) ───
        lines.append("[TOP MEMORIES (State-Weighted)]")
        for item in top_items:
            score = item.get("state_modulated_score", 0.0)
            tags = item.get("tags", [])
            facts = item.get("facts", [])
            
            # Build compact bullet
            parts = []
            if tags:
                parts.append(f"tags={','.join(str(t) for t in tags[:2])}")
            if facts and isinstance(facts, list) and facts:
                parts.append(f"fact={str(facts[0])[:50]}")
            
            bullet = " | ".join(parts) if parts else f"id={item.get('id', '?')}"
            lines.append(f"- [{score:.3f}] {bullet}")
        
        return "\n".join(lines)
    
    def notify_new_message(self):
        """Pass through to TWDC."""
        if hasattr(self.twdc, 'notify_new_message'):
            self.twdc.notify_new_message()


# SINGLETON (CLI / root-dir usage)
_stateful_twdc = None

def get_stateful_twdc(data_dir=None):
    global _stateful_twdc
    if data_dir is not None:
        return StatefulTWDCWrapper(data_dir)    # per-user: fresh instance
    if _stateful_twdc is None:
        _stateful_twdc = StatefulTWDCWrapper()  # CLI: lazy-init global
    return _stateful_twdc

def load_long_memory_block_stateful(max_bullets: int = 10) -> str:
    return get_stateful_twdc().build_long_memory_block_stateful(max_bullets)
