# indelible_facts.py
"""
═══════════════════════════════════════════════════════════════════
INDELIBLE FACTS - Learned Identity Anchoring (NON-CHEATING VERSION)
═══════════════════════════════════════════════════════════════════

This is how SignalBot LEARNS important facts organically instead of having
them hardcoded. When you say "My name is Adam", SignalBot detects this is
an identity statement and LOCKS it as "indelible" (never decays).

WHY THIS ISN'T CHEATING:
- SignalBot must detect the pattern in your language
- SignalBot must recognize it's important (explicit directive, correction, etc.)
- SignalBot must store it and retrieve it later
- This tests the ENTIRE memory pipeline

WHAT GETS LOCKED AS INDELIBLE:
1. Name statements: "My name is X"
2. Relationships: "My children are X, Y, Z"
3. Explicit directives: "Remember that...", "Never forget..."
4. Corrections: When you fix SignalBot's mistakes

INDELIBLE FACTS GET:
- Decay rate of 0.0 (they don't fade)
- Importance score of 5.0 (massive boost in TWDC)
- Always included in prompt when identity_adherence > 0.6
"""

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    # nltk not available — use a minimal fallback set
    STOP_WORDS = {
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
        'they', 'them', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
        'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'to', 'of', 'in',
        'on', 'at', 'for', 'and', 'or', 'but', 'not', 'so', 'if',
        'that', 'this', 'with', 'from', 'by', 'as', 'up', 'out', 'no',
    }
INDELIBLE_PATH = Path(__file__).parent / "indelible_facts.json"

# Names that are LLM/assistant identities — never the user's name. Guards against
# model-identity bleed ("I'm Gemini" got locked as importance-5.0 user name once).
MODEL_NAMES = {
    "gemini", "claude", "gpt", "chatgpt", "openai", "anthropic", "mistral",
    "gemma", "phi", "phi3", "llama", "sonnet", "opus", "haiku", "fable",
    "bard", "copilot", "bing", "deepseek", "qwen", "grok", "assistant",
    "ai", "bot", "signalbot", "model", "llm", "chatbot",
}

# Words that commonly follow "I'm" but are states/feelings, not names. With
# autocapitalize "I'm Tired" looks just like "I'm Adam" by capitalization alone.
NON_NAME_WORDS = {
    "tired", "sorry", "happy", "sad", "fine", "good", "great", "okay", "ok",
    "ready", "sure", "here", "back", "done", "busy", "hungry", "thirsty",
    "cold", "hot", "scared", "afraid", "confused", "curious", "excited",
    "bored", "lost", "late", "early", "old", "young", "right", "wrong",
    "serious", "kidding", "joking", "glad", "alive", "awake", "asleep",
    "well", "unwell", "sick", "ill", "free", "available", "out", "off",
    "down", "fed", "exhausted", "stuck", "worried", "nervous", "angry",
}

@dataclass
class IndelibleFact:
    """A fact that should never be forgotten."""
    id: str
    fact: str
    category: str  # "name", "relationship", "directive", "custom"
    first_mentioned: float
    last_confirmed: float
    confirmation_count: int = 1
    locked: bool = True
    importance: float = 5.0  # Massively high
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "fact": self.fact,
            "category": self.category,
            "first_mentioned": self.first_mentioned,
            "last_confirmed": self.last_confirmed,
            "confirmation_count": self.confirmation_count,
            "locked": self.locked,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'IndelibleFact':
        return cls(**d)


class IndelibleFactsEngine:
    """Detects and manages facts that should never decay."""

    def __init__(self, data_dir=None):
        # data_dir: per-user Path for web app; None = use module-level INDELIBLE_PATH (CLI)
        self._path = Path(data_dir) / "indelible_facts.json" if data_dir else INDELIBLE_PATH
        self.facts: Dict[str, IndelibleFact] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for fact_dict in data.get("facts", []):
                    fact = IndelibleFact.from_dict(fact_dict)
                    self.facts[fact.id] = fact
            except Exception:
                pass

    def _save(self):
        data = {
            "facts": [f.to_dict() for f in self.facts.values()],
            "last_updated": time.time()
        }
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    #═══════════════════════════════════════════════════════════
    # PATTERN DETECTION
    #═══════════════════════════════════════════════════════════
    
    def _is_bad_name(self, cand_lower: str) -> bool:
        """True if the candidate can't be a real user name: a model/assistant
        identity, a common state/feeling word, or a gerund ('going')."""
        return (cand_lower in MODEL_NAMES
                or cand_lower in NON_NAME_WORDS
                or (cand_lower.endswith("ing") and len(cand_lower) > 4))

    def _detect_name_statement(self, text: str) -> Optional[Dict]:
        """Detect 'My name is X' / "I'm X" name patterns. Filters out model-name
        bleed ('I'm Gemini') and state words ('I'm tired') — see _is_bad_name."""
        t = text.lower().strip()

        # "my name is X" — explicit phrasing, just guard against a bad capture.
        if "my name is" in t:
            parts = t.split("my name is", 1)
            if len(parts) == 2 and parts[1].strip():
                # Strip punctuation from the individual word, not the whole remainder
                name = parts[1].strip().split()[0].strip(".,!?'\"-")
                if (name and len(name) > 1 and name.lower() not in STOP_WORDS
                        and not self._is_bad_name(name.lower())):
                    return {
                        "category": "name",
                        "fact": f"User's name is {name.capitalize()}"
                    }

        # "I'm X" / "I am X" — far noisier, so accept ONLY a TERMINAL proper-noun
        # name: 1-2 capitalized alpha tokens and NOTHING after them. This is the
        # key filter — "I'm going home" / "I'm Adam the dev" carry trailing
        # content, a bare name declaration does not.
        #   "I'm Adam" / "I am Adam Smith"  → name
        #   "I'm tired" · "I'm Gemini" · "I'm going home"  → rejected
        m = re.match(r"i(?:'m|\s+am)\s+(.+)", text.strip(), re.IGNORECASE)
        if m:
            name_tokens = [tok.strip(".,!?'\"-") for tok in m.group(1).split()]
            if 1 <= len(name_tokens) <= 2:
                cand = name_tokens[0]
                cand_l = cand.lower()
                # a second token (surname) must also look like a name
                second_ok = (len(name_tokens) == 1) or (
                    name_tokens[1].isalpha() and name_tokens[1][:1].isupper())
                if (cand and len(cand) > 1 and cand.isalpha()
                        and cand[0].isupper()
                        and cand_l not in STOP_WORDS
                        and not self._is_bad_name(cand_l)
                        and second_ok):
                    return {
                        "category": "name",
                        "fact": f"User's name is {' '.join(name_tokens).title()}"
                    }

        return None
    
    def _detect_relationship_statement(self, text: str) -> Optional[Dict]:
        """Detect 'My children are X, Y, Z' patterns."""
        t = text.lower().strip()
    
        if "my children are" in t or "my kids are" in t:
            parts = t.split("are", 1)
            if len(parts) == 2:
                names_part = parts[1].strip().strip(".,!?")
                if names_part and len(names_part) > 1 and names_part.lower() not in STOP_WORDS:
                    return {
                        "category": "relationship",
                        "fact": f"User's children: {names_part}"
                    }

        # "my son/daughter is X"
        for marker in ["my son", "my daughter"]:
            if marker in t:
                parts = t.split(marker, 1)
                if len(parts) == 2:
                    rest = parts[1].strip()
                    if rest.startswith("is "):
                        name = rest.replace("is ", "").strip().strip(".,!?").split()[0]
                        if name and len(name) > 1 and name.lower() not in STOP_WORDS:
                            return {
                                "category": "relationship",
                                "fact": f"User's {marker.replace('my ', '')} is {name.capitalize()}"
                            }

        return None
    
    def _detect_explicit_directive(self, text: str) -> Optional[Dict]:
        """Detect explicit memory commands."""
        t = text.lower().strip()
        patterns = [
            ("remember that", "directive"),
            ("never forget", "directive"),
            ("always remember", "directive"),
            ("from now on", "directive"),
            ("don't forget", "directive")
        ]
        
        for pattern, category in patterns:
            if pattern in t:
                parts = t.split(pattern, 1)
                if len(parts) == 2:
                    directive = parts[1].strip().strip(".,!?")
                    if directive and len(directive) > 3:
                        return {"category": category, "fact": directive}
        return None
    
    def _detect_correction(self, user_text: str, bot_previous: str) -> Optional[Dict]:
        """Detect when user corrects the bot."""
        u = user_text.lower().strip()
        if any(marker in u for marker in ["no,", "actually,", "wrong", "incorrect"]):
            name_info = self._detect_name_statement(user_text)
            if name_info:
                return name_info
            rel_info = self._detect_relationship_statement(user_text)
            if rel_info:
                return rel_info
        return None
    
    #═══════════════════════════════════════════════════════════
    # REGISTRATION
    #═══════════════════════════════════════════════════════════
    
    def register_fact(self, user_input: str, bot_output: str = "") -> bool:
        """
        Scan user input for indelible facts and register them.
        Returns True if a new fact was registered.
        """
        detected = []
        detected.append(self._detect_name_statement(user_input))
        detected.append(self._detect_relationship_statement(user_input))
        detected.append(self._detect_explicit_directive(user_input))
        detected.append(self._detect_correction(user_input, bot_output))
        
        detected = [d for d in detected if d is not None]
        if not detected:
            return False
        
        now = time.time()
        registered_new = False
        
        for info in detected:
            fact_text = info["fact"]
            category = info["category"]
            fact_id = self._generate_id(fact_text)
            
            if fact_id in self.facts:
                # Update confirmation
                self.facts[fact_id].last_confirmed = now
                self.facts[fact_id].confirmation_count += 1
            else:
                # New fact
                self.facts[fact_id] = IndelibleFact(
                    id=fact_id,
                    fact=fact_text,
                    category=category,
                    first_mentioned=now,
                    last_confirmed=now
                )
                registered_new = True
        
        self._save()
        return registered_new
    
    def _generate_id(self, fact_text: str) -> str:
        """Generate stable ID from fact text."""
        normalized = fact_text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]
    
    #═══════════════════════════════════════════════════════════
    # RETRIEVAL
    #═══════════════════════════════════════════════════════════
    
    def get_all_facts(self) -> List[IndelibleFact]:
        """Get all facts, sorted by importance."""
        facts = list(self.facts.values())
        facts.sort(key=lambda f: (f.importance, f.confirmation_count), reverse=True)
        return facts
    
    def format_for_prompt(self, max_facts: int = 20) -> str:
        """
        Format for injection into prompt.
        These go at the TOP of CORE DATA section.
        """
        facts = self.get_all_facts()[:max_facts]
        if not facts:
            return ""
        
        lines = ["[INDELIBLE FACTS - NEVER FORGET]"]
        
        # Group by category (names first, then relationships, then directives)
        by_category: Dict[str, List] = {}
        for fact in facts:
            if fact.category not in by_category:
                by_category[fact.category] = []
            by_category[fact.category].append(fact)
        
        for category in ["name", "relationship", "directive"]:
            if category in by_category:
                for fact in by_category[category]:
                    lines.append(f"- {fact.fact}")
        
        return "\n".join(lines)
    
    def extract_identity_keywords(self) -> List[str]:
        """
        Extract keywords for TWDC alignment scoring.
        This REPLACES reading from signal_identity.txt for identity keywords.
        """
        keywords = []
        for fact in self.facts.values():
            words = fact.fact.lower().split()
            for word in words:
                word = word.strip(".,!?:;")
                if len(word) > 2 and word not in ["the", "is", "are", "and"]:
                    keywords.append(word)
        
        # Deduplicate
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:60]


# SINGLETON (CLI / root-dir usage)
_engine: Optional[IndelibleFactsEngine] = None

def get_indelible_engine(data_dir=None) -> IndelibleFactsEngine:
    global _engine
    if data_dir is not None:
        return IndelibleFactsEngine(data_dir)   # per-user: fresh instance
    if _engine is None:
        _engine = IndelibleFactsEngine()        # CLI: lazy-init global
    return _engine

# CONVENIENCE FUNCTIONS
def register_fact(user_input: str, bot_output: str = "") -> bool:
    return get_indelible_engine().register_fact(user_input, bot_output)

def get_indelible_prompt_section(max_facts: int = 20) -> str:
    return get_indelible_engine().format_for_prompt(max_facts)

def get_indelible_keywords() -> List[str]:
    return get_indelible_engine().extract_identity_keywords()
