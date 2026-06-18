#!/usr/bin/env python3
"""
OmniAddress v0.1 — Unified address encoding for SignalBot memories.

Address format: subject.verb.object.tense.negator
Modalities: language | visual | action

Retrieval is by embedding similarity (MiniLM), not keyword match.
New vocabulary entries accumulate automatically — variable from day 1.
Visual and linguistic memories live in the same address space natively.

Requires: pip install sentence-transformers --break-system-packages
"""

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════════
# ADDRESS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OmniAddress:
    subject:  str   # who/what is acting or being described
    verb:     str   # what is happening / the relationship
    object_:  str   # what is being acted on / described
    tense:    str   # "past" | "present" | "future" | "now"
    negator:  str   # "true" | "false"
    modality: str   # "language" | "visual" | "action"

    def to_string(self) -> str:
        return f"{self.subject}.{self.verb}.{self.object_}.{self.tense}.{self.negator}"

    def __str__(self):
        return self.to_string()


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE MAPS  (verb → description)
# These anchor the address vocabulary. New maps can be appended.
# ═══════════════════════════════════════════════════════════════════

LANGUAGE_MAPS = [
    ("said",     "agent stated something"),
    ("knows",    "agent holds a fact"),
    ("asked",    "agent posed a question"),
    ("wants",    "agent has a goal or desire"),
    ("did",      "agent completed an action"),
    ("is",       "agent is in a state"),
    ("likes",    "agent has a positive preference"),
    ("dislikes", "agent has a negative preference"),
    ("learned",  "agent acquired knowledge"),
    ("cannot",   "agent lacks a capability"),
]

VISUAL_MAPS = [
    ("detected",  "object is present in frame"),
    ("absent",    "object is not in frame"),
    ("moving",    "object is in motion"),
    ("near",      "object is close to robot"),
    ("far",       "object is distant from robot"),
    ("blocking",  "object is in the robot's path"),
    ("contains",  "scene has multiple objects"),
    ("at",        "object is in a spatial zone"),
    ("counted",   "multiple instances of object detected"),
    ("changed",   "object state changed since last frame"),
]

ACTION_MAPS = [
    ("executed",   "robot completed an action successfully"),
    ("failed",     "robot attempted but could not complete action"),
    ("navigating", "robot is en route to a destination"),
    ("observed",   "robot logged a sensory event"),
]


# ═══════════════════════════════════════════════════════════════════
# VOCABULARY  (grows automatically as new tokens appear)
# ═══════════════════════════════════════════════════════════════════

VOCAB_PATH = BASE_DIR / "omniaddress_vocab.json"

def load_vocab() -> Dict[str, List[str]]:
    if VOCAB_PATH.exists():
        try:
            return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Seed vocab from template maps
    all_verbs = [v for v, _ in LANGUAGE_MAPS + VISUAL_MAPS + ACTION_MAPS]
    return {
        "subject":  ["signalbot", "adam", "robot", "user", "object", "scene"],
        "verb":     all_verbs,
        "object_":  ["thing", "location", "goal", "fact", "action", "state",
                     "left_near", "left_far", "center_near", "center_far",
                     "right_near", "right_far"],
        "tense":    ["past", "present", "future", "now"],
        "negator":  ["true", "false"],
    }

def save_vocab(vocab: Dict[str, List[str]]):
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

def register_token(vocab: Dict[str, List[str]], slot: str, token: str) -> bool:
    """Add a new token to a slot's vocabulary if not already present."""
    token = token.lower().replace(" ", "_")
    if token not in vocab.get(slot, []):
        vocab.setdefault(slot, []).append(token)
        save_vocab(vocab)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# ENCODERS
# ═══════════════════════════════════════════════════════════════════

def _clean(s: str) -> str:
    return s.lower().strip().replace(" ", "_")


def encode_language(raw: str, agent: str = "signalbot",
                    verb: str = "knows", obj: str = "fact",
                    tense: str = "present", negated: bool = False,
                    vocab: Optional[Dict] = None) -> OmniAddress:
    """
    Encode a linguistic memory as an OmniAddress.
    v0.1: caller supplies slots. Future: infer from raw text automatically.
    """
    if vocab is None:
        vocab = load_vocab()
    subject = _clean(agent)
    verb_   = _clean(verb)
    object_ = _clean(obj)
    register_token(vocab, "subject", subject)
    register_token(vocab, "verb",    verb_)
    register_token(vocab, "object_", object_)
    return OmniAddress(subject, verb_, object_, tense,
                       "false" if negated else "true", "language")


def encode_visual(detection: Dict[str, Any],
                  vocab: Optional[Dict] = None) -> OmniAddress:
    """
    Encode a YOLO detection as an OmniAddress.
    Expected keys: class_name, confidence, bbox [x1,y1,x2,y2], frame_w, frame_h
    Spatial zone is inferred from bounding box centroid.
    """
    if vocab is None:
        vocab = load_vocab()

    cls   = _clean(detection.get("class_name", "object"))
    conf  = detection.get("confidence", 0.0)
    bbox  = detection.get("bbox", [0, 0, 0, 0])
    fw    = detection.get("frame_w", 640)
    fh    = detection.get("frame_h", 480)

    # Divide frame into 3 columns × 2 rows for spatial zone
    cx  = (bbox[0] + bbox[2]) / 2
    cy  = (bbox[1] + bbox[3]) / 2
    col = "left"   if cx < fw / 3      else ("right" if cx > 2 * fw / 3 else "center")
    row = "near"   if cy > fh * 0.6   else "far"   # lower in frame = physically closer
    zone = f"{col}_{row}"

    # Low confidence → "possible" rather than "detected"
    verb_ = "detected" if conf >= 0.5 else "possible"

    register_token(vocab, "subject",  cls)
    register_token(vocab, "object_",  zone)

    return OmniAddress(cls, verb_, zone, "now", "true", "visual")


def encode_action(action: str, success: bool = True,
                  target: str = "destination",
                  in_progress: bool = False,
                  vocab: Optional[Dict] = None) -> OmniAddress:
    """Encode a robot action as an OmniAddress."""
    if vocab is None:
        vocab = load_vocab()

    if in_progress:
        verb_  = "navigating"
        tense  = "present"
        negator = "true"
    else:
        verb_   = "executed" if success else "failed"
        tense   = "past"
        negator = "true" if success else "false"

    obj = _clean(target)
    register_token(vocab, "object_", obj)
    return OmniAddress("robot", verb_, obj, tense, negator, "action")


# ═══════════════════════════════════════════════════════════════════
# AUTO-PARSER  (T5-small fine-tuned on omni_phase1_corpus.jsonl)
# Loaded lazily from ./omniaddress_model/ — run train_omniaddress.py first.
# ═══════════════════════════════════════════════════════════════════

_parser_model     = None
_parser_tokenizer = None
_PARSER_MODEL_DIR = BASE_DIR / "omniaddress_model"

def _load_parser():
    """Load fine-tuned T5 parser. Returns True on success."""
    global _parser_model, _parser_tokenizer
    if not _PARSER_MODEL_DIR.exists():
        return False
    if _parser_model is not None:
        return True  # already loaded
    try:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        import torch
        print("[OMNI] Loading fine-tuned OmniAddress parser...")
        _parser_tokenizer = AutoTokenizer.from_pretrained(str(_PARSER_MODEL_DIR))
        _parser_model     = T5ForConditionalGeneration.from_pretrained(str(_PARSER_MODEL_DIR))
        _parser_model.eval()
        print("[OMNI] Parser ready")
        return True
    except Exception as e:
        print(f"[OMNI] Parser load failed ({e})")
        return False

def auto_parse(text: str) -> Optional[str]:
    """
    Parse natural language → OmniAddress string using the fine-tuned T5 model.
    Requires ./omniaddress_model/ (run train_omniaddress.py first).
    Returns None if the model isn't loaded yet.

    Example:
        auto_parse("The robot bumped into a chair") → "robot.detected.chair.now.false"
    """
    if not _load_parser():
        return None
    try:
        import torch
        inp = _parser_tokenizer(
            "omniaddress: " + text,
            return_tensors="pt", max_length=96, truncation=True,
        )
        with torch.no_grad():
            out = _parser_model.generate(**inp, max_new_tokens=24, num_beams=4)
        return _parser_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[OMNI] auto_parse error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING  (MiniLM, lazy-loaded)
# ═══════════════════════════════════════════════════════════════════

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("[OMNI] Loading all-MiniLM-L6-v2...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[OMNI] Model ready")
    return _model

def embed(text: str) -> List[float]:
    return get_model().encode(text, convert_to_numpy=True).tolist()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ═══════════════════════════════════════════════════════════════════
# MEMORY STORE
# ═══════════════════════════════════════════════════════════════════

STORE_PATH = BASE_DIR / "omniaddress_store.json"

@dataclass
class MemoryEntry:
    address:   str          # "subject.verb.object.tense.negator"
    modality:  str          # "language" | "visual" | "action"
    raw:       str          # original text or description
    timestamp: str
    embedding: List[float]


class MemoryStore:
    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self.entries: List[MemoryEntry] = []
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [MemoryEntry(**e) for e in data]
            print(f"[OMNI] Loaded {len(self.entries)} memories from store")
        except Exception as e:
            print(f"[OMNI] Store load error: {e}")

    def save(self):
        self.path.write_text(
            json.dumps([asdict(e) for e in self.entries], indent=2),
            encoding="utf-8"
        )

    def add(self, address: OmniAddress, raw: str) -> MemoryEntry:
        addr_str  = address.to_string()
        embedding = embed(addr_str)
        entry = MemoryEntry(
            address   = addr_str,
            modality  = address.modality,
            raw       = raw,
            timestamp = datetime.now().isoformat(),
            embedding = embedding,
        )
        self.entries.append(entry)
        self.save()
        print(f"[OMNI] +{address.modality:<8} {addr_str}")
        return entry

    def query(self, query_text: str, top_k: int = 5,
              modality: Optional[str] = None) -> List[Dict]:
        """Return top_k most similar memories. Optionally filter by modality."""
        if not self.entries:
            return []
        q_emb = embed(query_text)
        scored = []
        for e in self.entries:
            if modality and e.modality != modality:
                continue
            scored.append({"entry": e, "score": cosine_similarity(q_emb, e.embedding)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def summary(self):
        by_mod = {}
        for e in self.entries:
            by_mod[e.modality] = by_mod.get(e.modality, 0) + 1
        parts = " | ".join(f"{k} {v}" for k, v in sorted(by_mod.items()))
        print(f"[OMNI] Store: {parts}  ({len(self.entries)} total)")


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    store = MemoryStore()
    vocab = load_vocab()

    print("\n── Encoding memories ──")

    # Language
    store.add(encode_language("Adam told me he is AuDHD",
                agent="adam", verb="told", obj="identity", vocab=vocab),
              raw="Adam told me he is AuDHD")

    store.add(encode_language("SignalBot cannot yet do visual grounding",
                agent="signalbot", verb="cannot", obj="visual_grounding",
                negated=True, vocab=vocab),
              raw="SignalBot cannot yet do visual grounding")

    store.add(encode_language("The hex address architecture unifies modalities",
                agent="signalbot", verb="knows", obj="hex_address_architecture",
                vocab=vocab),
              raw="The hex address architecture unifies modalities")

    store.add(encode_language("Adam wants to build a persistent robot memory system",
                agent="adam", verb="wants", obj="persistent_robot_memory",
                vocab=vocab),
              raw="Adam wants to build a persistent robot memory system")

    # Visual (simulated YOLO detections)
    store.add(encode_visual({
        "class_name": "chair", "confidence": 0.91,
        "bbox": [50, 300, 200, 450], "frame_w": 640, "frame_h": 480
    }, vocab=vocab), raw="YOLO: chair 0.91 @ left_near")

    store.add(encode_visual({
        "class_name": "person", "confidence": 0.87,
        "bbox": [280, 100, 400, 380], "frame_w": 640, "frame_h": 480
    }, vocab=vocab), raw="YOLO: person 0.87 @ center_far")

    store.add(encode_visual({
        "class_name": "cup", "confidence": 0.76,
        "bbox": [500, 320, 580, 410], "frame_w": 640, "frame_h": 480
    }, vocab=vocab), raw="YOLO: cup 0.76 @ right_near")

    # Actions
    store.add(encode_action("forward", success=True, target="hallway", vocab=vocab),
              raw="robot moved forward toward hallway")
    store.add(encode_action("turn_left", success=False, target="obstacle", vocab=vocab),
              raw="robot failed to turn left — obstacle blocked")
    store.add(encode_action("scan", in_progress=True, target="living_room", vocab=vocab),
              raw="robot scanning living room")

    print()
    store.summary()

    # Retrieval tests — cross-modal is the key capability
    queries = [
        "what does signalbot know about itself",
        "what objects is the robot seeing right now",
        "what movement has the robot done",
        "what does adam want",
    ]
    for q in queries:
        print(f"\n── '{q}' ──")
        for r in store.query(q, top_k=3):
            e = r["entry"]
            print(f"  {r['score']:.3f}  [{e.modality:<8}]  {e.address}")
            print(f"         {e.raw}")
