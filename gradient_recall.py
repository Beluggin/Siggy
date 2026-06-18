# gradient_recall.py
"""
═══════════════════════════════════════════════════════════════════
GRADIENT RECALL — v0.1
═══════════════════════════════════════════════════════════════════

One scoring function over ALL memory, no modes, no thresholds:

    score(memory) = similarity(query, memory) × energy^λ

- similarity : embedding cosine distance (all-MiniLM-L6-v2).
               Retrieval is DISTANCE, not keyword match — the hex
               address insight applied to the archive.
- energy     : continuous TWDC-style temperature. Decays with age,
               RE-HEATS on recall (being remembered makes a memory
               recent again — promotion is just access).
- λ (lambda) : the gradient dial. λ=1 → energy matters, hot/recent
               memories win ("active mode"). λ→0 → temperature-blind,
               the whole timeline competes on pure semantic distance
               ("remember mode"). Driven by the existing resonance
               detector in cognitive_modes.py — keywords stop GATING
               a mode and instead slide the dial.

The three memory layers fall out as regions of λ × energy,
not as separate systems. This module does NOT touch the existing
recall path — rollback = don't wire it in.
"""

import json
import math
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

ROOT = Path(__file__).parent
ARCHIVE_PATH = ROOT / "memory_archive.json"
LOG_PATH = ROOT / "memory_log.json"
HEAT_PATH = ROOT / "recall_heat.json"          # re-heat sidecar: {item_id: last_recall_ts}
EMBED_CACHE_PATH = ROOT / "gradient_embed_cache.json"

# Energy tuning. Half-life in days: after 14 days untouched, a memory
# is at half temperature. Floor keeps cold memories findable at λ=1
# (they're damped, never erased — seamless gradient, no cliff).
ENERGY_HALF_LIFE_DAYS = 14.0
ENERGY_FLOOR = 0.02


# ═══════════════════════════════════════════════════════════════════
# MEMORY ITEMS — one unified view over active log + archive
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryItem:
    item_id: str          # stable id (hash of source content)
    source: str           # "active" | "archive"
    text: str             # what gets embedded
    ts: float             # when the memory happened (ts_end for episodes)
    payload: Dict[str, Any]   # the original record, untouched


def _item_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _episode_text(ep: Dict) -> str:
    # Everything the episode kept after compression — summary is the
    # stored gloss, tags the bare constituents, quotes the verbatim residue.
    parts = [ep.get("summary", "")]
    parts.append(" ".join(ep.get("tags", [])))
    parts.extend(ep.get("fact_index", []))
    parts.extend(ep.get("key_quotes", []))
    return " ".join(p for p in parts if p)


def load_memory_items(data_dir=None) -> List[MemoryItem]:
    """Load archive episodes + active log rows into one flat list."""
    base = Path(data_dir) if data_dir else ROOT
    items: List[MemoryItem] = []

    archive_p = base / "memory_archive.json"
    if archive_p.exists():
        try:
            for ep in json.loads(archive_p.read_text(encoding="utf-8")):
                text = _episode_text(ep)
                if text.strip():
                    items.append(MemoryItem(
                        item_id=_item_id(text),
                        source="archive",
                        text=text,
                        ts=ep.get("ts_end", 0),
                        payload=ep,
                    ))
        except Exception:
            pass

    log_p = base / "memory_log.json"
    if log_p.exists():
        try:
            for row in json.loads(log_p.read_text(encoding="utf-8")):
                text = (row.get("user", "") + " " + row.get("bot", "")).strip()
                if text:
                    items.append(MemoryItem(
                        item_id=_item_id(text),
                        source="active",
                        text=text,
                        ts=row.get("ts", 0),
                        payload=row,
                    ))
        except Exception:
            pass

    return items


# ═══════════════════════════════════════════════════════════════════
# ENERGY — temperature with re-heat
# ═══════════════════════════════════════════════════════════════════

def _load_heat() -> Dict[str, float]:
    if HEAT_PATH.exists():
        try:
            return json.loads(HEAT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_heat(heat: Dict[str, float]):
    HEAT_PATH.write_text(json.dumps(heat, indent=2), encoding="utf-8")


def energy(item: MemoryItem, heat: Dict[str, float], now: Optional[float] = None) -> float:
    """
    Temperature of a memory. Age counts from the LATER of:
    when it happened, or when it was last recalled. Recalling a
    memory literally makes it recent again — that's the re-heat.
    """
    now = now or time.time()
    effective_ts = max(item.ts, heat.get(item.item_id, 0))
    age_days = max(0.0, (now - effective_ts) / 86400.0)
    e = math.pow(2.0, -age_days / ENERGY_HALF_LIFE_DAYS)
    return max(ENERGY_FLOOR, e)


# ═══════════════════════════════════════════════════════════════════
# LAMBDA — the gradient dial, driven by resonance
# ═══════════════════════════════════════════════════════════════════

def lambda_from_resonance(peak: float) -> float:
    """
    Map resonance peak (0–1) to λ (1–0).
    No resonance → λ=1, recency rules. Strong nostalgia/temporal-gap
    signal → λ→0, the search goes temperature-blind.

    Squared, not linear: a single "remember when" (peak 0.4–0.5) should
    already mostly blind the search — A/B on the real archive showed
    linear λ=0.5 still let a hot sim-0.17 episode outrank the actual
    first-consciousness episode (cold, sim 0.53). TUNE on real input,
    same as CONF_THRESHOLD.
    """
    return max(0.0, 1.0 - peak) ** 2


def lambda_for_query(query: str) -> float:
    """Convenience: run the existing resonance detector on a raw query."""
    try:
        from cognitive_modes import detect_resonance
        sig = detect_resonance(query, "", active_memory_hit=False)
        return lambda_from_resonance(sig.peak())
    except Exception:
        return 1.0   # detector unavailable → behave like plain recency recall


# ═══════════════════════════════════════════════════════════════════
# EMBEDDINGS — cached, model loaded lazily
# ═══════════════════════════════════════════════════════════════════

_model = None
_embedder_override: Optional[Callable[[List[str]], List[List[float]]]] = None


def set_embedder(fn: Callable[[List[str]], List[List[float]]]):
    """Inject a fake embedder for offline tests (no model, no download)."""
    global _embedder_override
    _embedder_override = fn


def _embed(texts: List[str]) -> List[List[float]]:
    global _model
    if _embedder_override is not None:
        return _embedder_override(texts)
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return [v.tolist() for v in _model.encode(texts, normalize_embeddings=True)]


def _load_embed_cache() -> Dict[str, List[float]]:
    if EMBED_CACHE_PATH.exists():
        try:
            return json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def get_embeddings(items: List[MemoryItem]) -> Dict[str, List[float]]:
    """Embed all items, reusing the on-disk cache (keyed by item_id)."""
    cache = _load_embed_cache()
    missing = [it for it in items if it.item_id not in cache]
    if missing:
        vecs = _embed([it.text for it in missing])
        for it, v in zip(missing, vecs):
            cache[it.item_id] = v
        if _embedder_override is None:   # don't pollute cache from tests
            EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ═══════════════════════════════════════════════════════════════════
# RECALL — the one query
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RecallHit:
    score: float
    sim: float
    energy: float
    lam: float
    item: MemoryItem


def recall(query: str, k: int = 3, lam: Optional[float] = None,
           items: Optional[List[MemoryItem]] = None,
           reheat: bool = True) -> List[RecallHit]:
    """
    The whole gradient in one function.
    λ is computed from resonance unless passed explicitly.
    Returned items get re-heated (recall makes them recent).
    """
    if items is None:
        items = load_memory_items()
    if not items:
        return []
    if lam is None:
        lam = lambda_for_query(query)

    heat = _load_heat()
    embeds = get_embeddings(items)
    qvec = _embed([query])[0]
    now = time.time()

    hits: List[RecallHit] = []
    for it in items:
        vec = embeds.get(it.item_id)
        if vec is None:
            continue
        sim = max(0.0, _cosine(qvec, vec))
        e = energy(it, heat, now)
        hits.append(RecallHit(
            score=sim * math.pow(e, lam),
            sim=sim, energy=e, lam=lam, item=it,
        ))

    hits.sort(key=lambda h: h.score, reverse=True)
    top = hits[:k]

    if reheat and top:
        for h in top:
            heat[h.item.item_id] = now
        _save_heat(heat)

    return top


# ═══════════════════════════════════════════════════════════════════
# A/B HARNESS — old keyword path vs gradient, side by side
# ═══════════════════════════════════════════════════════════════════

def _fmt_hit(h: RecallHit) -> str:
    ep = h.item.payload
    when = ep.get("time_range", time.strftime("%Y-%m-%d", time.localtime(h.item.ts)))
    summary = ep.get("summary", h.item.text)[:90]
    return (f"  {h.score:.3f} (sim {h.sim:.2f} × E {h.energy:.2f}^λ{h.lam:.2f}) "
            f"[{when}] {summary}")


def ab_compare(query: str, k: int = 3):
    from memory_archive import search_archive
    print(f"\n{'═'*70}\nQUERY: {query}")

    lam = lambda_for_query(query)
    print(f"λ = {lam:.2f}  (resonance-driven; 1=recency rules, 0=temperature-blind)")

    print("\n─── OLD: search_archive (keyword overlap) ───")
    old = search_archive(query, max_results=k)
    if not old:
        print("  (no hits)")
    for ep in old:
        print(f"  {ep.get('_relevance', 0):.3f} [{ep.get('time_range','?')}] "
              f"{ep.get('summary','')[:90]}")

    print("\n─── NEW: gradient recall (sim × energy^λ) ───")
    for h in recall(query, k=k, reheat=False):   # no reheat during A/B
        print(_fmt_hit(h))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ab":
        ab_compare(" ".join(sys.argv[2:]))
    else:
        # Canned probe set: one recency-flavored, one nostalgia-flavored,
        # one identity, one unresolved — exercises the λ range.
        for q in [
            "what are we working on with the tank",
            "remember when we first talked about consciousness",
            "how have I changed since the beginning",
            "what about that thing we never finished",
        ]:
            ab_compare(q)
