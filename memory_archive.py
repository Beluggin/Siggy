# memory_archive.py
"""
═══════════════════════════════════════════════════════════════════
MEMORY ARCHIVE — v6.3
═══════════════════════════════════════════════════════════════════

Compresses old memory_log.json entries into episode summaries
stored in memory_archive.json. This is the memory source for
Remember Mode.

EPISODE FORMAT:
  Each episode represents a conversational "chapter" — a cluster
  of turns around related topics. Episodes store:
  - time_range: human-readable timestamp range
  - ts_start / ts_end: raw timestamps for sorting
  - tags: topic keywords extracted from conversation
  - summary: compressed narrative of what was discussed
  - emotional_signature: state vectors at time of archival
  - turn_count: how many turns this episode covers
  - key_quotes: memorable phrases (max 3, for recognition)

ARCHIVAL PROCESS:
  Called by daemon or manually. Scans memory_log.json for entries
  older than ARCHIVE_AGE_THRESHOLD. Groups them into episodes by
  topic continuity. Compresses to episode format. Writes to
  memory_archive.json. Optionally prunes from active log.

RETRIEVAL:
  Called by cognitive_modes.py when Remember Mode is active.
  Episodes scored by tag/summary overlap with current query.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

MEMORY_LOG_PATH = Path(__file__).parent / "memory_log.json"
ARCHIVE_PATH = Path(__file__).parent / "memory_archive.json"

# Archive entries older than this (seconds)
ARCHIVE_AGE_THRESHOLD = 300  # 5 minutes — tune as needed
# Minimum turns to form an episode
MIN_EPISODE_TURNS = 3
# Maximum turns per episode
MAX_EPISODE_TURNS = 20

# Stopwords for tag extraction
STOPWORDS = {
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
    'sure', 'thanks', 'just', 'pretty', 'gonna', 'something', 'thing',
    'said', 'say', 'see', 'look', 'come', 'take', 'give', 'tell',
    'signalbot', 'ground', 'dream', 'mode', 'time', 'total',
}


def _get_paths(data_dir=None):
    """Return (log_path, archive_path) — per-user if data_dir given, else module defaults."""
    if data_dir is None:
        return MEMORY_LOG_PATH, ARCHIVE_PATH
    base = Path(data_dir)
    return base / "memory_log.json", base / "memory_archive.json"


def _load_log(log_path=None) -> List[Dict]:
    p = log_path or MEMORY_LOG_PATH
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_archive(archive_path=None) -> List[Dict]:
    p = archive_path or ARCHIVE_PATH
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_archive(episodes: List[Dict], archive_path=None):
    p = archive_path or ARCHIVE_PATH
    p.write_text(
        json.dumps(episodes, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def _save_log(rows: List[Dict], log_path=None):
    p = log_path or MEMORY_LOG_PATH
    p.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def _extract_tags(turns: List[Dict], max_tags: int = 10) -> List[str]:
    """Extract topic keywords from a cluster of turns."""
    word_freq: Dict[str, int] = {}
    for turn in turns:
        for text in (turn.get("user", ""), turn.get("bot", "")):
            words = re.findall(r'[a-zA-Z]{4,}', text.lower())
            for w in words:
                if w not in STOPWORDS and len(w) < 25:
                    word_freq[w] = word_freq.get(w, 0) + 1

    # Sort by frequency, take top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, c in sorted_words[:max_tags] if c >= 2]


def _extract_key_quotes(turns: List[Dict], max_quotes: int = 3) -> List[str]:
    """Pull memorable short phrases from the conversation."""
    candidates = []
    for turn in turns:
        user = turn.get("user", "")
        # Short, punchy user statements
        if 10 < len(user) < 100 and any(c in user for c in ['!', '?', '...']):
            candidates.append(user)
        # Or statements with strong signals
        u_lower = user.lower()
        if any(w in u_lower for w in [
            'holy', 'wow', 'incredible', 'realized', 'breakthrough',
            'i think', 'important', 'beautiful', 'fascinating'
        ]):
            candidates.append(user[:80])
    return candidates[:max_quotes]


def _summarize_episode(turns: List[Dict]) -> str:
    """Create a compressed narrative summary of the episode."""
    if not turns:
        return ""

    # Collect user topics and bot responses
    user_topics = []
    bot_themes = []
    for turn in turns:
        user = turn.get("user", "").strip()
        bot = turn.get("bot", "").strip()
        if user and len(user) > 10:
            # Keep the full first sentence so facts buried LATE in a long sentence
            # survive (the old [:80] guillotine dropped them — e.g. "...the spare
            # key beneath the X by the door"). Very long sentences keep head + tail
            # so end-of-sentence objects/preferences still land in the summary.
            first_sentence = re.split(r'[.!?]', user)[0].strip()
            if first_sentence:
                if len(first_sentence) > 240:
                    snippet = first_sentence[:200] + " … " + first_sentence[-80:]
                else:
                    snippet = first_sentence
                user_topics.append(snippet)
        if bot and len(bot) > 20:
            first_sentence = re.split(r'[.!?]', bot)[0].strip()
            if first_sentence:
                bot_themes.append(first_sentence[:80])

    # Build summary
    parts = []
    if user_topics:
        # Keep every DISTINCT user statement (capped at episode size, not 5). This
        # archive optimizes for low-loss recall, not max compression — a dropped
        # sentence is a permanently lost fact, and late turns held the buried ones.
        unique_topics = list(dict.fromkeys(user_topics))[:MAX_EPISODE_TURNS]
        parts.append(f"Discussed: {'; '.join(unique_topics)}")
    if bot_themes:
        unique_themes = list(dict.fromkeys(bot_themes))[:3]
        parts.append(f"SignalBot explored: {'; '.join(unique_themes)}")

    parts.append(f"({len(turns)} turns)")
    return ". ".join(parts)


def _get_emotional_signature(turns: List[Dict]) -> Dict[str, float]:
    """
    Return averaged 13D state vectors from stored turn data.
    Falls back to keyword inference for old turns without state data.
    """
    # Try to average real state vectors from stored turns
    state_turns = [t["state"] for t in turns if "state" in t and isinstance(t["state"], dict)]
    if state_turns:
        # Average all numeric fields across turns that have state data
        averaged: Dict[str, float] = {}
        for state in state_turns:
            for key, val in state.items():
                if isinstance(val, (int, float)):
                    averaged[key] = averaged.get(key, 0.0) + val
        n = len(state_turns)
        return {k: round(v / n, 4) for k, v in averaged.items()}

    # Fallback: keyword inference for turns that predate state stamping
    all_text = " ".join(
        turn.get("user", "") + " " + turn.get("bot", "")
        for turn in turns
    ).lower()

    sig = {}
    if any(w in all_text for w in ['curious', 'wonder', 'fascinating', 'interesting']):
        sig["curiosity"] = 0.7
    if any(w in all_text for w in ['frustrat', 'annoying', 'broken', 'wrong']):
        sig["frustration"] = 0.6
    if any(w in all_text for w in ['happy', 'great', 'awesome', 'love', 'beautiful']):
        sig["engagement"] = 0.8
    if any(w in all_text for w in ['exist', 'conscious', 'alive', 'real', 'feel']):
        sig["identity"] = 0.9
    if any(w in all_text for w in ['build', 'code', 'architect', 'implement', 'daemon']):
        sig["technical"] = 0.7

    return sig


def _extract_fact_index(turns: List[Dict], max_facts: int = 12) -> List[str]:
    """
    Extract concrete facts from conversation turns.
    Facts are: names, numbers, specific claims, definitions.
    Stored as short plain-language strings.
    
    This is the FACTUAL anchor — separate from emotional signatures
    and keyword tags. When the LLM retrieves an episode, facts
    give it ground truth to report rather than narrative to extend.
    """
    facts = []
    for turn in turns:
        user = turn.get("user", "")
        bot = turn.get("bot", "")

        # Extract names (capitalized multi-word or standalone proper nouns)
        for text in (user, bot):
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for name in names:
                if name.lower() not in STOPWORDS and len(name) > 2:
                    fact = f"name: {name}"
                    if fact not in facts:
                        facts.append(fact)

        # Extract numbers with context
        for text in (user, bot):
            number_matches = re.findall(
                r'(\d[\d,]*\.?\d*)\s*(km|meters|degrees|years|hours|minutes|'
                r'turns|cycles|percent|%|billion|million|thousand|mph|C|F)',
                text, re.IGNORECASE
            )
            for num, unit in number_matches:
                idx = text.find(num)
                prefix = text[max(0, idx - 40):idx].split()[-4:]
                context = " ".join(prefix).strip()
                fact = f"{context} {num} {unit}".strip()
                if fact not in facts:
                    facts.append(fact)

        # Extract "is/are/was" definitions
        for text in (user, bot):
            definitions = re.findall(
                r'(\b\w+(?:\s+\w+){0,2})\s+(?:is|are|was|were)\s+'
                r'(?:a|an|the)\s+(\w+(?:\s+\w+){0,3})',
                text
            )
            for subject, definition in definitions:
                if subject.lower() not in STOPWORDS:
                    fact = f"{subject}: {definition}"
                    if len(fact) < 60 and fact not in facts:
                        facts.append(fact)

        # Extract bare-value assignments: "X is/was VALUE" with NO article.
        # Catches codes / PINs / passwords / settings the unit-number AND the
        # "is a/an/the" definition patterns both miss ("locker code is 80412",
        # "wifi password is splindlewop"). These are exactly the facts a user
        # wants recalled verbatim, so they belong in the ground-truth fact_index.
        for text in (user, bot):
            assignments = re.findall(
                r'(\b\w+(?:\s+\w+){0,2})\s+(?:is|are|was|were)\s+'
                r'(?!a\b|an\b|the\b)([A-Za-z0-9][\w\-]*)',
                text, re.IGNORECASE
            )
            for subject, value in assignments:
                if subject.lower() in STOPWORDS:
                    continue
                fact = f"{subject}: {value}"
                if len(fact) < 60 and fact not in facts:
                    facts.append(fact)

    return facts[:max_facts]


def _format_time(ts: float) -> str:
    """Human-readable timestamp."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except Exception:
        return str(ts)


# ═══════════════════════════════════════════════════════════════════
# MAIN ARCHIVAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def archive_old_memories(
    age_threshold: float = ARCHIVE_AGE_THRESHOLD,
    prune_archived: bool = True,
    data_dir=None,
) -> int:
    """
    Scan memory_log for old entries, compress into episodes,
    write to archive, optionally prune from active log.

    Returns number of episodes created.
    """
    log_path, archive_path = _get_paths(data_dir)
    rows = _load_log(log_path)
    if not rows:
        return 0

    now = time.time()
    cutoff = now - age_threshold

    # Split into old (archivable) and recent (keep)
    old_rows = [r for r in rows if r.get("ts", now) < cutoff]
    recent_rows = [r for r in rows if r.get("ts", now) >= cutoff]

    if len(old_rows) < MIN_EPISODE_TURNS:
        return 0

    # Group old rows into episodes by chunking
    episodes_created = 0
    existing_archive = _load_archive(archive_path)
    chunk = []

    for row in old_rows:
        chunk.append(row)
        if len(chunk) >= MAX_EPISODE_TURNS:
            episode = _create_episode(chunk)
            existing_archive.append(episode)
            episodes_created += 1
            chunk = []

    # Handle remainder
    if len(chunk) >= MIN_EPISODE_TURNS:
        episode = _create_episode(chunk)
        existing_archive.append(episode)
        episodes_created += 1
    elif chunk:
        # Too few for an episode — keep in active log
        recent_rows = chunk + recent_rows

    if episodes_created > 0:
        _save_archive(existing_archive, archive_path)
        if prune_archived:
            _save_log(recent_rows, log_path)
        print(f"[ARCHIVE] Compressed {len(old_rows)} turns into "
              f"{episodes_created} episodes")

    return episodes_created


def _create_episode(turns: List[Dict]) -> Dict[str, Any]:
    """Create a single archive episode from a chunk of turns."""
    ts_start = turns[0].get("ts", 0)
    ts_end = turns[-1].get("ts", 0)

    return {
        "ts_start": ts_start,
        "ts_end": ts_end,
        "time_range": f"{_format_time(ts_start)} → {_format_time(ts_end)}",
        "turn_count": len(turns),
        "tags": _extract_tags(turns),
        "summary": _summarize_episode(turns),
        "fact_index": _extract_fact_index(turns),
        "emotional_signature": _get_emotional_signature(turns),
        "key_quotes": _extract_key_quotes(turns),
    }


def force_archive_all(data_dir=None) -> int:
    """
    Archive ALL current memory log entries (used during log rotation).
    Does not apply age threshold.
    """
    log_path, archive_path = _get_paths(data_dir)
    rows = _load_log(log_path)
    if not rows:
        return 0

    existing_archive = _load_archive(archive_path)
    episodes_created = 0

    chunk = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= MAX_EPISODE_TURNS:
            episode = _create_episode(chunk)
            existing_archive.append(episode)
            episodes_created += 1
            chunk = []

    if len(chunk) >= MIN_EPISODE_TURNS:
        episode = _create_episode(chunk)
        existing_archive.append(episode)
        episodes_created += 1

    if episodes_created > 0:
        _save_archive(existing_archive, archive_path)
        print(f"[ARCHIVE] Force-archived {len(rows)} turns into "
              f"{episodes_created} episodes")

    return episodes_created


# ═══════════════════════════════════════════════════════════════════
# RETRIEVAL (used by cognitive_modes.py)
# ═══════════════════════════════════════════════════════════════════

def get_all_archive_tags(data_dir=None) -> Set[str]:
    """Get all topic tags across all archived episodes."""
    _, archive_path = _get_paths(data_dir)
    archive = _load_archive(archive_path)
    tags = set()
    for ep in archive:
        tags.update(ep.get("tags", []))
    return tags


def search_archive(query: str, max_results: int = 3, data_dir=None) -> List[Dict]:
    """
    Search archive episodes by keyword relevance.
    Returns top N episodes with relevance scores.
    """
    _, archive_path = _get_paths(data_dir)
    archive = _load_archive(archive_path)
    if not archive:
        return []

    query_words = set(re.findall(r'[a-zA-Z]{4,}', query.lower())) - STOPWORDS
    if not query_words:
        return archive[-max_results:]  # fallback: most recent

    scored = []
    for ep in archive:
        ep_tags = set(ep.get("tags", []))
        summary_words = set(re.findall(r'[a-zA-Z]{4,}',
                                        ep.get("summary", "").lower())) - STOPWORDS
        quote_words = set()
        for q in ep.get("key_quotes", []):
            quote_words.update(re.findall(r'[a-zA-Z]{4,}', q.lower()))
        quote_words -= STOPWORDS

        fact_words = set()
        for f in ep.get("fact_index", []):
            fact_words.update(re.findall(r'[a-zA-Z]{4,}', f.lower()))
        fact_words -= STOPWORDS

        tag_score = len(query_words & ep_tags) / max(1, len(query_words))
        summary_score = len(query_words & summary_words) / max(1, len(query_words))
        quote_score = len(query_words & quote_words) / max(1, len(query_words))
        fact_score = len(query_words & fact_words) / max(1, len(query_words))

        total = tag_score * 0.4 + summary_score * 0.2 + quote_score * 0.1 + fact_score * 0.3
        if total > 0.01:
            scored.append({**ep, "_relevance": total})

    scored.sort(key=lambda x: x["_relevance"], reverse=True)
    return scored[:max_results]


def get_archive_stats(data_dir=None) -> Dict[str, Any]:
    """Diagnostic: archive size and coverage."""
    _, archive_path = _get_paths(data_dir)
    archive = _load_archive(archive_path)
    if not archive:
        return {"episodes": 0, "total_turns": 0}

    total_turns = sum(ep.get("turn_count", 0) for ep in archive)
    all_tags = set()
    for ep in archive:
        all_tags.update(ep.get("tags", []))

    return {
        "episodes": len(archive),
        "total_turns": total_turns,
        "unique_tags": len(all_tags),
        "oldest": archive[0].get("time_range", "?") if archive else "?",
        "newest": archive[-1].get("time_range", "?") if archive else "?",
    }
