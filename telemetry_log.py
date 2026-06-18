# telemetry_log.py
"""
Shadow telemetry log — external observability only.
SignalBot NEVER reads from this file. It is not wired into any
retrieval, memory, or prompt pipeline. It exists so you can analyze
runs externally without the bot being able to introspect its own raw data.

Format: JSONL — one JSON record per line, append-only.
(Switched from a single rewritten JSON array 2026-06-11: appends are O(1)
instead of rewriting the whole file every turn, a torn write loses one
line instead of the whole history, and JSONL is the native format for
dataset tooling. Legacy .json files auto-migrate on first write.)

Schema per record:
  turn             — turn number in the session
  timestamp        — unix epoch (float)
  model            — model name string
  tokens_in        — input token count (0 if backend doesn't report)
  tokens_out       — output token count (0 if backend doesn't report)
  prompt           — full assembled prompt sent to the LLM
  response         — raw LLM output
  cognitive_state  — 13D state vector dict at time of response
  daemon_cognition — daemon's background thinking injected into prompt (or null)
  daemon_initiative — proactive message the daemon sent this turn (or null)
"""

import json
import time
from pathlib import Path

# Stored alongside the other log files in the root dir — anchored to repo root, not cwd
TELEMETRY_PATH = Path(__file__).parent / "signalbot_telemetry.jsonl"


def _get_path(data_dir=None) -> Path:
    if data_dir is None:
        return TELEMETRY_PATH
    return Path(data_dir) / "signalbot_telemetry.jsonl"


def _migrate_legacy(path: Path):
    """One-time: convert the old single-array .json file to .jsonl lines.

    Runs only if the legacy file exists and the .jsonl doesn't yet — so it
    fires exactly once per location, then the legacy file is renamed to
    .json.migrated (kept as a backup, delete it once you trust the swap).
    """
    legacy = path.with_suffix(".json")
    if path.exists() or not legacy.exists():
        return
    try:
        records = json.loads(legacy.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            records = []
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        legacy.rename(legacy.with_suffix(".json.migrated"))
        print(f"[TELEMETRY] Migrated {len(records)} records: {legacy.name} → {path.name}")
    except Exception as e:
        print(f"[TELEMETRY] Migration failed (will log fresh): {e}")


def log_turn(
    turn_num: int,
    prompt: str,
    response: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cog_state: dict,
    daemon_cognition: str = None,
    daemon_initiative: str = None,
    data_dir=None,
):
    """Append one turn record as a single JSONL line."""
    path = _get_path(data_dir)
    _migrate_legacy(path)

    record = {
        "turn": turn_num,
        "timestamp": time.time(),
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "prompt": prompt,
        "response": response,
        "cognitive_state": cog_state,
        "daemon_cognition": daemon_cognition or None,
        "daemon_initiative": daemon_initiative or None,
    }

    try:
        # Append-only: one line per turn, no rewrite of existing data
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[TELEMETRY] Write failed: {e}")


def load_records(data_dir=None) -> list:
    """Read all records back as a list of dicts (for analysis/dataset scripts).

    Skips corrupt lines instead of dying — a torn write costs one record.
    """
    path = _get_path(data_dir)
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records
