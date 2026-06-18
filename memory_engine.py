# memory_engine.py
import json
import shutil
import time
from pathlib import Path

MAX_RECENT = 12

# Module-level default paths — anchored to repo root, not cwd
_DEFAULT_MEMORY_PATH = Path(__file__).parent / "memory_log.json"
_DEFAULT_CORRUPT_DIR = Path(__file__).parent / "memory_corrupt_backups"


class MemoryEngine:
    """
    Handles memory_log.json read/write for a specific data directory.
    data_dir=None means use root (CLI default behavior).
    """

    def __init__(self, data_dir=None):
        # Anchor the CLI default to the repo root, NOT cwd — launching from
        # any other directory must not fork a fresh memory (Cerberus rule).
        base = Path(data_dir) if data_dir else Path(__file__).parent
        self._path = base / "memory_log.json"
        self._corrupt_dir = base / "memory_corrupt_backups"

    def _load_all(self):
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                print(f"[MEMORY] WARNING: memory_log.json root is {type(data).__name__}, expected list. Backing up.")
                self._backup_corrupt_file()
                return []
            return data
        except json.JSONDecodeError as e:
            print(f"[MEMORY] CORRUPTION DETECTED in memory_log.json: {e}")
            self._backup_corrupt_file()
            return []
        except Exception as e:
            print(f"[MEMORY] ERROR reading memory_log.json: {e}")
            return []

    def _backup_corrupt_file(self):
        try:
            self._corrupt_dir.mkdir(exist_ok=True)
            backup_name = f"memory_log_corrupt_{int(time.time())}.json"
            shutil.copy2(self._path, self._corrupt_dir / backup_name)
            print(f"[MEMORY] Corrupted file backed up to {self._corrupt_dir / backup_name}")
        except Exception as e:
            print(f"[MEMORY] Could not backup corrupted file: {e}")

    def _save_all(self, rows):
        """Atomic write via temp file — prevents corruption from interrupted writes."""
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self._path)  # atomic on POSIX
        except Exception as e:
            print(f"[MEMORY] ERROR writing memory_log.json: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def save_interaction(self, user_text: str, bot_text: str, state_dict: dict = None):
        rows = self._load_all()
        row = {"ts": time.time(), "user": user_text, "bot": bot_text}
        if state_dict:
            row["state"] = state_dict  # 13D cognitive state at time of turn
        rows.append(row)
        self._save_all(rows)

    def save_initiative(self, bot_text: str, state_dict: dict = None):
        """Record a daemon-initiated message (a ping the bot sent unprompted).
        Same memory log as conversation so the bot remembers what it reached
        out about — no user turn, flagged initiative:true so load/archive can
        tell self-talk from a real exchange."""
        rows = self._load_all()
        row = {"ts": time.time(), "user": "", "bot": bot_text, "initiative": True}
        if state_dict:
            row["state"] = state_dict
        rows.append(row)
        self._save_all(rows)

    def load_recent_memory(self, n: int = MAX_RECENT) -> str:
        rows = self._load_all()[-n:]
        if not rows:
            return "(none)"
        lines = []
        for r in rows:
            # Initiative rows have no user turn — render the bot reaching out
            # on its own so it reads as self-talk, not a reply to nothing.
            if r.get("initiative"):
                lines.append(f"SignalBot (reached out on its own): {r.get('bot', '')}")
            else:
                lines.append(f"User: {r.get('user', '')}")
                lines.append(f"SignalBot: {r.get('bot', '')}")
            lines.append("---")
        return "\n".join(lines)


# ── Backward-compat module-level functions (CLI / signalbot.py) ──
# These use the root-directory instance so existing imports keep working.
_default = MemoryEngine()

def save_interaction(user_text: str, bot_text: str, state_dict: dict = None):
    _default.save_interaction(user_text, bot_text, state_dict)

def load_recent_memory(n: int = MAX_RECENT) -> str:
    return _default.load_recent_memory(n)
