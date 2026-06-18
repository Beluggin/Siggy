# persistent_behavior.py
import json
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "behavior_log.json"

class PersistentBehaviorModifier:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self.data = {"events": []}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {"events": []}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def record_event(self, event: str, outcome: str, severity: str, notes: str = ""):
        self.data["events"].append(
            {
                "ts": time.time(),
                "event": event,
                "outcome": outcome,
                "severity": severity,
                "notes": notes,
                "resolved": False,
            }
        )
        self._save()

    def get_unresolved_events(self):
        return {e["event"] for e in self.data["events"] if not e.get("resolved", False)}

    def resolve_event(self, event_name: str):
        for e in self.data["events"]:
            if e["event"] == event_name:
                e["resolved"] = True
        self._save()

