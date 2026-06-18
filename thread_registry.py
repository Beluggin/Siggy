#thread_registry.py
"""
thread_registry.py
------------------
Tracks conversational threads across turns.

A "thread" is a topic that persists across multiple messages.
This registry:
- detects topic similarity
- updates thread metadata
- marks unresolved threads
- provides revival prompts
"""

import time
import uuid
from typing import Dict, Optional, List
from dataclasses import dataclass, field

@dataclass
class Thread:
    id: str
    topic: str
    summary: str
    last_active: float
    unresolved: bool = False
    turns: int = 0

class ThreadRegistry:
    def __init__(self):
        self.threads: Dict[str, Thread] = {}
        self.last_thread_id: Optional[str] = None

    def _similar(self, text: str, topic: str) -> bool:
        """
        Very cheap similarity heuristic for V1.
        Later we can upgrade to embeddings.
        """
        text_l = text.lower()
        topic_l = topic.lower()
        return topic_l in text_l or text_l in topic_l

    def update_from_turn(self, user_input: str, bot_output: str):
        """
        Called every turn. Detects whether the user is continuing
        an existing thread or starting a new one.
        """
        now = time.time()

        # Try to match an existing thread
        for th in self.threads.values():
            if self._similar(user_input, th.topic):
                th.last_active = now
                th.turns += 1
                th.unresolved = False
                self.last_thread_id = th.id
                return

        # No match → create a new thread
        new_id = str(uuid.uuid4())[:8]
        new_thread = Thread(
            id=new_id,
            topic=user_input[:80],
            summary=user_input[:200],
            last_active=now,
            unresolved=False,
            turns=1,
        )
        self.threads[new_id] = new_thread
        self.last_thread_id = new_id

    def get_stale_threads(self, max_age_seconds: float = 600) -> List[Thread]:
        """
        Returns threads that haven't been active recently.
        """
        now = time.time()
        return [
            th for th in self.threads.values()
            if (now - th.last_active) > max_age_seconds
        ]

    def mark_unresolved(self, thread_id: str):
        if thread_id in self.threads:
            self.threads[thread_id].unresolved = True

    def generate_revival_prompt(self, thread_id: str) -> Optional[str]:
        """
        Produces a question to revive a stale or unresolved thread.
        """
        th = self.threads.get(thread_id)
        if not th:
            return None

        return f"You mentioned '{th.topic}' earlier. Should we continue that thread?"

