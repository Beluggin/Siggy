# room.py
import json
import time
from pathlib import Path

class Room:
    def __init__(self, name, path="room.jsonl"):
        self.name = name
        self.path = Path(path)
        self.participants = {}  # name -> adapter fn

    def add_participant(self, name, adapter):
        """adapter: fn(transcript, my_name) -> response_text"""
        self.participants[name] = adapter
        self._append("system", f"{name} joined the room.")

    def say(self, speaker, content):
        """Host (you) or system speaks directly."""
        self._append(speaker, content)
        print(f"[{speaker}]: {content}\n")

    def invoke(self, name):
        """Have a participant generate their next turn."""
        if name not in self.participants:
            raise ValueError(f"{name} is not in the room")
        response = self.participants[name](self.transcript(), name)
        self._append(name, response)
        print(f"[{name}]: {response}\n")
        return response

    def transcript(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f]

    def _append(self, speaker, content):
        entry = {"ts": time.time(), "speaker": speaker, "content": content}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    # Continue the chat
    def run(self, rotation):
        """
        Round-robin loop. Press Enter to invoke the next participant.
        Type a message first to interject as Adam, then they respond to you.
        Commands: /q quit | /skip skip next speaker
        """
        print(f"\n[room: {self.name}]")
        print(f"[rotation: {' → '.join(rotation)}]")
        print(f"[commands: Enter=next | type to interject | /q | /skip]\n")
    
        turn = 0
        while True:
            try:
                next_speaker = rotation[turn % len(rotation)]
                user_input = input(f"[→ {next_speaker}]> ").strip()
                
                if user_input in ("/q", "/exit"):
                    break
                if user_input == "/skip":
                    turn += 1
                    continue
                if user_input:
                    self.say("Adam", user_input)
                
                self.invoke(next_speaker)
                turn += 1
            except KeyboardInterrupt:
                print("\n[bye]")
                break
            except Exception as e:
                print(f"\n[error] {e}\n[retrying same speaker]")
                continue
