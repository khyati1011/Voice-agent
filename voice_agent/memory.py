"""
memory.py — In-session memory for context-aware conversations.

Stores a rolling window of the last N turns and exposes:
  - get_context()  : formatted string for LLM context injection
  - add_turn()     : record a new user turn
  - record_file()  : track created files
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import datetime


@dataclass
class Turn:
    timestamp: str
    user_text: str
    intent: str


class SessionMemory:
    def __init__(self, max_turns: int = 10):
        self.turns: List[Turn] = []
        self.max_turns = max_turns
        self.files_created: int = 0
        self._created_file_names: List[str] = []

    def add_turn(self, user_text: str, intent: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.turns.append(Turn(timestamp=ts, user_text=user_text, intent=intent))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def record_file(self, path: str) -> None:
        self.files_created += 1
        self._created_file_names.append(path)

    def get_context(self) -> str:
        """Return a short context string to inject into LLM prompts."""
        if not self.turns:
            return ""
        lines = []
        for t in self.turns[-5:]:  # last 5 turns
            lines.append(f"[{t.timestamp}] ({t.intent}) {t.user_text}")
        if self._created_file_names:
            lines.append("Files created this session: " + ", ".join(self._created_file_names[-5:]))
        return "\n".join(lines)

    def get_history(self) -> List[Turn]:
        return list(self.turns)
