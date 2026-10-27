"""The user's own messages, added by them or a helper. Stored only on this computer
(data/custom_messages.json) and shown before the built-in phrases."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from headvoice.io.schemas import Phrase, Phrasebook

MAX_LEN = 80
MAX_COUNT = 60


class CustomMessages:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.items: list[Phrase] = []
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.items = [Phrase.model_validate(p) for p in data.get("messages", [])]

    def add(self, text: str) -> Phrase:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("The message is empty.")
        if len(text) > MAX_LEN:
            raise ValueError(f"Keep it under {MAX_LEN} characters so it fits on a card.")
        if len(self.items) >= MAX_COUNT:
            raise ValueError(f"You can have up to {MAX_COUNT} of your own messages.")
        if any(p.text.lower() == text.lower() for p in self.items):
            raise ValueError("That message is already in your list.")
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:24] or "message"
        phrase = Phrase(id=f"custom_{slug}_{uuid.uuid4().hex[:6]}", text=text, group="custom")
        self.items.append(phrase)
        self._save()
        return phrase

    def remove(self, phrase_id: str) -> bool:
        before = len(self.items)
        self.items = [p for p in self.items if p.id != phrase_id]
        if len(self.items) != before:
            self._save()
            return True
        return False

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"messages": [p.model_dump() for p in self.items]}
        self.path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def merged(builtin: Phrasebook, custom: list[Phrase]) -> Phrasebook:
    """Custom messages first, then the built-in phrasebook."""
    return Phrasebook(language=builtin.language, phrases=[*custom, *builtin.phrases])
