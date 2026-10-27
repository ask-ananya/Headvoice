"""Which phrases are on screen. "None of these" moves to the next set."""

from __future__ import annotations

from headvoice.interaction.layouts import PAGE_SIZE
from headvoice.io.schemas import LayoutName, Phrasebook


class MenuSource:
    """Pages through an ordered phrase list. Suggestions, when present, come first."""

    def __init__(self, phrasebook: Phrasebook):
        self.phrasebook = phrasebook
        self.order: list[str] = [p.id for p in phrasebook.phrases]
        self.page = 0
        self.suggested: list[str] = []

    def set_suggestions(self, ids: list[str]) -> None:
        known = set(self.order)
        self.suggested = [i for i in ids if i in known]
        self.page = 0

    def _sequence(self) -> list[str]:
        rest = [i for i in self.order if i not in self.suggested]
        return self.suggested + rest

    def current(self, layout: LayoutName) -> list[str]:
        seq = self._sequence()
        n = PAGE_SIZE[layout]
        pages = max(1, -(-len(seq) // n))
        start = (self.page % pages) * n
        return seq[start : start + n]

    def next_set(self, layout: LayoutName) -> list[str]:
        self.page += 1
        return self.current(layout)

    def reset(self) -> None:
        self.page = 0
