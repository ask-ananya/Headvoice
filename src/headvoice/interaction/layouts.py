"""Menu layouts: how a counted tilt changes what is highlighted.

four_way    up, left, right, down each highlight one of 4 items
two_choice  left or right highlights one of 2 items
binary_scan left keeps the first half, right keeps the second half, until one item remains

"None of these" is in every layout: a long blink while nothing is highlighted asks for a
different set. It never speaks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from headvoice.io.schemas import Direction, LayoutName

SLOTS: dict[str, tuple[Direction, ...]] = {
    "four_way": ("up", "left", "right", "down"),
    "two_choice": ("left", "right"),
    "binary_scan": ("left", "right"),
}
PAGE_SIZE: dict[str, int] = {"four_way": 4, "two_choice": 2, "binary_scan": 8}


@dataclass(frozen=True)
class Menu:
    layout: LayoutName
    items: tuple[str, ...]
    lo: int = 0  # binary_scan: current range [lo, hi)
    hi: int = 0
    highlighted: int | None = None

    @classmethod
    def create(cls, layout: LayoutName, items: list[str] | tuple[str, ...]) -> Menu:
        items = tuple(items)[: PAGE_SIZE[layout]]
        return cls(layout=layout, items=items, lo=0, hi=len(items))

    def reset(self) -> Menu:
        return replace(self, lo=0, hi=len(self.items), highlighted=None)

    @property
    def highlighted_item(self) -> str | None:
        return None if self.highlighted is None else self.items[self.highlighted]

    @property
    def narrowed(self) -> bool:
        """True if binary_scan has narrowed the range but not down to one item."""
        return (
            self.layout == "binary_scan"
            and self.highlighted is None
            and (self.hi - self.lo) < len(self.items)
        )

    def apply_tilt(self, direction: Direction) -> Menu | None:
        """Return the new menu, or None if this direction does nothing in this layout."""
        if direction not in SLOTS[self.layout] or not self.items:
            return None
        if self.layout == "binary_scan":
            lo, hi = (0, len(self.items)) if self.highlighted is not None else (self.lo, self.hi)
            mid = lo + (hi - lo + 1) // 2
            lo, hi = (lo, mid) if direction == "left" else (mid, hi)
            if hi <= lo:
                return None
            return replace(self, lo=lo, hi=hi, highlighted=lo if hi - lo == 1 else None)
        idx = SLOTS[self.layout].index(direction)
        if idx >= len(self.items):
            return None
        return replace(self, highlighted=idx)

    def view(self) -> dict:
        """Plain dict for the UI and logs."""
        return {
            "layout": self.layout,
            "items": list(self.items),
            "slots": list(SLOTS[self.layout]),
            "range": [self.lo, self.hi],
            "highlighted": self.highlighted,
        }
