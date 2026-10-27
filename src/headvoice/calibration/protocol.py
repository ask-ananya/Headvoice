"""The guided calibration clip: what the user is asked to do, and when.

Shared by the app (prompts), local analysis and the calibrate Lambda (labels by time).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    label: str  # neutral, left, right, up, down, natural, open, closed
    seconds: float
    prompt: str


STEPS: tuple[Step, ...] = (
    Step("neutral", 2.5, "Look at the screen and relax"),
    Step("left", 2.0, "Tilt your head to the LEFT and hold"),
    Step("neutral", 1.2, "Back to the middle"),
    Step("right", 2.0, "Tilt your head to the RIGHT and hold"),
    Step("neutral", 1.2, "Back to the middle"),
    Step("up", 2.0, "Tilt your head UP and hold"),
    Step("neutral", 1.2, "Back to the middle"),
    Step("down", 2.0, "Tilt your head DOWN and hold"),
    Step("neutral", 1.2, "Back to the middle"),
    Step("natural", 3.5, "Look at the screen and blink normally"),
    Step("open", 1.0, "Eyes open"),
    Step("closed", 1.4, "Close your eyes and hold (low tone), open at the high tone"),
    Step("open", 1.0, "Eyes open"),
    Step("closed", 1.4, "Close your eyes and hold"),
    Step("open", 1.0, "Eyes open"),
    Step("closed", 1.4, "Close your eyes and hold"),
    Step("open", 1.0, "Done. Eyes open"),
)
REACTION_S = 0.5  # the first part of each step is ignored (time to react to the prompt)


def total_seconds() -> float:
    return sum(s.seconds for s in STEPS)


def step_at(t: float) -> tuple[int, Step] | None:
    """Step index and step for a time since the clip started, or None after the end."""
    acc = 0.0
    for i, s in enumerate(STEPS):
        if t < acc + s.seconds:
            return i, s
        acc += s.seconds
    return None


def label_at(t: float) -> tuple[str, int, bool] | None:
    """(label, step index, settled) where settled is False during the reaction time."""
    acc = 0.0
    for i, s in enumerate(STEPS):
        if t < acc + s.seconds:
            return s.label, i, t - acc >= REACTION_S
        acc += s.seconds
    return None
