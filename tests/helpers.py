"""Synthetic frame streams for state machine tests."""

from headvoice.interaction.state_machine import MachineState, initial_state, step
from headvoice.io.schemas import FrameFeatures, Profile

FPS = 30
ITEMS4 = ["need_water", "need_help", "call_family", "feel_pain"]


def frame(
    t: float,
    roll=0.0,
    pitch=0.0,
    eyes=1.0,
    eye_l=None,
    eye_r=None,
    quality=0.9,
    source="model",
    face=True,
    flags=None,
) -> FrameFeatures:
    return FrameFeatures(
        t=t,
        frame_idx=int(t * FPS),
        face_found=face,
        bbox=(200, 100, 200, 200) if face else None,
        landmark_source=source if face else "none",
        yaw=0.0 if face else None,
        pitch=pitch if face else None,
        roll=roll if face else None,
        eye_open_l=(eye_l if eye_l is not None else eyes) if face else None,
        eye_open_r=(eye_r if eye_r is not None else eyes) if face else None,
        quality=quality if face else 0.0,
        quality_flags=flags or [],
    )


class Sim:
    def __init__(self, profile: Profile | None = None, layout="four_way", items=None):
        self.p = profile or Profile()
        self.s: MachineState = initial_state(layout, items or ITEMS4)
        self.t = 0.0
        self.events: list[dict] = []

    def run(self, seconds: float, **kw) -> "Sim":
        for _ in range(max(1, round(seconds * FPS))):
            self.s, ev = step(self.s, frame(self.t, **kw), self.p, self.t)
            self.events += ev
            self.t += 1 / FPS
        return self

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def spoken(self) -> list[str]:
        return [e["item"] for e in self.events if e["type"] == "speak"]

    # Gestures (default profile: left = roll +, 12 deg threshold, 300 ms hold).
    def settle(self) -> "Sim":
        return self.run(0.5)

    def tilt(self, direction: str, deg=20.0, hold=0.5) -> "Sim":
        kw = {
            "left": {"roll": deg},
            "right": {"roll": -deg},
            "up": {"pitch": deg},
            "down": {"pitch": -deg},
        }[direction]
        return self.run(hold, **kw).run(0.3)

    def blink(self, seconds: float, **kw) -> "Sim":
        return self.run(seconds, eyes=0.2, **kw).run(0.2)
