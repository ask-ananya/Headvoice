"""Evaluation arms (plan section 9.3), shared by run_eval.py and the replay Lambda.

A  fixed menu, generic profile: default profile, four_way, no adaptation
B  dwell selection: highlight by tilt, confirm by holding still (no blink)
C  calibrated: the user's calibrated profile and layout, no adaptation
D  calibrated + rule policy: deterministic adaptation baseline
E  calibrated + supervisor agent: Bedrock supervisor adaptation
"""

from __future__ import annotations

from headvoice.io.schemas import Profile

ARMS = {
    "A": "Fixed menu, generic profile",
    "B": "Dwell selection (no blink)",
    "C": "Calibrated, no adaptation",
    "D": "Calibrated + rule policy",
    "E": "Calibrated + supervisor agent",
}


def arm_setup(arm: str, generic: Profile, calibrated: Profile | None) -> tuple[Profile, str]:
    """(profile, policy) for an arm. Neutral always comes from the recording's own profile,
    because logged pose is relative to it."""
    base = calibrated or generic
    if arm == "A":
        p = generic.model_copy(deep=True, update={"layout": "four_way", "confirm_mode": "blink"})
        return p.model_copy(update={"neutral": base.neutral, "neutral_set": True}), "none"
    if arm == "B":
        p = generic.model_copy(deep=True, update={"layout": "four_way", "confirm_mode": "dwell"})
        return p.model_copy(update={"neutral": base.neutral, "neutral_set": True}), "none"
    p = base.model_copy(deep=True, update={"confirm_mode": "blink"})
    return p, {"C": "none", "D": "rule", "E": "agent"}[arm]
