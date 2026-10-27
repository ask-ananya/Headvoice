"""Deterministic adaptation baseline (evaluation arm D).

Fixed if/then rules, each on one number, in priority order. The supervisor agent is
compared against this to show whether combining signals and querying evidence helps.
"""

from __future__ import annotations

from headvoice.agent.tools import ToolCall
from headvoice.io.schemas import Profile


def decide(summary: dict, profile: Profile) -> list[ToolCall]:
    if summary["quality_mean"] < 0.6:
        return [
            ToolCall(tool="request_hold_still", reason=f"quality_mean {summary['quality_mean']}")
        ]
    amps = summary.get("tilt_amplitude_vs_calibrated") or {}
    low = min(amps.values()) if amps else 1.0
    if low < 0.75 and profile.tilt_scale > 0.55:
        scale = round(max(0.5, profile.tilt_scale - 0.15), 2)
        return [
            ToolCall(
                tool="adjust_sensitivity", args={"scale": scale}, reason=f"min amplitude {low}"
            )
        ]
    if summary["timeouts"] >= 2:
        value = min(12000, profile.highlight_timeout_ms + 1500)
        return [
            ToolCall(
                tool="adjust_timing",
                args={"param": "highlight_timeout_ms", "value": value},
                reason=f"timeouts {summary['timeouts']}",
            )
        ]
    if summary["undos"] >= 3 and profile.layout == "four_way":
        return [
            ToolCall(
                tool="set_layout",
                args={"layout": "binary_scan"},
                reason=f"undos {summary['undos']}",
            )
        ]
    return [ToolCall(tool="no_change", reason="no rule fired")]
