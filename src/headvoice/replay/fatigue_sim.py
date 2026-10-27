"""Simulated fatigue: head movements shrink over a session and get noisier.

Healthy volunteers do not fatigue like the target users, so the evaluation also replays
normal recordings with displacement from neutral scaled down linearly from 1.0 to
`final_scale`, plus Gaussian jitter. Clearly labeled as simulated in all results.
"""

from __future__ import annotations

import numpy as np

from headvoice.io.schemas import FrameFeatures

DEFAULT = {"final_scale": 0.5, "jitter_deg": 0.8, "onset_frac": 0.1, "seed": 0}


def scale_at(t: float, t0: float, t1: float, final_scale: float, onset_frac: float) -> float:
    if t1 <= t0:
        return 1.0
    start = t0 + onset_frac * (t1 - t0)
    x = float(np.clip((t - start) / max(t1 - start, 1e-6), 0.0, 1.0))
    return 1.0 - (1.0 - final_scale) * x


def apply(
    frames: list[FrameFeatures],
    final_scale: float = DEFAULT["final_scale"],
    jitter_deg: float = DEFAULT["jitter_deg"],
    onset_frac: float = DEFAULT["onset_frac"],
    seed: int = DEFAULT["seed"],
) -> list[FrameFeatures]:
    if not frames:
        return []
    rng = np.random.default_rng(seed)
    t0, t1 = frames[0].t, frames[-1].t
    out = []
    for f in frames:
        if f.yaw is None or f.pitch is None or f.roll is None:
            out.append(f)
            continue
        k = scale_at(f.t, t0, t1, final_scale, onset_frac)
        n = rng.normal(0.0, jitter_deg, 3)
        out.append(
            f.model_copy(
                update={
                    "yaw": f.yaw * k + float(n[0]),
                    "pitch": f.pitch * k + float(n[1]),
                    "roll": f.roll * k + float(n[2]),
                }
            )
        )
    return out
