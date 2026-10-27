"""Evaluation metrics (plan section 9.4) from replay or live events.

Task metrics need a scripted task (target events). Without one they are None and
reported as "not measured", never guessed.
"""

from __future__ import annotations

import numpy as np

from headvoice.io.schemas import FrameFeatures


def task_metrics(events: list[dict]) -> dict:
    # Onboarding practice tasks are not evaluation tasks: start at the first real one.
    first = next(
        (i for i, e in enumerate(events) if e["type"] == "task_start" and not e.get("practice")),
        None,
    )
    events = events[first:] if first is not None else []
    starts = [e for e in events if e["type"] == "task_start"]
    if not starts:
        return {
            "targets": None,
            "correct": None,
            "false_triggers": None,
            "task_success_rate": None,
            "selection_time_s_median": None,
            "retries_per_correct": None,
        }
    n_targets = len(starts[0]["targets"])
    target_t, hits, misses, times, retries, cur_retries = None, 0, 0, [], [], 0
    for e in events:
        if e["type"] == "target":
            target_t, cur_retries = e["t"], 0
        elif e["type"] in ("undo", "highlight_timeout"):
            cur_retries += 1
        elif e["type"] == "task_hit":
            hits += 1
            if target_t is not None:
                times.append(e["t"] - target_t)
            retries.append(cur_retries)
        elif e["type"] == "task_miss":
            misses += 1
    return {
        "targets": n_targets,
        "correct": hits,
        "false_triggers": misses,
        "task_success_rate": round(hits / n_targets, 3) if n_targets else None,
        "selection_time_s_median": round(float(np.median(times)), 2) if times else None,
        "retries_per_correct": round(float(np.mean(retries)), 2) if retries else None,
    }


def abstention(
    events: list[dict], frames: list[FrameFeatures], quality_min: float, win_s: float = 2.0
) -> dict:
    """Fraction of low-quality windows in which nothing was selected (should be 1.0)."""
    if not frames:
        return {"low_quality_windows": 0, "abstention_correct": None}
    t0, t1 = frames[0].t, frames[-1].t
    speaks = [e["t"] for e in events if e["type"] == "speak"]
    q = np.array([f.quality for f in frames])
    t = np.array([f.t for f in frames])
    low, correct = 0, 0
    w = t0
    while w + win_s <= t1:
        m = (t >= w) & (t < w + win_s)
        if m.sum() and np.median(q[m]) < quality_min:
            low += 1
            correct += not any(w <= s < w + win_s for s in speaks)
        w += win_s
    return {
        "low_quality_windows": low,
        "abstention_correct": round(correct / low, 3) if low else None,
    }


def interaction_counts(events: list[dict]) -> dict:
    count = lambda k: sum(1 for e in events if e["type"] == k)  # noqa: E731
    return {
        "speaks": count("speak"),
        "undos": count("undo"),
        "highlight_timeouts": count("highlight_timeout"),
        "none_of_these": count("none_of_these"),
        "tilts": count("tilt"),
        "armed": count("armed"),
        "arm_cancelled": count("arm_cancelled"),
        "drops_to_idle": sum(
            1 for e in events if e["type"] == "transition" and e.get("to") == "idle"
        ),
    }


def agent_metrics(
    traces: list[dict], price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0
) -> dict:
    if not traces:
        return {"decisions": 0}
    lat = [t["latency_ms"] for t in traces if t.get("latency_ms") is not None]
    tin = sum(t.get("usage", {}).get("inputTokens", 0) for t in traces)
    tout = sum(t.get("usage", {}).get("outputTokens", 0) for t in traces)
    acts = [a for t in traces for a in t.get("applied", [])]
    tools = [a["tool"] for t in traces for a in t.get("actions", [])]
    return {
        "decisions": len(traces),
        "latency_ms_p50": round(float(np.percentile(lat, 50))) if lat else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95))) if lat else None,
        "rejected_tool_calls": sum(1 for a in acts if a.get("rejected_reason")),
        "stale_discards": sum(1 for a in acts if a.get("rejected_reason") == "stale_epoch"),
        "request_failures": sum(
            1 for t in traces if str(t.get("rejected_reason", "")).startswith("request_failed")
        ),
        "no_change_rate": round(tools.count("no_change") / len(tools), 3) if tools else None,
        "actions": {k: tools.count(k) for k in sorted(set(tools))},
        "input_tokens": tin,
        "output_tokens": tout,
        "cost_usd": round(tin / 1e6 * price_in_per_mtok + tout / 1e6 * price_out_per_mtok, 5),
    }


def all_metrics(
    events: list[dict],
    frames: list[FrameFeatures],
    traces: list[dict],
    quality_min: float,
    prices: tuple[float, float] = (0.0, 0.0),
) -> dict:
    return {
        **task_metrics(events),
        **interaction_counts(events),
        **abstention(events, frames, quality_min),
        "agent": agent_metrics(traces, *prices),
    }
