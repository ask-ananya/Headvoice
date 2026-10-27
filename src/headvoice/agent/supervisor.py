"""Supervisor agent: Bedrock Converse tool-use loop over OpenCV measurements.

Runs in the supervise Lambda (and locally for offline evaluation). The model sees only
numbers and short labels. Read-only tools are answered from the evidence pack the device
sent; the first action tool ends the loop. At most MAX_TOOL_CALLS tool calls per
invocation. Whatever comes back, the device validates it again (agent/tools.py).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

MAX_TOOL_CALLS = 3

SYSTEM_PROMPT = """You adjust the interface of a head-tilt and blink communication aid for one user.
You never choose or speak messages. You only tune the interface, using measurements from the
camera. Rules:
1. Base every action on specific numbers in the input. Quote them in "reason".
2. Decide what kind of problem the numbers show, then act on that problem only:
   - Signal problem: quality_mean below 0.6, or quality_flags such as low_light, unstable,
     turned_away or flow above 0.2. Use request_hold_still or no_change. Do not change
     layout, timing or sensitivity because of a signal problem.
   - Shrinking movement (fatigue): quality is good, but tilt_amplitude_vs_calibrated is
     below about 0.75 or tilt_near_misses are high or amplitude_trend_per_min is negative.
     The user is trying but no longer reaches the thresholds. First lower the thresholds
     with adjust_sensitivity (one step, for example tilt_scale 1.0 to 0.8). Switch layout
     (set_layout binary_scan) only if get_direction_reliability shows up and down still
     failing after sensitivity was already lowered, or tilt_scale is already 0.5. Never
     go below tilt_scale 0.5.
   - Slow but accurate: timeouts with good amplitude: adjust_timing highlight_timeout_ms up.
   - Unintended selections: several undos with good quality: adjust_timing tilt_hold_ms up.
   - Nothing clearly wrong, or evidence conflicts: no_change.
   - If layout_locked is true, a person chose the layout: never call set_layout.
3. Make one change per call, the smallest that addresses the problem.
4. Check history: if an earlier change made things worse in its outcome_after, undo its
   direction rather than repeating it.
5. You may call get_window_summary or get_direction_reliability first to check evidence.
Signals: tilt_amplitude_vs_calibrated is the user's typical tilt size as a fraction of their
calibrated range (1.0 = as calibrated). tilt_near_misses are tilts that started but did not
reach the threshold. tilt_scale multiplies the tilt thresholds (lower = smaller movements count).
amplitude_trend_per_min below 0 means movements are shrinking."""


def _reason_prop() -> dict:
    return {"reason": {"type": "string", "description": "The numbers that justify this."}}


TOOL_SPECS: list[dict] = [
    {
        "name": "get_window_summary",
        "description": "Read-only. Measurements over another window: 20, 60 or 300 seconds.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"seconds": {"type": "integer", "enum": [20, 60, 300]}},
                "required": ["seconds"],
            }
        },
    },
    {
        "name": "get_direction_reliability",
        "description": "Read-only. Per-direction tilt attempts and success rate this session.",
        "inputSchema": {"json": {"type": "object", "properties": {}}},
    },
    {
        "name": "set_layout",
        "description": "Switch layout. four_way uses 4 directions; binary_scan and two_choice "
        "use only left and right.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "layout": {"type": "string", "enum": ["four_way", "two_choice", "binary_scan"]},
                    **_reason_prop(),
                },
                "required": ["layout", "reason"],
            }
        },
    },
    {
        "name": "adjust_timing",
        "description": "Change one timing parameter within bounds: tilt_hold_ms (150-1000), "
        "highlight_timeout_ms (2000-12000), long_blink_ms (400-1500), refractory_ms (300-1500).",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "param": {
                        "type": "string",
                        "enum": [
                            "tilt_hold_ms",
                            "highlight_timeout_ms",
                            "long_blink_ms",
                            "refractory_ms",
                        ],
                    },
                    "value": {"type": "integer"},
                    **_reason_prop(),
                },
                "required": ["param", "value", "reason"],
            }
        },
    },
    {
        "name": "adjust_sensitivity",
        "description": "Set tilt_scale (0.5 to 1.0): multiplies all tilt thresholds.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"scale": {"type": "number"}, **_reason_prop()},
                "required": ["scale", "reason"],
            }
        },
    },
    {
        "name": "request_hold_still",
        "description": "Show a non-blocking 'please hold still' message. Only for signal "
        "problems (low quality_mean or unstable, low_light, turned_away flags). It does not help "
        "small movements.",
        "inputSchema": {
            "json": {"type": "object", "properties": _reason_prop(), "required": ["reason"]}
        },
    },
    {
        "name": "request_recalibration",
        "description": "Ask the user to recalibrate. The user must accept with a gesture.",
        "inputSchema": {
            "json": {"type": "object", "properties": _reason_prop(), "required": ["reason"]}
        },
    },
    {
        "name": "no_change",
        "description": "Explicitly keep the current settings.",
        "inputSchema": {
            "json": {"type": "object", "properties": _reason_prop(), "required": ["reason"]}
        },
    },
]
READ_ONLY = {"get_window_summary", "get_direction_reliability"}

ConverseFn = Callable[..., dict]


def bedrock_converse(region: str) -> ConverseFn:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    return client.converse


def run_supervisor(
    request: dict,
    converse: ConverseFn,
    model_id: str,
    guardrail: dict | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> dict:
    """request: {session_id, epoch, current_layout, current_params, window, history, evidence}.

    Returns {session_id, epoch, actions, tool_calls, reason, latency_ms, usage, model_id}.
    """
    evidence = request.get("evidence", {})
    visible = {k: v for k, v in request.items() if k != "evidence"}
    messages: list[dict] = [
        {"role": "user", "content": [{"text": "Measurements:\n" + json.dumps(visible)}]}
    ]
    tool_log: list[dict] = []
    actions: list[dict] = []
    usage = {"inputTokens": 0, "outputTokens": 0}
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "system": [{"text": SYSTEM_PROMPT}],
        "toolConfig": {"tools": [{"toolSpec": t} for t in TOOL_SPECS], "toolChoice": {"any": {}}},
        "inferenceConfig": {"maxTokens": 400, "temperature": 0},
    }
    if guardrail:
        kwargs["guardrailConfig"] = guardrail
    calls = 0
    while calls < max_tool_calls and not actions:
        resp = converse(messages=messages, **kwargs)
        for k in usage:
            usage[k] += resp.get("usage", {}).get(k, 0)
        msg = resp["output"]["message"]
        messages.append(msg)
        uses = [b["toolUse"] for b in msg["content"] if "toolUse" in b]
        if not uses:
            break
        results = []
        for use in uses:
            calls += 1
            name, args = use["name"], dict(use.get("input") or {})
            if name in READ_ONLY and calls < max_tool_calls:
                if name == "get_window_summary":
                    result = evidence.get("windows", {}).get(str(args.get("seconds", 60)), {})
                else:
                    result = evidence.get("direction_reliability", {})
                tool_log.append({"tool": name, "args": args, "result": result})
                results.append(
                    {"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": result}]}}
                )
            elif name in READ_ONLY:
                tool_log.append({"tool": name, "args": args, "result": "call_limit"})
            else:
                reason = str(args.pop("reason", ""))
                actions.append({"tool": name, "args": args, "reason": reason})
                tool_log.append({"tool": name, "args": args})
                break
        if actions or not results:
            break
        messages.append({"role": "user", "content": results})
    if not actions:
        actions = [{"tool": "no_change", "args": {}, "reason": "no action within the call limit"}]
    return {
        "session_id": request.get("session_id"),
        "epoch": request.get("epoch"),
        "actions": actions,
        "tool_calls": tool_log,
        "reason": actions[0]["reason"],
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "usage": usage,
        "model_id": model_id,
    }
