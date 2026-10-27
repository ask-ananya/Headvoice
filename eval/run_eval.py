"""Run all evaluation arms on recorded sessions and write eval/results/.

    python eval/run_eval.py                       # arms A-D, local
    python eval/run_eval.py --agent api           # arm E through the deployed /supervise
    python eval/run_eval.py --agent local         # arm E calling Bedrock directly (same code)

Only sessions with log format 2 (current pipeline) are used; others are listed as
excluded. Each session is replayed as recorded and with simulated fatigue. Task metrics
need sessions recorded with a scripted task (operator key T); otherwise "not measured".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from headvoice import LOG_FORMAT  # noqa: E402
from headvoice.io.logger import read_session  # noqa: E402
from headvoice.io.schemas import Phrasebook, Profile  # noqa: E402
from headvoice.replay import fatigue_sim  # noqa: E402
from headvoice.replay.arms import ARMS, arm_setup  # noqa: E402
from headvoice.replay.engine import replay, task_targets  # noqa: E402
from headvoice.replay.metrics import all_metrics  # noqa: E402

RESULTS = ROOT / "eval" / "results"
PRICES = (1.0, 5.0)  # Haiku 4.5 USD per million tokens (confirm on the AWS pricing page)
FLAT = [
    "task_success_rate",
    "correct",
    "false_triggers",
    "selection_time_s_median",
    "retries_per_correct",
    "speaks",
    "undos",
    "highlight_timeouts",
    "none_of_these",
    "drops_to_idle",
    "abstention_correct",
    "low_quality_windows",
]
AGENT = [
    "decisions",
    "latency_ms_p50",
    "latency_ms_p95",
    "rejected_tool_calls",
    "stale_discards",
    "no_change_rate",
    "cost_usd",
]


def transport_for(mode: str | None):
    if mode == "api":
        from headvoice.io.cloud import CloudClient

        c = CloudClient.from_env()
        if c is None:
            raise SystemExit("set HEADVOICE_API_URL and HEADVOICE_API_TOKEN for --agent api")
        return c.supervise
    if mode == "local":
        from headvoice.agent.supervisor import bedrock_converse, run_supervisor

        conv = bedrock_converse(os.environ.get("AWS_REGION", "us-east-1"))
        model = os.environ.get(
            "HEADVOICE_BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        return lambda r: run_supervisor(r, conv, model)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=str(ROOT / "data" / "sessions" / "session_*.jsonl"))
    ap.add_argument("--agent", choices=["api", "local"], default=None, help="include arm E")
    ap.add_argument("--no-fatigue", action="store_true")
    args = ap.parse_args()

    pb = Phrasebook.load(ROOT / "config" / "phrasebook_en.json")
    generic = Profile.load(ROOT / "config" / "default_profile.json")
    transport = transport_for(args.agent)
    arms = [a for a in ARMS if a != "E" or transport is not None]
    pattern = Path(args.sessions)
    files = sorted(pattern.parent.glob(pattern.name))
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows, excluded, traces_out = [], [], []
    for path in files:
        header, frames, events = read_session(path)
        if header.get("format", 1) < LOG_FORMAT:
            excluded.append((path.name, f"log format {header.get('format', 1)} (older pipeline)"))
            continue
        if len(frames) < 300:
            excluded.append((path.name, f"too short ({len(frames)} frames)"))
            continue
        recorded = Profile.model_validate(header.get("profile", {}))
        calibrated = recorded if recorded.calibrated_range_deg is not None else None
        conditions = [("as_recorded", frames)]
        if not args.no_fatigue:
            conditions.append(("fatigue_sim", fatigue_sim.apply(frames)))
        for cond, fr in conditions:
            for arm in arms:
                profile, policy = arm_setup(arm, generic, calibrated or recorded)
                res = replay(
                    fr,
                    profile,
                    pb,
                    policy=policy,
                    transport=transport if arm == "E" else None,
                    targets=task_targets(events),
                    session_id=header.get("session_id", path.stem),
                )
                m = all_metrics(res.events, fr, res.traces, profile.quality_min, PRICES)
                row = {
                    "session": path.stem,
                    "recording": header.get("condition", "normal"),
                    "calibrated": calibrated is not None,
                    "condition": cond,
                    "arm": arm,
                    "minutes": round((fr[-1].t - fr[0].t) / 60, 2),
                }
                row.update({k: m.get(k) for k in FLAT})
                row.update({f"agent_{k}": m["agent"].get(k) for k in AGENT})
                rows.append(row)
                for t in res.traces:
                    traces_out.append({"session": path.stem, "condition": cond, "arm": arm, **t})
                print(
                    f"{path.stem} {cond:11s} {arm}: speaks={m['speaks']} undos={m['undos']} "
                    f"success={m['task_success_rate']} decisions={m['agent'].get('decisions')}"
                )

    with (RESULTS / "arms.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (RESULTS / "traces.jsonl").write_text(
        "".join(json.dumps(t, default=str) + "\n" for t in traces_out)
    )
    charts(rows)
    summary(rows, excluded, arms)
    print(f"wrote {RESULTS}")


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def charts(rows) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = sorted({r["condition"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    for key, title in [
        ("speaks", "Messages spoken per session"),
        ("undos", "Undos per session"),
        ("drops_to_idle", "Signal dropouts per session"),
        ("task_success_rate", "Task success rate"),
        ("false_triggers", "False triggers"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        width = 0.8 / max(len(conds), 1)
        any_data = False
        for i, c in enumerate(conds):
            vals = [
                _mean([r for r in rows if r["arm"] == a and r["condition"] == c], key) for a in arms
            ]
            any_data |= any(v is not None for v in vals)
            ax.bar(
                [j + i * width for j in range(len(arms))], [v or 0 for v in vals], width, label=c
            )
        ax.set_xticks(
            [j + width * (len(conds) - 1) / 2 for j in range(len(arms))], [f"{a}" for a in arms]
        )
        ax.set_title(title + ("" if any_data else " (not measured)"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / f"chart_{key}.png", dpi=130)
        plt.close(fig)


def summary(rows, excluded, arms) -> None:
    lines = ["# Evaluation summary (generated by eval/run_eval.py)", ""]
    sessions = sorted({r["session"] for r in rows})
    lines.append(
        f"Sessions used: {len(sessions)}. Arms: {', '.join(f'{a} ({ARMS[a]})' for a in arms)}."
    )
    if "E" not in arms:
        lines.append("Arm E (supervisor agent) not run in this pass: use --agent api.")
    lines.append(
        "Fatigue rows are simulated (recorded head movement scaled down to 50% over the session)."
    )
    lines.append(
        "How to read this: each recorded session is replayed open loop through every setup. A replay cannot "
        "show how the person would have reacted to a different layout, so counts like messages spoken compare "
        "setups only roughly, and task metrics are meaningful only for sessions recorded with a scripted task. "
        "abstention_correct is the share of 2-second windows with low signal quality in which nothing was selected."
    )
    lines.append("")
    if excluded:
        lines += ["Excluded sessions:", ""] + [f"- {n}: {why}" for n, why in excluded] + [""]
    if not rows:
        lines.append("No eligible sessions: every metric is not measured.")
    else:
        for cond in sorted({r["condition"] for r in rows}):
            lines += [
                f"## {cond}",
                "",
                "| Metric | " + " | ".join(arms) + " |",
                "|---" * (len(arms) + 1) + "|",
            ]
            for key in FLAT + [f"agent_{k}" for k in AGENT]:
                vals = [
                    _mean([r for r in rows if r["arm"] == a and r["condition"] == cond], key)
                    for a in arms
                ]
                cells = ["not measured" if v is None else str(v) for v in vals]
                lines.append(f"| {key} | " + " | ".join(cells) + " |")
            lines.append("")
    if rows and all(r["task_success_rate"] is None for r in rows):
        lines.append(
            "Task metrics are not measured: no session had a scripted task (operator key T)."
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
