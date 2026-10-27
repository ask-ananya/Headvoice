"""Simulated users study. SIMULATION, NOT PEOPLE.

A model of a user drives the real Controller closed-loop at 30 fps: it reads the menu on
screen, plans head tilts and blinks toward its target message, and produces FrameFeatures
with noise, involuntary blinks, signal dropouts, low-light spells and fatigue. Each
simulated user is first calibrated with the real calibration analysis on a simulated
calibration clip, so arms C to E use profiles computed by the actual code.

    python eval/sim_users.py                 # arms A-D, 30 users, normal + fatigue
    python eval/sim_users.py --agent api     # adds arm E through the deployed /supervise

Parameters are anchored to the developer's measured sessions (see PARAMS_SOURCE); values
marked "assumed" are not measured. Results are reported only under "Simulation".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from headvoice.app.controller import Controller  # noqa: E402
from headvoice.calibration.analyze import analyze  # noqa: E402
from headvoice.calibration.protocol import STEPS, label_at, total_seconds  # noqa: E402
from headvoice.interaction import state_machine as sm  # noqa: E402
from headvoice.io.schemas import FrameFeatures, Phrasebook, Profile  # noqa: E402
from headvoice.replay.arms import ARMS, arm_setup  # noqa: E402
from headvoice.replay.metrics import all_metrics  # noqa: E402

FPS = 30
NL = chr(10)
FATIGUE_HORIZON_S = 120.0  # fatigue condition: reachable tilt is 50% after 2 minutes
GIVE_UP_S = 45.0
DT = 1 / FPS
RESULTS = ROOT / "eval" / "results"
PRICES = (1.0, 5.0)

PARAMS_SOURCE = {
    "tilt amplitude (roll)": "measured: developer sessions, roll p5/p95 about +/-22 deg during tilts",
    "tilt amplitude (pitch)": "measured: developer sessions, pitch p5/p95 about -11 / +13 deg",
    "pose jitter": "measured: filtered pose second-difference median 0.15 (roll) to 0.35 (yaw) deg",
    "long blink length": "measured: developer deliberate closures held 0.9 to 1.3 s (eye recordings)",
    "involuntary blinks": "assumed: 15 per minute, 100 to 300 ms (typical adult values)",
    "reaction time, hold time": "assumed",
    "dropouts, low-light spells": "assumed: 0.5 dropouts and 0.2 low-light spells per minute",
    "limited-range and tremor profiles": "assumed: up/down range 3 to 7 deg; jitter 1.5 to 2.5 deg",
}


@dataclass
class SimUser:
    uid: str
    kind: str  # typical, limited_ud, tremor
    amp: dict  # reachable tilt per direction, degrees
    jitter: float
    ramp_s: float
    hold_s: float
    reaction_s: float
    long_blink_s: float
    blink_per_min: float
    err_dir: float
    dropout_per_min: float
    lowlight_per_min: float


def population(n: int, seed: int) -> list[SimUser]:
    rng = np.random.default_rng(seed)
    users = []
    kinds = ["typical"] * (n * 2 // 5) + ["limited_ud"] * (n * 3 // 10)
    kinds += ["tremor"] * (n - len(kinds))
    for i, kind in enumerate(kinds):
        lr = rng.uniform(16, 24)
        ud = rng.uniform(10, 15) if kind != "limited_ud" else rng.uniform(3, 7)
        if kind == "tremor":
            lr, ud = rng.uniform(12, 18), rng.uniform(8, 12)
        users.append(
            SimUser(
                uid=f"sim{i:02d}",
                kind=kind,
                amp={
                    "left": lr * rng.uniform(0.9, 1.1),
                    "right": lr * rng.uniform(0.9, 1.1),
                    "up": ud * rng.uniform(0.9, 1.1),
                    "down": ud * rng.uniform(0.9, 1.1),
                },
                jitter=rng.uniform(0.2, 0.6) if kind != "tremor" else rng.uniform(1.5, 2.5),
                ramp_s=rng.uniform(0.2, 0.4),
                hold_s=rng.uniform(0.45, 0.8),
                reaction_s=rng.uniform(0.6, 1.4),
                long_blink_s=rng.uniform(0.9, 1.3),
                blink_per_min=rng.uniform(10, 20),
                err_dir=rng.uniform(0.02, 0.08),
                dropout_per_min=0.5,
                lowlight_per_min=0.2,
            )
        )
    return users


class Body:
    """Generates FrameFeatures from planned gestures plus background noise."""

    def __init__(
        self, u: SimUser, rng: np.random.Generator, fatigue_final: float, duration_s: float
    ):
        self.u, self.rng = u, rng
        self.fatigue_final, self.duration_s = fatigue_final, duration_s
        self.queue: deque = deque()  # per-frame (target pose (yaw, pitch, roll), eyes)
        self.blink_left = 0
        self.drop_left = 0
        self.low_left = 0
        self.t = 0.0

    def busy(self) -> bool:
        return bool(self.queue)

    def scale(self) -> float:
        x = min(1.0, self.t / max(self.duration_s, 1))  # reaches fatigue_final at duration_s
        return 1.0 - (1.0 - self.fatigue_final) * x

    def _pose(self, d: str, mag: float) -> tuple[float, float, float]:
        return {
            "left": (0, 0, mag),
            "right": (0, 0, -mag),
            "up": (0, mag, 0),
            "down": (0, -mag, 0),
        }[d]

    def tilt(self, d: str) -> None:
        if self.rng.random() < self.u.err_dir:
            d = {"left": "right", "right": "left", "up": "down", "down": "up"}[d]
        mag = self.u.amp[d] * self.scale()
        n_ramp, n_hold = max(1, int(self.u.ramp_s * FPS)), int(self.u.hold_s * FPS)
        for i in range(n_ramp):
            self.queue.append((self._pose(d, mag * (i + 1) / n_ramp), 1.0))
        for _ in range(n_hold):
            self.queue.append((self._pose(d, mag), 1.0))
        for i in range(n_ramp):
            self.queue.append((self._pose(d, mag * (1 - (i + 1) / n_ramp)), 1.0))
        self.rest(self.u.reaction_s)

    def long_blink(self) -> None:
        for _ in range(int(self.u.long_blink_s * FPS)):
            self.queue.append(((0, 0, 0), 0.08))
        self.rest(self.u.reaction_s)

    def double_blink(self) -> None:
        for seg in (0.15, -0.25, 0.15):
            for _ in range(int(abs(seg) * FPS)):
                self.queue.append(((0, 0, 0), 0.08 if seg > 0 else 1.0))
        self.rest(self.u.reaction_s)

    def rest(self, s: float) -> None:
        for _ in range(int(s * FPS)):
            self.queue.append(((0, 0, 0), 1.0))

    def frame(self, idx: int) -> FrameFeatures:
        pose, eyes = self.queue.popleft() if self.queue else ((0, 0, 0), 1.0)
        r = self.rng
        if self.blink_left == 0 and eyes > 0.5 and r.random() < self.u.blink_per_min / 60 * DT:
            self.blink_left = int(r.uniform(0.1, 0.3) * FPS)
        if self.blink_left:
            eyes, self.blink_left = 0.08, self.blink_left - 1
        if self.drop_left == 0 and r.random() < self.u.dropout_per_min / 60 * DT:
            self.drop_left = int(r.uniform(0.3, 1.0) * FPS)
        if self.low_left == 0 and r.random() < self.u.lowlight_per_min / 60 * DT:
            self.low_left = int(3 * FPS)
        t, self.t = self.t, self.t + DT
        if self.drop_left:
            self.drop_left -= 1
            return FrameFeatures(
                t=t, frame_idx=idx, face_found=False, quality=0.0, quality_flags=["no_face"]
            )
        q, flags = 0.92, []
        if self.low_left:
            self.low_left -= 1
            q, flags = 0.35, ["low_light"]
        j = r.normal(0, self.u.jitter, 3)
        e = float(np.clip(eyes + r.normal(0, 0.06), -0.2, 1.3))
        return FrameFeatures(
            t=t,
            frame_idx=idx,
            face_found=True,
            landmark_source="model",
            yaw=pose[0] + j[0],
            pitch=pose[1] + j[1],
            roll=pose[2] + j[2],
            eye_open_l=e + float(r.normal(0, 0.03)),
            eye_open_r=e + float(r.normal(0, 0.03)),
            quality=q,
            quality_flags=flags,
        )


def calibrate(u: SimUser, seed: int) -> Profile:
    """Simulated calibration clip through the real analysis."""
    rng = np.random.default_rng(seed)
    body = Body(u, rng, 1.0, 1e9)
    body.u = SimUser(
        **{**asdict(u), "err_dir": 0.0, "dropout_per_min": 0.0, "lowlight_per_min": 0.0}
    )
    frames = []
    for i in range(int(total_seconds() * FPS)):
        t = i / FPS
        lab, step, _ = label_at(t)
        start = sum(s.seconds for s in STEPS[:step])
        k = min(1.0, max(0.0, (t - start - 0.35) / u.ramp_s))
        pose = body._pose(lab, u.amp[lab] * k) if lab in u.amp else (0, 0, 0)
        eyes = 0.08 if lab == "closed" and t - start > 0.35 else 1.0
        body.queue.append((pose, eyes))
        frames.append(body.frame(i))
    res = analyze(frames, Profile(neutral_set=True))
    return res.profile


def plan(body: Body, snap: dict) -> None:
    """Choose the next gesture from what is on screen."""
    if body.busy() or snap["state"] in ("speaking", "idle", "recovering"):
        return
    menu, task = snap["menu"], snap["task"]

    def undo() -> None:  # the undo gesture this user's profile uses
        if snap["params"]["undo_gesture"] == "tilt_down":
            body.tilt("down")
        else:
            body.double_blink()

    if not task or not task["target"]:
        body.rest(0.5)
        return
    target, items, hi = task["target"], menu["items"], menu["highlighted"]
    if target not in items:
        if hi is not None or (
            menu["layout"] == "binary_scan" and menu["range"][1] - menu["range"][0] < len(items)
        ):
            undo()  # cancel the wrong choice, then ask for other messages
        else:
            body.long_blink()  # none of these
        return
    idx = items.index(target)
    if hi == idx:
        body.long_blink()
        return
    if menu["layout"] == "binary_scan":
        lo, hi_r = menu["range"]
        if hi is not None or not (lo <= idx < hi_r):
            if hi is not None or hi_r - lo < len(items):
                undo()
                return
            lo, hi_r = 0, len(items)
        mid = lo + (hi_r - lo + 1) // 2
        body.tilt("left" if idx < mid else "right")
    else:
        body.tilt(menu["slots"][idx])


def run_session(
    u: SimUser,
    profile: Profile,
    policy: str,
    transport,
    targets: list[str],
    fatigue_final: float,
    seed: int,
    max_s: float = 360.0,
    sessions_dir: Path | None = None,
) -> dict:
    pb = Phrasebook.load(ROOT / "config" / "phrasebook_en.json")
    c = Controller(
        profile.model_copy(deep=True),
        pb,
        sessions_dir=sessions_dir,
        policy=policy,
        transport=transport,
        background_agent=False,
    )
    c.events = deque()
    c.start(f"s_{u.uid}")
    c.start_task(targets=targets)
    rng = np.random.default_rng(seed)
    body = Body(u, rng, fatigue_final, FATIGUE_HORIZON_S)
    frames, speak_t, i = [], None, 0
    target_i, target_since = 0, 0.0
    while i < max_s * FPS:
        if c.task_i >= len(c.task):
            break
        if c.task_i != target_i:
            target_i, target_since = c.task_i, body.t
        elif body.t - target_since > GIVE_UP_S:  # a helper skips a target the user cannot reach
            c.skip_target()
            target_i, target_since = c.task_i, body.t
        if not body.busy():  # read the screen only when choosing the next gesture
            plan(body, c.snapshot())
        f = body.frame(i)
        frames.append(f)
        c.handle_features(f, f.t)
        if c.state.state == sm.SPEAKING:
            speak_t = f.t if speak_t is None else speak_t
            if f.t - speak_t >= 1.5:
                c.speech_ended(f.t)
                speak_t = None
        i += 1
    traces = [r.model_dump() for r in c.adapter.traces.records] if c.adapter else []
    m = all_metrics(list(c.events), frames, traces, profile.quality_min, PRICES)
    m["minutes"] = round(len(frames) / FPS / 60, 2)
    m["final_tilt_scale"] = c.profile.tilt_scale
    m["final_layout"] = c.profile.layout
    c.stop()
    return {"metrics": m, "traces": traces}


def export_samples(out: Path, n: int, seed: int, per_kind: int = 2) -> list[Path]:
    """Write replayable session logs of simulated users for the judge replay page.

    Each file is marked simulated (header "simulated": true, "sim_" file name) so the
    replay page labels it. Recorded with arm C (calibrated, no adaptation); the replay page
    can then replay it through any arm.
    """
    import tempfile

    pb = Phrasebook.load(ROOT / "config" / "phrasebook_en.json")
    ids = [p.id for p in pb.phrases]
    generic = Profile.load(ROOT / "config" / "default_profile.json").model_copy(
        update={"neutral_set": True}
    )
    users = population(n, seed)
    picked = [
        u
        for kind in ("typical", "limited_ud", "tremor")
        for u in [x for x in users if x.kind == kind][:per_kind]
    ]
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for u in picked:
        k = users.index(u)
        cal = calibrate(u, seed * 1000 + k)
        targets = list(np.random.default_rng(seed + k).choice(ids, 10, replace=False))
        prof, policy = arm_setup("C", generic, cal)
        for cond, final in (("normal", 1.0), ("fatigue", 0.5)):
            with tempfile.TemporaryDirectory() as tmp:
                run_session(
                    u,
                    prof,
                    policy,
                    None,
                    targets,
                    final,
                    seed=seed * 7919 + k * 31 + (cond == "fatigue"),
                    sessions_dir=Path(tmp),
                )
                src = next(Path(tmp).glob("session_*.jsonl"))
                name = f"sim_{u.kind}_{u.uid}_{cond}"
                dest = out / f"{name}.jsonl"
                lines = src.read_text(encoding="utf-8").splitlines()
                header = json.loads(lines[0])
                header.update(
                    {
                        "session_id": name,
                        "simulated": True,
                        "sim_kind": u.kind,
                        "condition": cond,
                        "note": "Simulated user (eval/sim_users.py), not a recording of a person.",
                    }
                )
                body = [json.dumps(header), *lines[1:]]
                dest.write_text(NL.join(body) + NL, encoding="utf-8")
                written.append(dest)
                print(f"wrote {dest.name}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--export-samples",
        metavar="DIR",
        default=None,
        help="write simulated session logs for the replay page instead of running the study",
    )
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agent", choices=["api", "local"], default=None)
    ap.add_argument("--arms", default=None, help="e.g. ABCD")
    args = ap.parse_args()
    if args.export_samples:
        export_samples(Path(args.export_samples), args.n, args.seed)
        return
    from run_eval import transport_for  # noqa: E402 (same folder)

    transport = transport_for(args.agent)
    arms = list(args.arms) if args.arms else [a for a in ARMS if a != "E" or transport is not None]
    pb = Phrasebook.load(ROOT / "config" / "phrasebook_en.json")
    ids = [p.id for p in pb.phrases]
    generic = Profile.load(ROOT / "config" / "default_profile.json").model_copy(
        update={"neutral_set": True}
    )
    users = population(args.n, args.seed)
    rows, traces = [], []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for k, u in enumerate(users):
        cal = calibrate(u, args.seed * 1000 + k)
        targets = list(np.random.default_rng(args.seed + k).choice(ids, 10, replace=False))
        for cond, final in (("normal", 1.0), ("fatigue", 0.5)):
            for arm in arms:
                prof, policy = arm_setup(arm, generic, cal)
                out = run_session(
                    u,
                    prof,
                    policy,
                    transport if arm == "E" else None,
                    targets,
                    final,
                    seed=args.seed * 7919 + k * 31 + (cond == "fatigue"),
                )
                m = out["metrics"]
                row = {
                    "user": u.uid,
                    "kind": u.kind,
                    "condition": cond,
                    "arm": arm,
                    "calibrated_layout": cal.layout,
                    **{
                        k2: m.get(k2)
                        for k2 in (
                            "task_success_rate",
                            "correct",
                            "false_triggers",
                            "selection_time_s_median",
                            "retries_per_correct",
                            "undos",
                            "highlight_timeouts",
                            "none_of_these",
                            "minutes",
                            "final_tilt_scale",
                            "final_layout",
                        )
                    },
                    "agent_decisions": m["agent"].get("decisions"),
                    "agent_cost_usd": m["agent"].get("cost_usd"),
                    "agent_latency_ms_p50": m["agent"].get("latency_ms_p50"),
                }
                rows.append(row)
                traces += [
                    {"user": u.uid, "condition": cond, "arm": arm, **t} for t in out["traces"]
                ]
                print(
                    f"{u.uid} {u.kind:10s} {cond:7s} {arm}: success={m['task_success_rate']} "
                    f"false={m['false_triggers']} time={m['selection_time_s_median']} "
                    f"layout={cal.layout} scale={m['final_tilt_scale']}",
                    flush=True,
                )
    with (RESULTS / "sim_arms.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "sim_traces.jsonl").write_text(
        "".join(json.dumps(t, default=str) + "\n" for t in traces)
    )
    summarize(rows, arms, args)


def _m(rows, key):
    v = [
        r[key]
        for r in rows
        if r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))
    ]
    return round(float(np.mean(v)), 3) if v else None


def summarize(rows, arms, args) -> None:
    lines = (
        [
            "# Simulated users study: SIMULATION, NOT PEOPLE",
            "",
            f"{args.n} simulated users (seed {args.seed}), each calibrated by the real calibration analysis on a "
            "simulated clip, then a 10-message scripted task per arm, closed loop against the real Controller. "
            "Fatigue condition: reachable tilt shrinks linearly to 50% after 2 minutes. A target "
            "not reached within 45 s is skipped (counted as not achieved).",
            "",
            "Parameter sources:",
            "",
        ]
        + [f"- {k}: {v}" for k, v in PARAMS_SOURCE.items()]
        + [""]
    )
    kinds = sorted({r["kind"] for r in rows})
    for cond in ("normal", "fatigue"):
        lines += [
            f"## {cond}",
            "",
            "| Metric | " + " | ".join(f"{a} ({ARMS[a]})" for a in arms) + " |",
            "|---" * (len(arms) + 1) + "|",
        ]
        for key in (
            "task_success_rate",
            "false_triggers",
            "selection_time_s_median",
            "retries_per_correct",
            "undos",
            "highlight_timeouts",
            "agent_decisions",
            "agent_cost_usd",
        ):
            vals = [
                _m([r for r in rows if r["condition"] == cond and r["arm"] == a], key) for a in arms
            ]
            lines.append(
                f"| {key} | " + " | ".join("n/a" if v is None else str(v) for v in vals) + " |"
            )
        lines.append("")
        lines += [
            "Task success by user type:",
            "",
            "| Type | " + " | ".join(arms) + " |",
            "|---" * (len(arms) + 1) + "|",
        ]
        for kd in kinds:
            vals = [
                _m(
                    [
                        r
                        for r in rows
                        if r["condition"] == cond and r["arm"] == a and r["kind"] == kd
                    ],
                    "task_success_rate",
                )
                for a in arms
            ]
            lines.append(
                f"| {kd} | " + " | ".join("n/a" if v is None else str(v) for v in vals) + " |"
            )
        lines.append("")
    layouts = {}
    for r in rows:
        if r["arm"] == arms[0] and r["condition"] == "normal":
            layouts.setdefault(r["kind"], []).append(r["calibrated_layout"])
    lines += (
        ["Calibration chose:", ""]
        + [
            f"- {k}: " + ", ".join(f"{lay} x{v.count(lay)}" for lay in sorted(set(v)))
            for k, v in layouts.items()
        ]
        + [""]
    )
    lines.append(
        "Limits: the user model is a simplification. It plans perfectly, does not learn, and its noise "
        "is synthetic. These numbers compare interface setups under the model; they are not evidence "
        "of how people with disabilities would perform."
    )
    (RESULTS / "sim_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
