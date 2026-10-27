import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import sim_users  # noqa: E402

from headvoice.io.schemas import Profile  # noqa: E402
from headvoice.replay.arms import arm_setup  # noqa: E402


def _run(arm, user, final=1.0, seed=3):
    cal = sim_users.calibrate(user, seed)
    prof, pol = arm_setup(arm, Profile(neutral_set=True), cal)
    return cal, sim_users.run_session(
        user,
        prof,
        pol,
        None,
        ["need_help", "call_family", "need_water"],
        final,
        seed=seed,
        max_s=120,
    )


def test_calibration_picks_scanning_for_limited_range():
    users = sim_users.population(10, 7)
    limited = next(u for u in users if u.kind == "limited_ud")
    typical = next(u for u in users if u.kind == "typical")
    assert sim_users.calibrate(limited, 1).layout == "binary_scan"
    assert sim_users.calibrate(typical, 1).layout == "four_way"


def test_simulated_user_completes_task_and_is_deterministic():
    u = next(u for u in sim_users.population(10, 7) if u.kind == "typical")
    _, a = _run("C", u)
    _, b = _run("C", u)
    assert a["metrics"]["task_success_rate"] == 1.0
    assert a["metrics"] == b["metrics"]
