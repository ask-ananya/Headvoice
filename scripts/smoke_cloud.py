"""Smoke test the deployed API. Reads ApiUrl and TokenSecretArn from cdk-outputs.json.

python scripts/smoke_cloud.py
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from headvoice.io.cloud import CloudClient  # noqa: E402


def main() -> int:
    out = json.loads((ROOT / "cdk-outputs.json").read_text())["HeadVoice"]
    url = out["ApiUrl"]
    token = subprocess.run(
        [
            "aws",
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            out["TokenSecretArn"],
            "--query",
            "SecretString",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
        check=True,
        shell=sys.platform == "win32",
    ).stdout.strip()
    c = CloudClient(url, token, timeout_s=30)
    ok = True

    def check(name, fn):
        nonlocal ok
        t0 = time.perf_counter()
        try:
            r = fn()
            print(f"PASS {name} ({(time.perf_counter() - t0) * 1000:.0f} ms): {str(r)[:160]}")
        except Exception as e:  # report and continue
            ok = False
            body = e.read().decode()[:200] if isinstance(e, urllib.error.HTTPError) else ""
            print(f"FAIL {name}: {e} {body}")

    def bad_token():
        try:
            CloudClient(url, "wrong-token").supervise({"session_id": "x", "epoch": 1})
        except urllib.error.HTTPError as e:
            assert e.code == 403, e.code
            return "403 as expected"
        raise AssertionError("bad token was accepted")

    req = {
        "session_id": "smoke",
        "epoch": 1,
        "profile_id": "default",
        "trigger": "amplitude_decline",
        "current_layout": "four_way",
        "current_params": {
            "tilt_hold_ms": 300,
            "highlight_timeout_ms": 4000,
            "long_blink_ms": 700,
            "refractory_ms": 600,
            "tilt_scale": 1.0,
        },
        "window": {
            "seconds": 60,
            "selections": 1,
            "undos": 0,
            "timeouts": 2,
            "tilt_attempts": 9,
            "tilt_near_misses": 6,
            "tilt_amplitude_vs_calibrated": {"left": 0.58, "right": 0.61},
            "amplitude_trend_per_min": -0.07,
            "mean_selection_time_s": 9.2,
            "quality_mean": 0.91,
            "quality_flags": {},
            "long_blink_success_rate": 1.0,
            "flow_fallback_fraction": 0.0,
        },
        "history": [],
        "evidence": {
            "windows": {"300": {"tilt_amplitude_vs_calibrated": {"left": 0.85, "right": 0.9}}},
            "direction_reliability": {"left": {"attempts": 5, "counted": 1}},
        },
    }
    check("authorizer rejects a bad token", bad_token)
    check("POST /supervise", lambda: c.supervise(req))
    check(
        "POST /suggest",
        lambda: c._req(
            "POST",
            "/suggest",
            {
                "variant": "evidence_first",
                "context": {
                    "gesture_context": "highlighted 'I need water' twice, cancelled by dropouts",
                    "signal_quality": 0.9,
                    "gesture_confidence": 0.8,
                },
                "phrasebook": json.loads((ROOT / "config" / "phrasebook_en.json").read_text())[
                    "phrases"
                ],
            },
            timeout=30,
        ),
    )
    check(
        "POST /audit",
        lambda: c.audit_put({"session_id": "smoke", "policy": "agent", "applied": []}),
    )
    check("GET /audit/smoke", lambda: len(c.audit_get("smoke")["items"]))
    check(
        "POST /calibrate upload_url",
        lambda: c._req("POST", "/calibrate", {"action": "upload_url"})["key"],
    )

    def bad_key():
        try:
            c._req(
                "POST",
                "/calibrate",
                {"action": "analyze", "key": "datasets/x.jsonl", "timestamps": []},
            )
        except urllib.error.HTTPError as e:
            assert e.code == 400
            return "400 as expected"
        raise AssertionError("non-clip key accepted")

    check("calibrate refuses keys outside calibration-clips/", bad_key)
    check("GET /judge", lambda: urllib.request.urlopen(url + "/judge", timeout=30).read()[:40])
    check(
        "GET /judge/datasets",
        lambda: json.loads(urllib.request.urlopen(url + "/judge/datasets", timeout=30).read()),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
