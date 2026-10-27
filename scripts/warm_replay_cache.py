"""Precompute agent (arm E) replays for every sample on the judge page.

API Gateway stops requests after 30 s; an agent replay of a long session makes several
sequential Bedrock calls and takes longer. This invokes the judge replay Lambda directly
(5 minute timeout), which stores each result in S3 so the page can serve it at once.

    python scripts/warm_replay_cache.py
"""

import json
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    out = json.loads((ROOT / "cdk-outputs.json").read_text())["HeadVoice"]
    s3 = boto3.client("s3")
    lam = boto3.client(
        "lambda", config=boto3.session.Config(read_timeout=320, retries={"max_attempts": 0})
    )
    keys = [
        o["Key"]
        for o in s3.list_objects_v2(Bucket=out["BucketName"], Prefix="sample/").get("Contents", [])
        if o["Key"].endswith(".jsonl")
    ]
    failures = 0
    for key in sorted(keys):
        for fatigue in (False, True):
            event = {"body": json.dumps({"dataset_key": key, "arm": "E", "fatigue": fatigue})}
            r = lam.invoke(
                FunctionName=out["JudgeReplayFunction"], Payload=json.dumps(event).encode()
            )
            resp = json.loads(r["Payload"].read())
            if r.get("FunctionError") or resp.get("statusCode") != 200:
                failures += 1
                print(f"FAIL {key} fatigue={fatigue}: {str(resp)[:200]}")
                continue
            body = json.loads(resp["body"])
            m = body["metrics"]["agent"]
            print(
                f"ok   {key} fatigue={fatigue} cached={body.get('cached', False)} "
                f"decisions={m.get('decisions')} cost={m.get('cost_usd')}",
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
