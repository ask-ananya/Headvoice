"""AWS Lambda handlers. One container image serves them all (cloud/Dockerfile); each
function sets its own handler. They import the same headvoice package as the device, so
cloud calibration and replay run exactly the device's OpenCV 5 code.

Environment: BUCKET, TABLE, TOKEN_SECRET_ARN, MODEL_ID, GUARDRAIL_ID, GUARDRAIL_VERSION,
PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK, AWS_REGION.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import time
import uuid
from decimal import Decimal
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
PRICES = (
    float(os.environ.get("PRICE_IN_PER_MTOK", "1.0")),
    float(os.environ.get("PRICE_OUT_PER_MTOK", "5.0")),
)
ROOT = Path(__file__).resolve().parents[3]

_clients: dict = {}
_token: str | None = None


def _c(name: str):
    if name not in _clients:
        _clients[name] = boto3.client(name, region_name=REGION)
    return _clients[name]


def _guardrail() -> dict | None:
    gid = os.environ.get("GUARDRAIL_ID")
    return (
        {
            "guardrailIdentifier": gid,
            "guardrailVersion": os.environ.get("GUARDRAIL_VERSION", "DRAFT"),
        }
        if gid
        else None
    )


def _resp(code: int, body, content_type: str = "application/json") -> dict:
    text = body if isinstance(body, str) else json.dumps(body, default=str)
    return {"statusCode": code, "headers": {"Content-Type": content_type}, "body": text}


def _body(event: dict) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    return json.loads(raw)


def _emf(metrics: dict, **dims) -> None:
    """CloudWatch Embedded Metric Format: one log line becomes custom metrics."""
    print(
        json.dumps(
            {
                "_aws": {
                    "Timestamp": int(time.time() * 1000),
                    "CloudWatchMetrics": [
                        {
                            "Namespace": "HeadVoice",
                            "Dimensions": [list(dims)] if dims else [[]],
                            "Metrics": [{"Name": k} for k in metrics],
                        }
                    ],
                },
                **dims,
                **metrics,
            }
        )
    )


def _audit_put(session_id: str, kind: str, record: dict) -> None:
    table = os.environ.get("TABLE")
    if not table:
        return
    item = json.loads(json.dumps(record, default=str), parse_float=Decimal)
    item.update(
        {
            "session_id": session_id or "unknown",
            "sk": f"{time.time():.3f}#{kind}#{uuid.uuid4().hex[:8]}",
            "kind": kind,
        }
    )
    boto3.resource("dynamodb", region_name=REGION).Table(table).put_item(Item=item)


# Auth -----------------------------------------------------------------------------------
def authorizer(event, _ctx):
    global _token
    if _token is None:
        _token = _c("secretsmanager").get_secret_value(SecretId=os.environ["TOKEN_SECRET_ARN"])[
            "SecretString"
        ]
    header = (event.get("headers") or {}).get("authorization", "")
    given = header.removeprefix("Bearer ").strip()
    return {"isAuthorized": bool(given) and hmac.compare_digest(given, _token)}


# Agents ---------------------------------------------------------------------------------
def supervise(event, _ctx):
    from headvoice.agent.supervisor import bedrock_converse, run_supervisor
    from headvoice.io.privacy import PrivacyViolation, check_outbound

    req = _body(event)
    try:
        check_outbound(req)
    except PrivacyViolation as e:
        _emf({"PrivacyRejections": 1})
        return _resp(400, {"error": f"privacy: {e}"})
    out = run_supervisor(req, bedrock_converse(REGION), MODEL_ID, guardrail=_guardrail())
    u = out["usage"]
    cost = u["inputTokens"] / 1e6 * PRICES[0] + u["outputTokens"] / 1e6 * PRICES[1]
    _emf(
        {
            "AgentLatencyMs": out["latency_ms"],
            "AgentToolCalls": len(out["tool_calls"]),
            "AgentNoChange": int(out["actions"][0]["tool"] == "no_change"),
            "EstimatedCostUSD": cost,
        },
        Agent="supervisor",
    )
    _audit_put(
        req.get("session_id"), "supervise", {"request": req, "response": out, "cost_usd": cost}
    )
    return _resp(200, out)


def suggest(event, _ctx):
    from headvoice.agent.suggest import run_suggest
    from headvoice.agent.supervisor import bedrock_converse

    req = _body(event)
    variant = req.pop("variant", "evidence_first")
    out = run_suggest(req, bedrock_converse(REGION), MODEL_ID, variant, guardrail=_guardrail())
    _emf(
        {
            "SuggestLatencyMs": out["latency_ms"],
            "SuggestInvalid": int(not out["valid"]),
            "SuggestAbstain": int(out.get("suggestion", {}).get("abstain", False)),
        },
        Agent="suggest",
    )
    return _resp(200, out)


# Calibration ----------------------------------------------------------------------------
def calibrate(event, _ctx):
    req = _body(event)
    bucket = os.environ["BUCKET"]
    if req.get("action") == "upload_url":
        key = f"calibration-clips/{uuid.uuid4().hex}.avi"
        url = _c("s3").generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": key, "ContentType": "video/x-msvideo"},
            ExpiresIn=300,
        )
        return _resp(200, {"upload_url": url, "key": key})
    if req.get("action") != "analyze" or not str(req.get("key", "")).startswith(
        "calibration-clips/"
    ):
        return _resp(400, {"error": "bad request"})
    from headvoice.calibration.analyze import analyze
    from headvoice.calibration.record import features_from_clip
    from headvoice.io.schemas import Profile

    local = Path("/tmp") / Path(req["key"]).name
    t0 = time.perf_counter()
    try:
        _c("s3").download_file(bucket, req["key"], str(local))
    finally:
        _c("s3").delete_object(Bucket=bucket, Key=req["key"])  # delete as soon as it is read
    try:
        base = Profile.model_validate(req.get("base_profile") or {})
        frames = features_from_clip(local, req["timestamps"], base)
    finally:
        local.unlink(missing_ok=True)
    res = analyze(frames, base)
    ms = round((time.perf_counter() - t0) * 1000)
    _emf({"CalibrationMs": ms, "CalibrationFrames": len(frames)}, Job="calibrate")
    out = {
        "ok": res.ok,
        "profile": res.profile.model_dump(),
        "report": res.report,
        "warnings": res.warnings,
        "clip_deleted": True,
        "processing_ms": ms,
    }
    _audit_put(
        req.get("session_id", "calibration"),
        "calibrate",
        {
            "ok": res.ok,
            "report": res.report,
            "warnings": res.warnings,
            "layout": res.profile.layout,
        },
    )
    return _resp(200, out)


# Replay ---------------------------------------------------------------------------------
def _run_replay(dataset_key: str, arm: str, fatigue: bool) -> dict:
    from headvoice.agent.supervisor import bedrock_converse, run_supervisor
    from headvoice.io.logger import read_session
    from headvoice.io.schemas import Phrasebook, Profile
    from headvoice.replay import fatigue_sim
    from headvoice.replay.arms import ARMS, arm_setup
    from headvoice.replay.engine import replay, task_targets
    from headvoice.replay.metrics import all_metrics

    if arm not in ARMS:
        raise ValueError("unknown arm")
    local = Path("/tmp") / Path(dataset_key).name
    _c("s3").download_file(os.environ["BUCKET"], dataset_key, str(local))
    header, frames, events = read_session(local)
    local.unlink(missing_ok=True)
    recorded = Profile.model_validate(header.get("profile", {}))
    calibrated = recorded if recorded.calibrated_range_deg is not None else None
    generic = Profile.load(ROOT / "config" / "default_profile.json")
    profile, policy = arm_setup(arm, generic, calibrated or recorded)
    if fatigue:
        frames = fatigue_sim.apply(frames)
    transport = None
    if policy == "agent":
        converse = bedrock_converse(REGION)
        transport = lambda r: run_supervisor(r, converse, MODEL_ID, guardrail=_guardrail())  # noqa: E731
    t0 = time.perf_counter()
    res = replay(
        frames,
        profile,
        Phrasebook.load(ROOT / "config" / "phrasebook_en.json"),
        policy=policy,
        transport=transport,
        targets=task_targets(events),
        session_id=header.get("session_id", "replay"),
    )
    m = all_metrics(res.events, frames, res.traces, profile.quality_min, PRICES)
    timeline = [
        {k: e[k] for k in ("t", "type", "item", "direction", "to", "reason") if k in e}
        for e in res.events
        if e["type"]
        in (
            "highlight",
            "speak",
            "undo",
            "none_of_these",
            "target",
            "task_hit",
            "task_miss",
            "signal_lost",
            "layout_changed",
            "agent_decision",
            "highlight_timeout",
        )
    ]
    traces = [
        {
            k: t[k]
            for k in (
                "t",
                "trigger",
                "policy",
                "input_summary",
                "tool_calls",
                "actions",
                "applied",
                "reason",
                "outcome_after",
                "latency_ms",
            )
        }
        for t in res.traces
    ]
    return {
        "arm": arm,
        "arm_name": ARMS[arm],
        "fatigue_simulated": fatigue,
        "frames": len(frames),
        "duration_s": round(frames[-1].t - frames[0].t, 1) if frames else 0,
        "replay_ms": round((time.perf_counter() - t0) * 1000),
        "metrics": m,
        "timeline": timeline[:400],
        "traces": traces,
    }


def _cached_replay(dataset_key: str, arm: str, fatigue: bool) -> dict:
    """Agent replays (arm E) make several sequential Bedrock calls and can exceed API
    Gateway's 30 s limit on long sessions, so their results are stored in S3, keyed by the
    dataset's ETag. scripts/warm_replay_cache.py precomputes them by invoking this Lambda
    directly (5 minute timeout)."""
    import hashlib

    if arm != "E":
        return _run_replay(dataset_key, arm, fatigue)
    bucket = os.environ["BUCKET"]
    etag = _c("s3").head_object(Bucket=bucket, Key=dataset_key)["ETag"].strip('"')
    key = (
        "cache/"
        + hashlib.sha256(f"{dataset_key}|{etag}|{arm}|{fatigue}".encode()).hexdigest()[:32]
        + ".json"
    )
    try:
        body = _c("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return {**json.loads(body), "cached": True}
    except _c("s3").exceptions.NoSuchKey:
        pass
    out = _run_replay(dataset_key, arm, fatigue)
    if out["metrics"]["agent"].get("request_failures"):
        return out  # a Bedrock outage must not be cached as the agent's behaviour
    _c("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(out, default=str).encode(),
        ContentType="application/json",
    )
    return out


def replay_handler(event, _ctx):
    req = _body(event)
    key = req.get("dataset_key", "")
    if not key.startswith(("datasets/", "sample/")) or ".." in key:
        return _resp(400, {"error": "bad dataset_key"})
    return _resp(200, _cached_replay(key, req.get("arm", "C"), bool(req.get("fatigue"))))


# Audit ----------------------------------------------------------------------------------
def audit(event, _ctx):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    if method == "POST":
        rec = _body(event)
        _audit_put(rec.get("session_id"), "trace", rec)
        applied = rec.get("applied") or []
        _emf(
            {
                "ToolCallRejections": sum(1 for a in applied if a.get("rejected_reason")),
                "StaleDiscards": sum(
                    1 for a in applied if a.get("rejected_reason") == "stale_epoch"
                ),
            },
            Agent=rec.get("policy", "unknown"),
        )
        return _resp(200, {"ok": True})
    sid = (event.get("pathParameters") or {}).get("session_id", "")
    from boto3.dynamodb.conditions import Key

    table = boto3.resource("dynamodb", region_name=REGION).Table(os.environ["TABLE"])
    items = table.query(KeyConditionExpression=Key("session_id").eq(sid), Limit=200)["Items"]
    return _resp(200, {"session_id": sid, "items": items})


# Judge page (public, read-only, sample datasets only) ----------------------------------
def judge_page(event, _ctx):
    html = (Path(__file__).parent / "judge.html").read_text(encoding="utf-8")
    return _resp(200, html, "text/html; charset=utf-8")


def judge_datasets(event, _ctx):
    objs = (
        _c("s3").list_objects_v2(Bucket=os.environ["BUCKET"], Prefix="sample/").get("Contents", [])
    )
    return _resp(200, {"datasets": [o["Key"] for o in objs if o["Key"].endswith(".jsonl")]})


def judge_replay(event, _ctx):
    req = _body(event)
    key = req.get("dataset_key", "")
    if not key.startswith("sample/") or ".." in key:
        return _resp(400, {"error": "judges can replay sample datasets only"})
    return _resp(200, _cached_replay(key, req.get("arm", "C"), bool(req.get("fatigue"))))
