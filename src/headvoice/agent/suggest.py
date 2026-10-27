"""Message agent: ranked phrase suggestions with an abstain path. It cannot speak.

Two prompt variants for evaluation, same model and inputs:
  careless       context-heavy, no abstention instruction
  evidence_first signal first, explicit abstention
Output is forced through a tool schema and validated with pydantic; candidates must be
phrasebook IDs. Anything else is rejected (the device then uses the offline list).
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field, ValidationError, field_validator

PROMPTS = {
    "careless": (
        "You help a person who cannot speak choose what to say. Use everything you know "
        "about the situation, the time of day and what people usually need, and propose the "
        "2 or 3 messages they most likely want, ranked."
    ),
    "evidence_first": (
        "You rank candidate messages for a person who cannot speak. Rules: 1. Use only the "
        "evidence given: recent messages, time of day and the gesture context. 2. Signal "
        "first: if gesture_confidence or signal_quality is low, or the evidence does not "
        "point to a clear need, set abstain to true and return no candidates. 3. Never "
        "invent context. 4. Return 2 or 3 candidate IDs from the phrasebook only, with "
        "honest confidences that sum to at most 1."
    ),
}

TOOL = {
    "name": "propose_candidates",
    "description": "Return ranked phrasebook IDs, or abstain.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "abstain": {"type": "boolean"},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["message_id", "confidence"],
                    },
                },
                "reason": {"type": "string", "maxLength": 300, "description": "One sentence."},
            },
            "required": ["abstain", "candidates", "reason"],
        }
    },
}


class Candidate(BaseModel):
    message_id: str
    confidence: float = Field(ge=0, le=1)


class Suggestion(BaseModel):
    abstain: bool
    candidates: list[Candidate] = Field(max_length=3)
    reason: str = Field(max_length=500)  # the schema asks the model for at most 300

    @field_validator("candidates")
    @classmethod
    def _abstain_consistent(cls, v, info):
        if info.data.get("abstain") and v:
            raise ValueError("abstain with candidates")
        return v


def validate(raw: dict, phrase_ids: set[str]) -> Suggestion:
    s = Suggestion.model_validate(raw)
    unknown = [c.message_id for c in s.candidates if c.message_id not in phrase_ids]
    if unknown:
        raise ValueError(f"unknown message ids: {unknown}")
    if not s.abstain and not s.candidates:
        raise ValueError("no candidates and no abstain")
    return s


def run_suggest(
    request: dict, converse, model_id: str, variant: str, guardrail: dict | None = None
) -> dict:
    """request: {context, phrasebook: [{id, text, group}]}. Returns a result dict with
    either a validated suggestion or an error, plus latency and usage."""
    phrase_ids = {p["id"] for p in request["phrasebook"]}
    kwargs = {
        "modelId": model_id,
        "system": [{"text": PROMPTS[variant]}],
        "messages": [{"role": "user", "content": [{"text": json.dumps(request)}]}],
        "toolConfig": {
            "tools": [{"toolSpec": TOOL}],
            "toolChoice": {"tool": {"name": TOOL["name"]}},
        },
        "inferenceConfig": {"maxTokens": 400, "temperature": 0},
    }
    if guardrail:
        kwargs["guardrailConfig"] = guardrail
    t0 = time.perf_counter()
    resp = converse(**kwargs)
    latency = round((time.perf_counter() - t0) * 1000)
    out = {
        "latency_ms": latency,
        "usage": resp.get("usage", {}),
        "variant": variant,
        "model_id": model_id,
    }
    uses = [b["toolUse"] for b in resp["output"]["message"]["content"] if "toolUse" in b]
    if not uses:
        return {**out, "valid": False, "error": "no_tool_output"}
    try:
        s = validate(uses[0]["input"], phrase_ids)
    except (ValidationError, ValueError) as e:
        return {**out, "valid": False, "error": str(e)[:200], "raw": uses[0]["input"]}
    return {**out, "valid": True, "suggestion": s.model_dump()}
