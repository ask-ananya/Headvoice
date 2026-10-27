"""Phase 0 check: one Bedrock Converse call with a tool definition succeeds.

Model comes from HEADVOICE_BEDROCK_MODEL_ID (an inference profile ID), region from
AWS_REGION (default us-east-1).
"""

import json
import os
import sys
import time

import boto3

MODEL_ID = os.environ.get(
    "HEADVOICE_BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
REGION = os.environ.get("AWS_REGION", "us-east-1")

TOOL = {
    "toolSpec": {
        "name": "no_change",
        "description": "Explicitly keep the current interface settings.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            }
        },
    }
}


def main() -> int:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    t0 = time.perf_counter()
    resp = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": "Signal quality 0.95, 0 undos in 60 s. "
                        "Call no_change with a short reason."
                    }
                ],
            }
        ],
        toolConfig={"tools": [TOOL], "toolChoice": {"any": {}}},
        inferenceConfig={"maxTokens": 200},
    )
    ms = (time.perf_counter() - t0) * 1000
    blocks = resp["output"]["message"]["content"]
    tool_uses = [b["toolUse"] for b in blocks if "toolUse" in b]
    print(f"model={MODEL_ID} region={REGION} latency={ms:.0f} ms stop={resp['stopReason']}")
    print(f"usage={json.dumps(resp['usage'])}")
    for tu in tool_uses:
        print(f"tool_use: {tu['name']} {json.dumps(tu['input'])}")
    ok = any(tu["name"] == "no_change" for tu in tool_uses)
    print("PASS" if ok else "FAIL: no tool call returned")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
