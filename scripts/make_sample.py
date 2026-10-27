"""Make an anonymized replay dataset from a consented session log.

Removes face geometry (landmarks, face box) and session identifiers, keeping only what
replay needs: time, pose angles relative to neutral, eye openness, quality and flags,
plus the events. Output goes to data/sample/ (for judges) or a given path.

    python scripts/make_sample.py data/sessions/session_X.jsonl --name volunteer01_normal
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DROP_FRAME_KEYS = {"landmarks", "bbox"}
KEEP_HEADER = {"format", "condition", "policy", "profile"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--name", required=True, help="anonymous name, e.g. volunteer01_normal")
    ap.add_argument("--out", default=str(ROOT / "data" / "sample"))
    args = ap.parse_args()
    out = Path(args.out) / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.session, encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] == "header":
                rec = {
                    "kind": "header",
                    "session_id": args.name,
                    **{k: v for k, v in rec.items() if k in KEEP_HEADER},
                }
            elif rec["kind"] == "frame":
                rec = {k: v for k, v in rec.items() if k not in DROP_FRAME_KEYS}
                n += 1
            g.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({n} frames, landmarks and face boxes removed)")


if __name__ == "__main__":
    main()
