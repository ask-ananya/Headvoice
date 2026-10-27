"""Train the eye open/closed model from labeled eye-crop recordings (.npz).

Each .npz holds crop_l, crop_r (48x24 grayscale) and label per frame, as written by the
guided eye recorder. Writes models/eye_state_svm.yml and models/eye_state.json.

    python scripts/train_eye_model.py recordings/*.npz
"""

import argparse
import json

import numpy as np

from headvoice.vision.eye_state import DEFAULT_META, DEFAULT_MODEL, save_compact, train

OPEN = {"open", "open_down", "open_tilt"}
CLOSED = {"closed", "closed_tilt"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    args = ap.parse_args()
    crops, closed = [], []
    for path in args.recordings:
        d = np.load(path)
        for cl, cr, lab in zip(d["crop_l"], d["crop_r"], d["label"], strict=True):
            if lab in OPEN or lab in CLOSED:
                crops += [cl, cr]
                closed += [lab in CLOSED] * 2
    svm, meta = train(crops, closed)
    meta["recordings"] = len(args.recordings)
    save_compact(svm, DEFAULT_MODEL)  # no per-crop training features in the file
    DEFAULT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {DEFAULT_MODEL.name} and {DEFAULT_META.name}: {meta}")


if __name__ == "__main__":
    main()
