"""Download the model files HeadVoice needs into models/. See docs/models.md."""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = {
    "face_detection_yunet_2026may.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2026may.onnx"
    ),
    "lbfmodel.yaml": (
        "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
    ),
}


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"ok      {name} (already present)")
            continue
        print(f"fetch   {name} <- {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"saved   {name} {dest.stat().st_size} bytes sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
