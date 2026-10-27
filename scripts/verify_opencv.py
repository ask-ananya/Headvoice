"""Phase 0 check: OpenCV 5 is installed and a face detector plus landmark path works.

Usage:
    python scripts/verify_opencv.py            # test image only
    python scripts/verify_opencv.py --camera 0 # also grab one webcam frame
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
YUNET = MODELS / "face_detection_yunet_2026may.onnx"
LBF = MODELS / "lbfmodel.yaml"
TEST_IMAGE = MODELS / "lena.jpg"
TEST_IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


def detect_yunet(img: np.ndarray):
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.8, 0.3, 5000)
    det.setInputSize((w, h))
    t0 = time.perf_counter()
    _, faces = det.detect(img)
    ms = (time.perf_counter() - t0) * 1000
    return faces, ms


def lbf_landmarks(img: np.ndarray, faces: np.ndarray):
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(LBF))
    rects = np.array([faces[0][:4]], dtype=np.int32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    ok, landmarks = facemark.fit(gray, rects)
    ms = (time.perf_counter() - t0) * 1000
    return ok, landmarks, ms


def check_image(label: str, img: np.ndarray) -> None:
    faces, ms = detect_yunet(img)
    n = 0 if faces is None else len(faces)
    if not record(f"{label}: YuNet face detection", n > 0, f"{n} face(s), {ms:.1f} ms"):
        return
    record(f"{label}: YuNet 5 keypoints", faces.shape[1] >= 14, "eyes, nose, mouth corners")
    try:
        ok, lms, ms = lbf_landmarks(img, faces)
        pts = lms[0].reshape(-1, 2) if ok else None
        good = bool(ok) and pts is not None and pts.shape[0] == 68
        record(f"{label}: FacemarkLBF 68 landmarks", good, f"{ms:.1f} ms" if good else "fit failed")
        if good:
            # Eye openness from 6-point eyes (iBUG 68: right 36-41, left 42-47).
            def ear(e):
                return (np.linalg.norm(e[1] - e[5]) + np.linalg.norm(e[2] - e[4])) / (
                    2 * np.linalg.norm(e[0] - e[3])
                )

            record(
                f"{label}: eyelid landmarks usable (EAR)",
                True,
                f"EAR r={ear(pts[36:42]):.3f} l={ear(pts[42:48]):.3f}",
            )
    except (cv2.error, AttributeError) as e:
        record(f"{label}: FacemarkLBF 68 landmarks", False, str(e).splitlines()[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()

    ver = cv2.__version__
    record("OpenCV version is 5.x", ver.startswith("5."), ver)
    record("cv2.FaceDetectorYN available", hasattr(cv2, "FaceDetectorYN"))
    record("cv2.face (contrib) available", hasattr(cv2, "face"))
    record("cv2.solvePnP available", hasattr(cv2, "solvePnP"))
    record("cv2.calcOpticalFlowPyrLK available", hasattr(cv2, "calcOpticalFlowPyrLK"))
    record("cv2.createCLAHE available", hasattr(cv2, "createCLAHE"))
    record("YuNet model present", YUNET.exists(), str(YUNET.name))
    record("LBF model present", LBF.exists(), str(LBF.name))
    # Plan path 1 (an ONNX landmark model through cv2.dnn): opencv_zoo ships no facial
    # landmark model as of Oct 2026, so it is not tried here. See docs/models.md.

    if not TEST_IMAGE.exists():
        urllib.request.urlretrieve(TEST_IMAGE_URL, TEST_IMAGE)
    check_image("test image", cv2.imread(str(TEST_IMAGE)))

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        frame = None
        for _ in range(15):  # let exposure settle
            ok, frame = cap.read()
        cap.release()
        if record("webcam frame captured", frame is not None, f"camera {args.camera}"):
            check_image("webcam", frame)

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    lbf_ok = any(n.endswith("FacemarkLBF 68 landmarks") and ok for n, ok, _ in results)
    path = "LBF 68 landmarks (plan path 2)" if lbf_ok else "YuNet 5 points + eye ROI (path 3)"
    print(f"\nChosen landmark path: {path}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
