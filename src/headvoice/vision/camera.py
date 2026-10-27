"""Open a webcam. On Windows, DirectShow first: Media Foundation can open a camera but
return no frames (seen on the dev laptop)."""

import logging
import sys

import cv2
import numpy as np

log = logging.getLogger(__name__)


def open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    apis = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    cap = None
    for api in apis:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                break
        cap.release()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def is_live(cap: cv2.VideoCapture, warmup: int = 8, frames: int = 12, min_changes: int = 4) -> bool:
    """True if the picture keeps changing between frames. A frozen virtual camera (iVCam
    with the phone not streaming) can change during its first frames, then repeats one."""
    for _ in range(warmup):
        if not cap.read()[0]:
            return False
    prev, changes = None, 0
    for _ in range(frames):
        ok, f = cap.read()
        if not ok:
            return False
        if prev is not None and (f.astype(np.int16) - prev.astype(np.int16)).any():
            changes += 1
        prev = f
    return changes >= min_changes


def open_live_camera(
    index: int | None, width: int = 640, height: int = 480, max_index: int = 5
) -> tuple[cv2.VideoCapture, int]:
    """Open the given camera, or with index None the first camera giving a live picture."""
    if index is not None:
        return open_camera(index, width, height), index
    for i in range(max_index):
        cap = open_camera(i, width, height)
        if cap.isOpened() and is_live(cap):
            return cap, i
        cap.release()
    # No live camera. A camera can also be unavailable because another program (or another
    # HeadVoice instance) holds it. Open 0 anyway: a frozen or dead feed fails closed.
    log.warning("no camera gave a live picture; is another program using the webcam?")
    return open_camera(0, width, height), 0


def probe_cameras(in_use: int | None = None, max_index: int = 5) -> list[dict]:
    """Status of each camera index: live, frozen, no_frames, or in_use (by HeadVoice).

    The camera HeadVoice is using is reported without opening it again.
    """
    out = []
    for i in range(max_index):
        if i == in_use:
            out.append({"index": i, "status": "in_use"})
            continue
        cap = open_camera(i)
        if not cap.isOpened():
            cap.release()
            continue
        status = "live" if is_live(cap, warmup=5, frames=10, min_changes=3) else "frozen"
        ok, frame = cap.read()
        cap.release()
        if not ok:
            status = "no_frames"
        out.append({"index": i, "status": status})
    return out
