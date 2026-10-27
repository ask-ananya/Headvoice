"""List camera indices and whether each gives a live picture.

Windows can reorder cameras, and phone-as-webcam drivers can repeat one frozen image
when the phone is not streaming. Run this to pick HEADVOICE_CAMERA / --camera.

    python scripts/list_cameras.py
"""

from headvoice.vision.camera import probe_cameras

LABEL = {
    "live": "LIVE",
    "frozen": "FROZEN (same image repeated)",
    "no_frames": "opens but gives no frames",
}

if __name__ == "__main__":
    for c in probe_cameras():
        print(f"camera {c['index']}: {LABEL.get(c['status'], c['status'])}")
