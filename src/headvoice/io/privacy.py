"""Outbound payload check: only numbers and short labels derived from FrameFeatures leave
the device. The consented calibration clip uses a separate presigned upload, not this path."""

from __future__ import annotations

from typing import Any

FORBIDDEN_KEYS = {"landmarks", "image", "frame", "frames", "video", "pixels", "jpeg", "png", "bbox"}
MAX_LIST = 64
MAX_STR = 300


class PrivacyViolation(ValueError):
    pass


def check_outbound(payload: Any, path: str = "$") -> None:
    """Raise PrivacyViolation if the payload could carry image or face geometry data."""
    if isinstance(payload, bytes | bytearray | memoryview):
        raise PrivacyViolation(f"{path}: binary data")
    if hasattr(payload, "__array__") and not isinstance(payload, int | float | bool):
        raise PrivacyViolation(f"{path}: array data")
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise PrivacyViolation(f"{path}.{k}: forbidden key")
            check_outbound(v, f"{path}.{k}")
    elif isinstance(payload, list | tuple):
        if len(payload) > MAX_LIST:
            raise PrivacyViolation(f"{path}: list of {len(payload)} items")
        for i, v in enumerate(payload):
            check_outbound(v, f"{path}[{i}]")
    elif isinstance(payload, str):
        if len(payload) > MAX_STR:
            raise PrivacyViolation(f"{path}: string of {len(payload)} chars")
    elif payload is not None and not isinstance(payload, int | float | bool):
        raise PrivacyViolation(f"{path}: unsupported type {type(payload).__name__}")
