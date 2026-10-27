"""Client for the HeadVoice API (API Gateway). Standard library only.

Every call has a timeout; callers treat any exception as "offline" and fall back.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


class CloudClient:
    def __init__(self, api_url: str, token: str, timeout_s: float = 4.0):
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> CloudClient | None:
        url, token = os.environ.get("HEADVOICE_API_URL"), os.environ.get("HEADVOICE_API_TOKEN")
        return cls(url, token) if url and token else None

    def _req(
        self, method: str, path: str, body: dict | None = None, timeout: float | None = None
    ) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.api_url + path,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as r:
            return json.loads(r.read().decode() or "{}")

    def supervise(self, request: dict) -> dict:
        return self._req("POST", "/supervise", request, timeout=12.0)

    def suggest(self, request: dict) -> dict:
        return self._req("POST", "/suggest", request)

    def audit_put(self, record: dict) -> dict:
        return self._req("POST", "/audit", record)

    def audit_get(self, session_id: str) -> dict:
        return self._req("GET", f"/audit/{session_id}")

    def replay(self, body: dict) -> dict:
        return self._req("POST", "/replay", body, timeout=60.0)

    def calibrate(self, clip: Path, timestamps: list[float], base_profile: dict) -> dict:
        """Upload a consented clip with a presigned URL, then run cloud calibration.
        The Lambda deletes the clip after processing."""
        start = self._req("POST", "/calibrate", {"action": "upload_url"})
        put = urllib.request.Request(
            start["upload_url"],
            data=clip.read_bytes(),
            method="PUT",
            headers={"Content-Type": "video/x-msvideo"},
        )
        with urllib.request.urlopen(put, timeout=30):
            pass
        return self._req(
            "POST",
            "/calibrate",
            {
                "action": "analyze",
                "key": start["key"],
                "timestamps": timestamps,
                "base_profile": base_profile,
            },
            timeout=60.0,
        )
