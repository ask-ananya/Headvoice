"""Local app: camera thread -> vision pipeline -> state machine -> WebSocket UI.

    uvicorn headvoice.app.server:app --port 8765
    then open http://localhost:8765

The server sends a "speak" message only when the state machine emits a speak event.
Environment: HEADVOICE_CAMERA (default: first live camera), HEADVOICE_PROFILE,
HEADVOICE_API_URL (unset = offline).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from headvoice.app.controller import Controller, controller_from_env
from headvoice.io.schemas import FrameFeatures

log = logging.getLogger("headvoice")
ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"


class CameraLoop(threading.Thread):
    def __init__(self, controller: Controller, camera: int | None):
        super().__init__(daemon=True, name="camera")
        self.controller = controller
        self.camera = camera
        self.stop_flag = threading.Event()
        self.switch_to: int | None = None  # set from the API; the loop reopens the camera

    def request_switch(self, index: int) -> None:
        self.switch_to = index

    def run(self) -> None:

        from headvoice.vision.camera import open_live_camera
        from headvoice.vision.pipeline import VisionPipeline

        pipe = VisionPipeline(self.controller.profile)  # shares the profile object (neutral)
        self.controller.neutral_hook = pipe.request_neutral
        cap, index = open_live_camera(self.camera)
        self.controller.camera_index = index
        log.warning("using camera %d", index)
        t0 = time.monotonic()
        while not self.stop_flag.is_set():
            if self.switch_to is not None:
                cap.release()
                cap, index = open_live_camera(self.switch_to)
                self.controller.camera_index, self.switch_to = index, None
                log.warning("switched to camera %d", index)
            ok, frame = cap.read()
            if not ok:
                self.controller.camera_ok = False
                time.sleep(0.5)
                continue
            self.controller.camera_ok = True
            now = time.monotonic() - t0
            ff = pipe.process(frame, now)
            self.controller.fps = pipe.fps
            self.controller.handle_features(ff, now, frame)
            if ff.frame_idx % 2 == 0:
                self.controller.preview_jpeg = render_preview(frame, ff)
        cap.release()


def render_preview(frame, ff: FrameFeatures) -> bytes | None:
    """Small annotated JPEG for the operator's inspector. Served on localhost only."""
    import cv2

    vis = frame.copy()
    for px, py in ff.landmarks or []:
        cv2.circle(vis, (int(px), int(py)), 2, (180, 120, 255), -1, cv2.LINE_AA)
    vis = cv2.resize(vis, (320, int(320 * vis.shape[0] / vis.shape[1])))
    vis = cv2.flip(vis, 1)  # mirror, like a selfie view: the user's left is on the left
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def create_app(
    controller: Controller | None = None, camera: int | None = None, use_camera: bool = False
) -> FastAPI:
    if controller is None:
        controller = controller_from_env(ROOT)
    cam_loop: CameraLoop | None = None

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal cam_loop
        if use_camera:
            cam_loop = CameraLoop(controller, camera)
            cam_loop.start()
        yield
        if cam_loop:
            cam_loop.stop_flag.set()
        controller.stop()

    app = FastAPI(title="HeadVoice", lifespan=lifespan)
    app.state.controller = controller

    @app.get("/")
    async def landing() -> FileResponse:
        return FileResponse(STATIC / "landing.html")

    @app.get("/start")
    async def start_page() -> FileResponse:
        return FileResponse(STATIC / "start.html")

    @app.get("/app")
    async def app_page() -> FileResponse:
        return FileResponse(STATIC / "app.html")

    @app.get("/api/cameras")
    async def cameras() -> dict[str, Any]:
        from headvoice.vision.camera import probe_cameras

        found = await asyncio.to_thread(probe_cameras, controller.camera_index)
        return {"current": controller.camera_index, "cameras": found}

    @app.post("/api/camera")
    async def switch_camera(body: dict[str, Any]) -> dict[str, Any]:
        if cam_loop is None:
            return {"ok": False, "error": "no camera thread"}
        cam_loop.request_switch(int(body["index"]))
        return {"ok": True}

    @app.get("/messages")
    async def messages_page() -> FileResponse:
        return FileResponse(STATIC / "messages.html")

    @app.get("/api/messages")
    async def list_messages() -> dict[str, Any]:
        return controller.messages()

    @app.post("/api/messages")
    async def add_message(body: dict[str, Any]) -> JSONResponse:
        try:
            return JSONResponse(
                {"ok": True, "message": controller.add_message(body.get("text", ""))}
            )
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    @app.delete("/api/messages/{phrase_id}")
    async def remove_message(phrase_id: str) -> dict[str, Any]:
        return {"ok": controller.remove_message(phrase_id)}

    @app.post("/api/condition")
    async def condition(body: dict[str, Any]) -> dict[str, Any]:
        return {"ok": controller.set_condition(str(body.get("condition", "")))}

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return controller.snapshot()

    @app.get("/preview.mjpg")
    async def preview() -> StreamingResponse:
        async def frames():
            while True:
                jpg = controller.preview_jpeg
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(1 / 10)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()

        async def sender() -> None:
            while True:
                for msg in controller.drain_outbox():
                    await sock.send_text(json.dumps(msg))
                await sock.send_text(json.dumps(controller.snapshot()))
                await asyncio.sleep(1 / 15)

        task = asyncio.create_task(sender())
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                kind = msg.get("type")
                f = controller.last_features
                now = f.t if f else 0.0
                if kind == "speech_ended":
                    controller.speech_ended(now)
                elif kind == "start":
                    controller.start()
                elif kind == "stop":
                    controller.stop()
                elif kind == "neutral":
                    controller.set_neutral()
                elif kind == "calibrate":
                    controller.start_calibration(bool(msg.get("upload")), ROOT / "data" / "tmp")
                elif kind == "calibrate_cancel":
                    controller.cancel_calibration()
                elif kind == "task_start":
                    controller.start_task(n=int(msg.get("n", 10)))
                elif kind == "task_skip":
                    controller.skip_target()
                elif kind == "practice":
                    controller.start_practice()
                elif kind == "layout":
                    controller.set_layout(str(msg.get("layout", "")))
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()

    app.mount("/", StaticFiles(directory=STATIC), name="static")  # last: routes win
    return app


_cam = os.environ.get("HEADVOICE_CAMERA")
# HEADVOICE_CAMERA unset: use the first camera that gives a live picture.
app = create_app(camera=int(_cam) if _cam else None, use_camera=True)
