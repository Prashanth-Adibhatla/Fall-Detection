# """
# Fall Detection Server
# ---------------------
# Multi-camera RTSP fall detection backend.
# Edit CAMERAS list below to add/remove cameras.
# """

import asyncio
import base64
import json
import logging
import os
import sys
import time
import threading
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────
# CONFIGURATION — edit this section
# ─────────────────────────────────────────────

CAMERAS = [
    {
        "id": "cam_1",
        "name": "Living Room",
        "url": "rtsp://100.93.196.174:8080/h264.sdp",
    },

]

# How many seconds between fall-detection inference runs per camera
INFERENCE_INTERVAL_SECONDS = 0.2

# How many seconds a camera stays in "FALL" alert state after detection
FALL_ALERT_DURATION_SECONDS = 10.0

# JPEG quality for streaming frames to frontend (1–95)
STREAM_JPEG_QUALITY = 60

# Fall detection confidence threshold (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.6

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fall-server")

# ─────────────────────────────────────────────
# Fall detection model setup
# ─────────────────────────────────────────────

# Add the fall-detection project to path
FALL_DETECT_DIR = Path(__file__).parent / "fall-detection"
sys.path.insert(0, str(FALL_DETECT_DIR))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def build_fall_detector():
    """Build one FallDetector instance (one per camera — not thread-safe to share)."""
    from src.pipeline.fall_detect import FallDetector

    model_dir = FALL_DETECT_DIR / "ai_models"
    config = {
        "model": {
            "tflite": str(model_dir / "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"),
            "edgetpu": str(model_dir / "posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite"),
        },
        "labels": str(model_dir / "pose_labels.txt"),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_name": "mobilenet",
    }
    return FallDetector(**config)


# ─────────────────────────────────────────────
# Camera worker
# ─────────────────────────────────────────────

class CameraState:
    def __init__(self, cam_id: str, name: str, url: str):
        self.id = cam_id
        self.name = name
        self.url = url

        self.connected: bool = False
        self.fall_detected: bool = False
        self.fall_score: float = 0.0
        self.fall_angle: float = 0.0
        self.fall_at: float = 0.0          # monotonic time of last fall

        self.latest_frame_b64: Optional[str] = None   # JPEG → base64
        self.latest_frame_ts: float = 0.0

        self._lock = threading.Lock()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "connected": self.connected,
                "fall_detected": self.fall_detected,
                "fall_score": round(self.fall_score, 3),
                "fall_angle": round(self.fall_angle, 1),
                "frame": self.latest_frame_b64,
                "frame_ts": self.latest_frame_ts,
            }


def camera_worker(state: CameraState):
    """Runs in a dedicated thread per camera. Captures frames and runs inference."""
    log.info("[%s] Worker starting — %s", state.id, state.url)

    detector = build_fall_detector()

    cap = None
    prev_pil: Optional[Image.Image] = None
    last_inference_time = 0.0

    while True:
        # ── Open / reconnect ──────────────────────────────────────
        if cap is None or not cap.isOpened():
            log.info("[%s] Connecting to %s …", state.id, state.url)
            cap = cv2.VideoCapture(state.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                log.warning("[%s] Could not connect. Retrying in 5 s …", state.id)
                with state._lock:
                    state.connected = False
                time.sleep(5)
                cap = None
                continue
            with state._lock:
                state.connected = True
            log.info("[%s] Connected.", state.id)

        # ── Grab frame ───────────────────────────────────────────
        ret, frame_bgr = cap.read()
        if not ret:
            log.warning("[%s] Frame read failed. Reconnecting …", state.id)
            cap.release()
            cap = None
            with state._lock:
                state.connected = False
            time.sleep(2)
            continue

        now = time.monotonic()

        # ── Encode frame for streaming ───────────────────────────
        _, jpeg_buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
        )
        frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

        with state._lock:
            state.latest_frame_b64 = frame_b64
            state.latest_frame_ts = now

        # ── Expire fall alert ────────────────────────────────────
        with state._lock:
            if state.fall_detected:
                if now - state.fall_at > FALL_ALERT_DURATION_SECONDS:
                    state.fall_detected = False
                    state.fall_score = 0.0
                    log.info("[%s] Fall alert expired.", state.id)

        # ── Run inference every INFERENCE_INTERVAL_SECONDS ───────
        if now - last_inference_time < INFERENCE_INTERVAL_SECONDS:
            continue
        last_inference_time = now

        # Convert frame to PIL for fall detector
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        current_pil = Image.fromarray(frame_rgb)

        if prev_pil is None:
            prev_pil = current_pil
            continue

        try:
            for result in detector.process_sample(image=current_pil):
                if result is None:
                    continue
                inf = result.get("inference_result", [])
                if inf:
                    item = inf[0]
                    score = float(item.get("confidence", 0))
                    angle = float(item.get("leaning_angle", 0))
                    log.info(
                        "[%s] FALL detected! score=%.2f angle=%.1f°",
                        state.id, score, angle,
                    )
                    with state._lock:
                        state.fall_detected = True
                        state.fall_score = score
                        state.fall_angle = angle
                        state.fall_at = now
        except Exception as e:
            log.debug("[%s] Inference error: %s", state.id, e)

        prev_pil = current_pil


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="Fall Detection Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Start camera threads ──────────────────────────────────────────────────────

camera_states: Dict[str, CameraState] = {}

@app.on_event("startup")
def startup():
    for cam in CAMERAS:
        state = CameraState(cam["id"], cam["name"], cam["url"])
        camera_states[cam["id"]] = state
        t = threading.Thread(
            target=camera_worker, args=(state,), daemon=True, name=f"cam-{cam['id']}"
        )
        t.start()
        log.info("Started thread for camera %s", cam["id"])


# ── REST: camera list ─────────────────────────────────────────────────────────

@app.get("/api/cameras")
def get_cameras():
    return [
        {"id": c["id"], "name": c["name"], "url": c["url"]}
        for c in CAMERAS
    ]


# ── WebSocket: push state + frames to frontend ────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients = [c for c in self._clients if c != ws]

    async def broadcast(self, data: str):
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            self._clients = [c for c in self._clients if c not in dead]


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    log.info("Frontend connected via WebSocket.")
    try:
        while True:
            await ws.receive_text()   # keep-alive ping from client
    except WebSocketDisconnect:
        log.info("Frontend disconnected.")
    finally:
        await manager.disconnect(ws)


async def broadcast_loop():
    """Push camera state + frames to all connected clients at ~10 fps."""
    while True:
        payload = [state.to_dict() for state in camera_states.values()]
        await manager.broadcast(json.dumps({"cameras": payload}))
        await asyncio.sleep(0.1)   # 10 fps push


@app.on_event("startup")
async def start_broadcast():
    asyncio.create_task(broadcast_loop())


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

# """
# Fall Detection Server
# ---------------------
# Multi-camera fall detection backend.
# Supports RTSP streams, local webcams (int), and video files (path string).
# Edit CAMERAS list below to add/remove cameras.
# """

# import asyncio
# import base64
# import json
# import logging
# import os
# import sys
# import time
# import threading
# from contextlib import asynccontextmanager
# from pathlib import Path
# from typing import Dict, List, Optional

# import cv2
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse

# # ─────────────────────────────────────────────
# # CONFIGURATION — edit this section
# # ─────────────────────────────────────────────

# CAMERAS = [
#     {
#         "id": "cam_1",
#         "name": "Living Room",
#         "url": "/mnt/c/Users/prashanth/Desktop/fall-detection-1.0.9/videos/video_5.mp4",
#     }
    
#     # ── Other source types ────────────────────────────────────────
#     # RTSP camera:   { "id": "cam_5", "name": "Garden",        "url": "rtsp://admin:pass@192.168.1.101:554/stream1" },
#     # Local webcam:  { "id": "cam_6", "name": "Laptop Webcam", "url": 0 },
# ]

# # Seconds between fall-detection inference runs per camera
# INFERENCE_INTERVAL_SECONDS = 0.2

# # Video file playback speed (1.0 = normal, 0.5 = half speed, 0.25 = quarter speed)
# VIDEO_PLAYBACK_SPEED = 1.3

# # Seconds a camera stays in FALL alert state after detection
# FALL_ALERT_DURATION_SECONDS = 5.0

# # JPEG quality for streaming frames to frontend (1–95)
# STREAM_JPEG_QUALITY = 60

# # Minimum pose confidence to trigger fall detection (0.0–1.0)
# CONFIDENCE_THRESHOLD = 0.6

# # ─────────────────────────────────────────────
# # Logging
# # ─────────────────────────────────────────────

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )
# log = logging.getLogger("fall-server")

# # ─────────────────────────────────────────────
# # Fall detection model setup
# # ─────────────────────────────────────────────

# FALL_DETECT_DIR = Path(__file__).parent / "fall-detection"
# sys.path.insert(0, str(FALL_DETECT_DIR))

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# def build_fall_detector():
#     """Build one FallDetector instance per camera (not thread-safe to share)."""
#     from src.pipeline.fall_detect import FallDetector
#     model_dir = FALL_DETECT_DIR / "ai_models"
#     config = {
#         "model": {
#             "tflite": str(model_dir / "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"),
#             "edgetpu": str(model_dir / "posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite"),
#         },
#         "labels": str(model_dir / "pose_labels.txt"),
#         "confidence_threshold": CONFIDENCE_THRESHOLD,
#         "model_name": "mobilenet",
#     }
#     return FallDetector(**config)

# # ─────────────────────────────────────────────
# # Source type helpers
# # ─────────────────────────────────────────────

# def _is_local_webcam(url) -> bool:
#     return isinstance(url, int)

# def _is_video_file(url) -> bool:
#     return isinstance(url, str) and not url.lower().startswith("rtsp")

# def _open_capture(url) -> cv2.VideoCapture:
#     """Open the correct VideoCapture backend for the given source."""
#     if _is_local_webcam(url):
#         cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)    # Windows local webcam
#     elif _is_video_file(url):
#         cap = cv2.VideoCapture(url)                    # video file
#     else:
#         cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)    # RTSP stream
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     return cap

# # ─────────────────────────────────────────────
# # Camera state
# # ─────────────────────────────────────────────

# class CameraState:
#     def __init__(self, cam_id: str, name: str, url):
#         self.id = cam_id
#         self.name = name
#         self.url = url

#         self.connected: bool = False
#         self.fall_detected: bool = False
#         self.fall_score: float = 0.0
#         self.fall_angle: float = 0.0
#         self.fall_at: float = 0.0

#         self.latest_frame_b64: Optional[str] = None
#         self.latest_frame_ts: float = 0.0

#         self._lock = threading.Lock()

#     def to_dict(self) -> dict:
#         with self._lock:
#             return {
#                 "id": self.id,
#                 "name": self.name,
#                 "connected": self.connected,
#                 "fall_detected": self.fall_detected,
#                 "fall_score": round(self.fall_score, 3),
#                 "fall_angle": round(self.fall_angle, 1),
#                 "frame": self.latest_frame_b64,
#                 "frame_ts": self.latest_frame_ts,
#             }

# # ─────────────────────────────────────────────
# # Camera worker thread
# # ─────────────────────────────────────────────

# def camera_worker(state: CameraState):
#     """Runs in a dedicated daemon thread per camera."""
#     log.info("[%s] Worker starting — %s", state.id, state.url)

#     detector = build_fall_detector()
#     cap = None
#     prev_pil: Optional[Image.Image] = None
#     last_inference_time = 0.0

#     while True:

#         # ── Open / reconnect ─────────────────────────────────────
#         if cap is None or not cap.isOpened():
#             log.info("[%s] Connecting to %s …", state.id, state.url)
#             cap = _open_capture(state.url)
#             if not cap.isOpened():
#                 log.warning("[%s] Could not open source. Retrying in 5 s …", state.id)
#                 with state._lock:
#                     state.connected = False
#                 time.sleep(5)
#                 cap = None
#                 continue
#             with state._lock:
#                 state.connected = True
#             log.info("[%s] Connected.", state.id)

#         # ── Grab frame ───────────────────────────────────────────
#         ret, frame_bgr = cap.read()

#         # ── Video playback speed control ─────────────────────────
#         if _is_video_file(state.url):
#             fps = cap.get(cv2.CAP_PROP_FPS) or 30
#             time.sleep((1.0 / fps) / max(VIDEO_PLAYBACK_SPEED, 0.01))

#         if not ret:
#             if _is_video_file(state.url):
#                 log.info("[%s] Video ended — looping.", state.id)
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 prev_pil = None   # reset pose history at loop boundary
#                 continue
#             else:
#                 log.warning("[%s] Frame read failed. Reconnecting …", state.id)
#                 cap.release()
#                 cap = None
#                 with state._lock:
#                     state.connected = False
#                 time.sleep(2)
#                 continue

#         now = time.monotonic()

#         # ── Encode frame for streaming ───────────────────────────
#         _, jpeg_buf = cv2.imencode(
#             ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
#         )
#         frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

#         with state._lock:
#             state.latest_frame_b64 = frame_b64
#             state.latest_frame_ts = now

#         # ── Expire fall alert ────────────────────────────────────
#         with state._lock:
#             if state.fall_detected:
#                 if now - state.fall_at > FALL_ALERT_DURATION_SECONDS:
#                     state.fall_detected = False
#                     state.fall_score = 0.0
#                     log.info("[%s] Fall alert expired.", state.id)

#         # ── Throttle inference ───────────────────────────────────
#         if now - last_inference_time < INFERENCE_INTERVAL_SECONDS:
#             continue
#         last_inference_time = now

#         # ── Run fall detection ───────────────────────────────────
#         frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#         current_pil = Image.fromarray(frame_rgb)

#         if prev_pil is None:
#             prev_pil = current_pil
#             continue

#         try:
#             for result in detector.process_sample(image=current_pil):
#                 if result is None:
#                     continue
#                 inf = result.get("inference_result", [])
#                 if inf:
#                     item = inf[0]
#                     score = float(item.get("confidence", 0))
#                     angle = float(item.get("leaning_angle", 0))
#                     log.info(
#                         "[%s] FALL detected! score=%.2f angle=%.1f°",
#                         state.id, score, angle,
#                     )
#                     with state._lock:
#                         state.fall_detected = True
#                         state.fall_score = score
#                         state.fall_angle = angle
#                         state.fall_at = now
#         except Exception as e:
#             log.debug("[%s] Inference error: %s", state.id, e)

#         prev_pil = current_pil

# # ─────────────────────────────────────────────
# # FastAPI app — lifespan (replaces on_event)
# # ─────────────────────────────────────────────

# camera_states: Dict[str, CameraState] = {}
# manager = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global manager

#     # ── Startup ──────────────────────────────
#     manager = ConnectionManager()

#     for cam in CAMERAS:
#         state = CameraState(cam["id"], cam["name"], cam["url"])
#         camera_states[cam["id"]] = state
#         t = threading.Thread(
#             target=camera_worker, args=(state,), daemon=True, name=f"cam-{cam['id']}"
#         )
#         t.start()
#         log.info("Started thread for camera %s", cam["id"])

#     broadcast_task = asyncio.create_task(broadcast_loop())

#     yield   # server is running

#     # ── Shutdown ─────────────────────────────
#     broadcast_task.cancel()
#     log.info("Server shutting down.")


# app = FastAPI(title="Fall Detection Monitor", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ─────────────────────────────────────────────
# # WebSocket connection manager
# # ─────────────────────────────────────────────

# class ConnectionManager:
#     def __init__(self):
#         self._clients: List[WebSocket] = []
#         self._lock = asyncio.Lock()

#     async def connect(self, ws: WebSocket):
#         await ws.accept()
#         async with self._lock:
#             self._clients.append(ws)

#     async def disconnect(self, ws: WebSocket):
#         async with self._lock:
#             self._clients = [c for c in self._clients if c != ws]

#     async def broadcast(self, data: str):
#         async with self._lock:
#             dead = []
#             for ws in self._clients:
#                 try:
#                     await ws.send_text(data)
#                 except Exception:
#                     dead.append(ws)
#             self._clients = [c for c in self._clients if c not in dead]


# async def broadcast_loop():
#     """Push camera state + frames to all clients at ~10 fps."""
#     while True:
#         payload = [state.to_dict() for state in camera_states.values()]
#         if manager:
#             await manager.broadcast(json.dumps({"cameras": payload}))
#         await asyncio.sleep(0.1)

# # ─────────────────────────────────────────────
# # Routes
# # ─────────────────────────────────────────────

# @app.get("/api/cameras")
# def get_cameras():
#     return [
#         {"id": c["id"], "name": c["name"], "url": str(c["url"])}
#         for c in CAMERAS
#     ]

# @app.websocket("/ws")
# async def websocket_endpoint(ws: WebSocket):
#     await manager.connect(ws)
#     log.info("Frontend connected via WebSocket.")
#     try:
#         while True:
#             await ws.receive_text()
#     except WebSocketDisconnect:
#         log.info("Frontend disconnected.")
#     finally:
#         await manager.disconnect(ws)

# @app.get("/", response_class=HTMLResponse)
# async def serve_frontend():
#     html_path = Path(__file__).parent / "index.html"
#     if html_path.exists():
#         return HTMLResponse(html_path.read_text())
#     return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

# # ─────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)