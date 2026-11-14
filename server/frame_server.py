"""
frame_server.py - runs on the laptop: receives sampled pictures from the Pi (sampler.py), runs YOLO on them,
saves the pictures and stores every detection in the same SQLite database as server.py.

Run (from the repository root):
  pip install -r requirements.txt fastapi uvicorn
  uvicorn server.frame_server:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 to see the latest annotated picture, and start sampler.py on the Pi.
Settings (environment variables): SPATIOSCRIBE_WEIGHTS, SPATIOSCRIBE_IMGSZ, SPATIOSCRIBE_CONF,
SPATIOSCRIBE_LABELS (e.g. "person" to keep only people), SPATIOSCRIBE_FRAMES (folder for pictures).
"""
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

import detector
import drawer
from server.server import get_db

WEIGHTS = os.environ.get("SPATIOSCRIBE_WEIGHTS", detector.DEFAULT_YOLO_WEIGHTS)
IMGSZ = int(os.environ.get("SPATIOSCRIBE_IMGSZ", 640))
CONF = float(os.environ.get("SPATIOSCRIBE_CONF", 0.55))
LABELS = {s.strip() for s in os.environ.get("SPATIOSCRIBE_LABELS", "").split(",") if s.strip()}
FRAMES_DIR = os.environ.get("SPATIOSCRIBE_FRAMES", "received_frames")

PAGE = """<!doctype html><title>SpatioScribe frames</title>
<body style="margin:0;background:#111;color:#ddd;font:16px monospace;text-align:center">
<img id="img" style="max-width:100%;max-height:92vh"><div id="info" style="padding:6px"></div>
<script>
async function tick() {
  try {
    const info = await (await fetch('/latest.json')).json();
    if (info.frame !== undefined) {
      document.getElementById('img').src = '/latest.jpg?' + info.count;
      document.getElementById('info').textContent =
        info.device + ' - frame ' + info.frame + ' - ' + (info.labels.join(', ') || 'nothing detected');
    }
  } catch (e) {}
  setTimeout(tick, 500);
}
tick();
</script>"""

app = FastAPI(title="SpatioScribe frame server")
lock = threading.Lock()
latest = {"jpeg": None, "info": {}}
count = 0


@app.on_event("startup")
def load():
    detector.load_model(WEIGHTS)


def frames_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            session_id TEXT, device TEXT, frame INTEGER,
            video_time REAL, wall_time REAL, image TEXT, objects INTEGER
        )
    """)
    return db


def process(data, session_id, device, frame_idx, t):
    global count
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Body is not a valid JPEG")

    started = time.time()
    with lock:
        detections = detector.detect_objects(frame, IMGSZ, CONF)
    ms = round((time.time() - started) * 1000)
    if LABELS:
        detections = [d for d in detections if d["label"] in LABELS]
    tracks = [{"id": i + 1, "label": d["label"], "bbox": d["bbox"]} for i, d in enumerate(detections)]

    folder = os.path.join(FRAMES_DIR, session_id)
    os.makedirs(folder, exist_ok=True)
    raw_path = os.path.join(folder, f"{frame_idx:06d}.jpg")
    with open(raw_path, "wb") as f:
        f.write(data)
    annotated = drawer.draw_tracks(frame, tracks)
    cv2.imwrite(os.path.join(folder, f"{frame_idx:06d}_det.jpg"), annotated)

    wall = time.time()
    with frames_db() as db:
        db.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (session_id, device, frame_idx, t, wall, raw_path, len(tracks)))
        db.executemany(
            "INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(session_id, device, frame_idx, t, wall, str(o["id"]), o["label"], *o["bbox"]) for o in tracks])

    labels = [o["label"] for o in tracks]
    ok, jpeg = cv2.imencode(".jpg", annotated)
    count += 1
    latest["jpeg"] = jpeg.tobytes()
    latest["info"] = {"device": device, "frame": frame_idx, "labels": labels, "count": count}
    return {"objects": len(tracks), "labels": labels, "ms": ms}


@app.post("/frames")
async def add_frame(request: Request, session_id: str, device: str, frame: int, t: float = 0.0):
    data = await request.body()
    return await run_in_threadpool(process, data, session_id, device, frame, t)


@app.get("/latest.jpg")
def latest_jpg():
    if latest["jpeg"] is None:
        raise HTTPException(404, "No picture received yet")
    return Response(latest["jpeg"], media_type="image/jpeg")


@app.get("/latest.json")
def latest_json():
    return latest["info"]


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
