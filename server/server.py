"""
server.py - receives SpatioScribe detections and stores them in a database.

Run:
  pip install -r server/requirements.txt
  uvicorn server.server:app --host 0.0.0.0 --port 8000

Then start live.py with --server http://<this-computer-ip>:8000/detections
"""
import os
import sqlite3
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel

DB_PATH = os.environ.get("SPATIOSCRIBE_DB", "detections.db")


class DetectedObject(BaseModel):
    id: Union[int, str]
    label: str
    bbox: List[int]


class FrameRecord(BaseModel):
    frame: int
    timestamp: float
    time: float
    objects: List[DetectedObject]


class Batch(BaseModel):
    session_id: str
    device: str
    records: List[FrameRecord]


def get_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            session_id TEXT, device TEXT, frame INTEGER,
            video_time REAL, wall_time REAL,
            track_id TEXT, label TEXT,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
        )
    """)
    return db


app = FastAPI(title="SpatioScribe server")


@app.post("/detections")
def add_detections(batch: Batch):
    rows = [
        (batch.session_id, batch.device, r.frame, r.timestamp, r.time, str(o.id), o.label, *o.bbox)
        for r in batch.records
        for o in r.objects
    ]
    with get_db() as db:
        db.executemany("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return {"frames": len(batch.records), "objects": len(rows)}


@app.get("/detections")
def list_detections(session_id: str = None, limit: int = 100):
    query = "SELECT * FROM detections"
    params = []
    if session_id:
        query += " WHERE session_id = ?"
        params.append(session_id)
    query += " ORDER BY wall_time DESC LIMIT ?"
    params.append(limit)
    with get_db() as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(query, params)]

