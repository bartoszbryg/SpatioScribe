"""
sampler.py - runs on the Raspberry Pi: takes pictures from the camera and sends a few of them to a laptop,
which does the detection (see server/frame_server.py). No YOLO or PyTorch is needed on the Pi.

Pictures go straight to the laptop from memory. Only while the laptop cannot be reached are they kept in
--buffer-dir; they are sent once the laptop answers again, and old ones are deleted automatically.

Examples
  python sampler.py --source picam --server http://192.168.1.20:8000/frames --interval 1
  python sampler.py --source demos/output_video_for_input.mp4 --server http://localhost:8000/frames --interval 0.5
"""
import argparse
import glob
import os
import queue
import socket
import threading
import time
import uuid

import cv2
import requests

from capture import open_source


def parse_args():
    p = argparse.ArgumentParser(description="SpatioScribe frame sampler")
    p.add_argument("--source", default="picam", help="'picam' for the Pi camera, a number for a webcam, or a video file")
    p.add_argument("--server", required=True, help="Frame server URL, e.g. http://<laptop-ip>:8000/frames")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between sent pictures")
    p.add_argument("--quality", type=int, default=80, help="JPEG quality (1-100)")
    p.add_argument("--duration", type=float, default=0, help="Stop after this many seconds (0 = no limit)")
    p.add_argument("--device", default=socket.gethostname(), help="Device name stored with the pictures")
    p.add_argument("--buffer-dir", default="sample_buffer", help="Where pictures wait while the laptop is unreachable")
    p.add_argument("--buffer-mb", type=float, default=200, help="Delete the oldest buffered pictures above this size")
    p.add_argument("--buffer-minutes", type=float, default=5, help="Delete buffered pictures older than this")
    return p.parse_args()


# Pictures waiting for the laptop, one folder per run. The capture time is part of the file name,
# so a picture sent late still gets its real time on the laptop.
class DiskBuffer:
    def __init__(self, folder, max_mb, max_minutes):
        self.folder = folder
        self.max_bytes = max_mb * 1024 * 1024
        self.max_age = max_minutes * 60
        self.current = None
        self.expired = 0

    def save(self, session_id, frame_idx, t, captured, jpeg):
        folder = os.path.join(self.folder, session_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{frame_idx:08d}_{t}_{captured:.3f}.jpg")
        with open(path + ".tmp", "wb") as f:
            f.write(jpeg)
        os.replace(path + ".tmp", path)

    def files(self):
        found = []
        for path in glob.glob(os.path.join(self.folder, "*", "*.jpg")):
            try:
                frame_idx, t, captured = os.path.basename(path)[:-4].split("_")
                found.append((float(captured), int(frame_idx), float(t), os.path.basename(os.path.dirname(path)), path))
            except ValueError:
                continue
        return sorted(found)

    def remove(self, path):
        try:
            os.remove(path)
            folder = os.path.dirname(path)
            if os.path.basename(folder) != self.current:
                os.rmdir(folder)
        except OSError:
            pass

    def prune(self):
        files = self.files()
        sizes = {}
        for entry in files:
            try:
                sizes[entry[4]] = os.path.getsize(entry[4])
            except OSError:
                sizes[entry[4]] = 0
        total = sum(sizes.values())
        now = time.time()
        for captured, _, _, _, path in files:
            if now - captured > self.max_age or total > self.max_bytes:
                self.remove(path)
                total -= sizes[path]
                self.expired += 1


# Only the newest pictures wait in memory. If the laptop is slow or gone, older ones move to the disk buffer
class Sender:
    def __init__(self, url, device, buffer, max_waiting=2):
        self.url = url
        self.device = device
        self.buffer = buffer
        self.session_id = uuid.uuid4().hex
        buffer.current = self.session_id
        self.http = requests.Session()
        self.queue = queue.Queue(maxsize=max_waiting)
        self.stop = threading.Event()
        self.link_ok = True
        self.sent = 0
        self.recovered = 0
        self.buffered = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, frame_idx, timestamp, captured, jpeg):
        item = (frame_idx, timestamp, captured, jpeg)
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.spill(self.queue.get_nowait())
                except queue.Empty:
                    pass

    def spill(self, item):
        frame_idx, timestamp, captured, jpeg = item
        self.buffer.save(self.session_id, frame_idx, timestamp, captured, jpeg)
        self.buffered += 1

    def post(self, session_id, frame_idx, timestamp, captured, jpeg):
        params = {"session_id": session_id, "device": self.device, "frame": frame_idx,
                  "t": timestamp, "captured": captured}
        r = self.http.post(self.url, params=params, data=jpeg, headers={"Content-Type": "image/jpeg"}, timeout=(3, 10))
        r.raise_for_status()
        return r.json()

    @staticmethod
    def describe(result):
        if result.get("duplicate"):
            return "already received"
        return f"{result['objects']} objects ({', '.join(result['labels']) or 'nothing'}), laptop took {result['ms']} ms"

    def send_live(self, item):
        frame_idx, timestamp, captured, jpeg = item
        try:
            result = self.post(self.session_id, frame_idx, timestamp, captured, jpeg)
        except requests.RequestException as e:
            if self.link_ok:
                print(f"No answer from the laptop ({type(e).__name__}), keeping pictures in {self.buffer.folder}")
            self.link_ok = False
            self.spill(item)
            return
        if not self.link_ok:
            print("Laptop answers again, sending the buffered pictures")
        self.link_ok = True
        self.sent += 1
        print(f"frame {frame_idx}: {len(jpeg) // 1024} KB, {self.describe(result)}")

    def drain(self, limit=3):
        for captured, frame_idx, t, session_id, path in self.buffer.files()[:limit]:
            try:
                with open(path, "rb") as f:
                    jpeg = f.read()
                result = self.post(session_id, frame_idx, t, captured, jpeg)
            except requests.RequestException:
                self.link_ok = False
                return
            except OSError:
                continue
            self.buffer.remove(path)
            self.recovered += 1
            print(f"frame {frame_idx} (sent late): {self.describe(result)}")

    def _run(self):
        last_prune = 0
        while True:
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                item = None
                if self.stop.is_set():
                    return
            if item:
                self.send_live(item)
            if self.link_ok:
                self.drain()
            if time.time() - last_prune > 5:
                self.buffer.prune()
                last_prune = time.time()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=60)
        self.drain(limit=100000)
        self.buffer.prune()
        self.buffer.current = None
        for folder in glob.glob(os.path.join(self.buffer.folder, "*")):
            try:
                os.rmdir(folder)
            except OSError:
                pass
        left = len(self.buffer.files())
        print(f"Done: {self.sent} sent live, {self.recovered} sent late, {self.buffer.expired} expired")
        if left:
            print(f"{left} pictures are still in {self.buffer.folder} and will be sent next time the laptop answers")


def main():
    args = parse_args()
    buffer = DiskBuffer(args.buffer_dir, args.buffer_mb, args.buffer_minutes)
    buffer.prune()
    cap = open_source(args.source)
    # A video file is played at its normal speed, so it behaves like a live camera
    file_delay = 1 / (cv2.VideoCapture(args.source).get(cv2.CAP_PROP_FPS) or 30) if os.path.isfile(args.source) else 0
    sender = Sender(args.server, args.device, buffer)
    t_start = time.time()
    last_sent = -args.interval
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time() - t_start
            if now - last_sent >= args.interval:
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if ok:
                    sender.add(frame_idx, round(now, 2), time.time(), jpeg.tobytes())
                    last_sent = now
            frame_idx += 1
            if args.duration and now >= args.duration:
                break
            if file_delay:
                time.sleep(file_delay)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sender.close()


if __name__ == "__main__":
    main()
