"""
sampler.py - runs on the Raspberry Pi: takes pictures from the camera and sends a few of them to a laptop,
which does the detection (see server/frame_server.py). No YOLO or PyTorch is needed on the Pi.

Examples
  python sampler.py --source picam --server http://192.168.1.20:8000/frames --interval 1
  python sampler.py --source demos/output_video_for_input.mp4 --server http://localhost:8000/frames --interval 0.5
"""
import argparse
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
    return p.parse_args()


# Only the newest pictures are kept, so a slow Wi-Fi link never makes memory grow or sends stale pictures
class Sender:
    def __init__(self, url, device, max_waiting=2):
        self.url = url
        self.device = device
        self.session_id = uuid.uuid4().hex
        self.queue = queue.Queue(maxsize=max_waiting)
        self.sent = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, frame_idx, timestamp, jpeg):
        item = (frame_idx, timestamp, jpeg)
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _run(self):
        params = {"session_id": self.session_id, "device": self.device}
        while True:
            item = self.queue.get()
            if item is None:
                return
            frame_idx, timestamp, jpeg = item
            try:
                r = requests.post(self.url, params={**params, "frame": frame_idx, "t": timestamp}, data=jpeg,
                                  headers={"Content-Type": "image/jpeg"}, timeout=5)
                r.raise_for_status()
                self.sent += 1
                result = r.json()
                print(f"frame {frame_idx}: {result['objects']} objects ({', '.join(result['labels']) or 'nothing'}), "
                      f"{len(jpeg) // 1024} KB, laptop took {result['ms']} ms")
            except requests.RequestException as e:
                self.dropped += 1
                print(f"frame {frame_idx} not sent ({type(e).__name__}), laptop unreachable?")

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=10)


def main():
    args = parse_args()
    cap = open_source(args.source)
    # A video file is played at its normal speed, so it behaves like a live camera
    file_delay = 1 / (cv2.VideoCapture(args.source).get(cv2.CAP_PROP_FPS) or 30) if os.path.isfile(args.source) else 0
    sender = Sender(args.server, args.device)
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
                    sender.add(frame_idx, round(now, 2), jpeg.tobytes())
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
        print(f"Done: {sender.sent} pictures sent, {sender.dropped} dropped")


if __name__ == "__main__":
    main()
