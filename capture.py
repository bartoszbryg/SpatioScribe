import os
import threading

import cv2


class PiCamera:
    def __init__(self):
        # Imported here so the script still runs on computers without picamera2
        from picamera2 import Picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
        self.cam.start()

    def read(self):
        return True, self.cam.capture_array()

    def release(self):
        self.cam.stop()


# Grabs camera frames in a background thread so the camera never waits for detection.
# read() always returns the newest frame; older ones are skipped, which keeps the output live.
class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap
        self.ok, self.frame, self.new = True, None, False
        self.running = True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok, self.frame, self.new = ok, frame, True
                self.cond.notify()
            if not ok:
                break

    def read(self):
        with self.cond:
            self.cond.wait_for(lambda: self.new)
            self.new = False
            return self.ok, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def open_source(source):
    if source == "picam":
        return LatestFrameReader(PiCamera())
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    # Video files are read frame by frame so none are skipped; live cameras and streams use the background reader
    return cap if os.path.isfile(source) else LatestFrameReader(cap)
