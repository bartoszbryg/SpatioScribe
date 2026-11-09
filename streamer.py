import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b"""<!doctype html><title>SpatioScribe live</title>
<body style="margin:0;background:#111;display:flex;justify-content:center">
<img src="/video" style="max-width:100%;height:100vh;object-fit:contain"></body>"""


# Serves the annotated frames as an MJPEG stream that any browser can open.
# JPEG encoding only happens while at least one browser is watching.
class Streamer:
    def __init__(self, port, quality=70):
        self.quality = quality
        self.jpeg = None
        self.viewers = 0
        self.cond = threading.Condition()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/video":
                    outer._stream(self)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, frame):
        if not self.viewers:
            return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def _stream(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.end_headers()
        self.viewers += 1
        try:
            while True:
                with self.cond:
                    self.cond.wait(timeout=5)
                    jpeg = self.jpeg
                if jpeg is None:
                    continue
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                handler.wfile.write(jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.viewers -= 1

    def close(self):
        self.server.shutdown()
