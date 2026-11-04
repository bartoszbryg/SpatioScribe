import queue
import threading
import time
import uuid

import requests


# Sends frame records to a remote server in batches from a background thread,
# so a slow or lost connection never slows down detection
class Uploader:
    def __init__(self, url, device, interval=1.0, max_pending=10000):
        self.url = url
        self.device = device
        self.session_id = uuid.uuid4().hex
        self.interval = interval
        self.max_pending = max_pending
        self.queue = queue.Queue()
        self.pending = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, entry):
        self.queue.put({**entry, "time": time.time()})

    def _send(self):
        while not self.queue.empty():
            self.pending.append(self.queue.get())
        if not self.pending:
            return
        # Keep unsent records for a retry, but drop the oldest if the connection is down for a long time
        self.pending = self.pending[-self.max_pending:]
        payload = {"session_id": self.session_id, "device": self.device, "records": self.pending}
        try:
            requests.post(self.url, json=payload, timeout=5).raise_for_status()
            self.pending = []
        except requests.RequestException as e:
            print(f"Upload failed, will retry ({len(self.pending)} records waiting): {e}")

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._send()

    def close(self):
        self.stop_event.set()
        self.thread.join()
        self._send()
        if self.pending:
            print(f"{len(self.pending)} records were not uploaded (they are still in the local JSON)")

