"""Record a game for the videos: window frames plus everything the pilot reads and does.

    <dir>/frames/<ms>.jpg   screenshots of the game window, ~4 per second
    <dir>/events.jsonl      {"t": seconds, "kind": ..., ...} - log lines, OCR reads, clicks

Frames come from a second X connection on a thread, so recording never slows
the pilot down and the pilot's own screenshots are untouched.
"""
from __future__ import annotations

import json
import os
import threading
import time

import cv2
from Xlib import display

from grab import find_window, grab


class Recorder:
    def __init__(self, out_dir: str, fps: float = 4.0, quality: int = 80):
        self.dir = out_dir
        os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
        self.t0 = time.time()
        events = os.path.join(out_dir, "events.jsonl")
        if os.path.exists(events) and os.path.getsize(events):
            # Resuming into an existing recording: continue its timeline (frames are named by time).
            with open(events) as f:
                last = json.loads(f.readlines()[-1])["t"]
            self.t0 -= last + 1.0
        self.fps, self.quality = fps, quality
        self._events = open(os.path.join(out_dir, "events.jsonl"), "a")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._frames, daemon=True)
        self._thread.start()

    def now(self) -> float:
        return round(time.time() - self.t0, 3)

    def event(self, kind: str, **data):
        with self._lock:
            self._events.write(json.dumps({"t": self.now(), "kind": kind, **data}) + "\n")
            self._events.flush()

    def _frames(self):
        d = display.Display()
        win = find_window(d)
        period = 1.0 / self.fps
        while not self._stop.is_set():
            start = time.time()
            try:
                img = grab(d, win)
                name = f"{int((start - self.t0) * 1000):08d}.jpg"
                cv2.imwrite(os.path.join(self.dir, "frames", name), img,
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            except Exception as e:  # window gone or mid-resize: skip this frame
                self.event("frame_error", error=str(e))
            self._stop.wait(max(0.0, period - (time.time() - start)))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._events.close()
