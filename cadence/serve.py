#!/usr/bin/env python3
"""Threaded static server with HTTP Range support — needed for the demo videos.

A single-threaded server deadlocks this page: the browser holds the <video>
connection open while streaming, so every later request (pose JSON, the next
clip, audio) queues behind it and the video never finishes loading. Threading
fixes that; Range support lets the browser seek and lets Safari play at all.
"""
import os, re, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8742


class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")   # always serve fresh clips
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
        if not m:
            return super().send_head()
        size = os.path.getsize(path)
        start, end = m.group(1), m.group(2)
        if start == "":                                  # suffix range: last N bytes
            start = max(0, size - int(end)); end = size - 1
        else:
            start = int(start); end = int(end) if end else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416); return None
        f = open(path, "rb"); f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Limited(f, end - start + 1)

    def log_message(self, *a):
        pass


class _Limited:
    """File wrapper that stops at the end of the requested range."""
    def __init__(self, f, n): self.f, self.n = f, n
    def read(self, sz=-1):
        if self.n <= 0: return b""
        if sz < 0 or sz > self.n: sz = self.n
        b = self.f.read(sz); self.n -= len(b); return b
    def close(self): self.f.close()


os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Cadence demo → http://localhost:{PORT}/capture.html", flush=True)
ThreadingHTTPServer(("", PORT), RangeHandler).serve_forever()
