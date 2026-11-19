"""Vercel serverless function: POST /api/run  (Scenario Lab re-simulation)."""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jalnetra.webapi import run_lab  # noqa: E402

MAX_BODY = 200_000


class handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._json({"error": "request too large"}, 413)
            cfg = json.loads(self.rfile.read(n) or b"{}")
            self._json(run_lab(cfg))
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 - surface the error to the UI
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self):
        self._json({"error": "use POST with a Scenario Lab config"}, 405)
