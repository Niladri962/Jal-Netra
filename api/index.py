"""Single Vercel Python entrypoint for JalNetra.

Routes
    GET  /api/ping                     health check
    GET  /api/config?scenario=mandal   default Scenario Lab config (mandal | pilot)
    POST /api/run                      run the full pipeline for a Lab config
    GET  /, /index.html, /pilot.html   pre-built pages from public/ (fallback; normally served by the CDN)

Declared in pyproject.toml:  [tool.vercel] entrypoint = "api.index:handler"
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jalnetra.webapi import config_for, run_lab  # noqa: E402

MAX_BODY = 200_000
PAGES = {"/": "index.html", "/index.html": "index.html", "/pilot.html": "pilot.html"}


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path.endswith("/api/ping"):
            return self._json({"ok": True})
        if path.endswith("/api/config"):
            key = parse_qs(u.query).get("scenario", ["mandal"])[0]
            try:
                return self._json(config_for(key))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
        if path.endswith("/api/run"):
            return self._json({"error": "use POST with a Scenario Lab config"}, 405)
        page = PAGES.get(path)
        if page:
            fp = os.path.join(ROOT, "public", page)
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8", "public, max-age=300")
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if not urlparse(self.path).path.rstrip("/").endswith("/api/run"):
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._json({"error": "request too large"}, 413)
            self._json(run_lab(json.loads(self.rfile.read(n) or b"{}")))
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 - surface the error to the UI
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, *args) -> None:
        pass
