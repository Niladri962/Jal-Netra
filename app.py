"""JalNetra interactive app - dashboard + Scenario Lab (live re-simulation).

    python app.py              # opens http://localhost:8050
    python app.py --port 9000 --no-browser

Uses only the Python standard library web server; no extra installs.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jalnetra.dashboard import TEMPLATE
from jalnetra.webapi import config_for, run_lab

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
STATE: dict = {}


def simulate(cfg: dict) -> dict:
    with LOCK:
        return run_lab(cfg, ROOT / "output" / "lab" / cfg.get("scenario", "mandal"))


def page(payload: dict) -> bytes:
    blob = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__JALNETRA_DATA__*/null", blob).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, *args) -> None:   # keep the console readable
        pass

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, page(STATE["payload"]), "text/html; charset=utf-8")
        elif u.path == "/api/ping":
            self._json({"ok": True})
        elif u.path == "/api/config":
            key = parse_qs(u.query).get("scenario", ["mandal"])[0]
            try:
                self._json(config_for(key))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            cfg = json.loads(self.rfile.read(n) or b"{}")
            payload = simulate(cfg)
            STATE["payload"] = payload
            self._json(payload)
        except Exception as exc:  # report to the UI instead of crashing the server
            import traceback
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("JalNetra - running the default mandal scenario ...")
    STATE["payload"] = simulate(config_for("mandal"))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  JalNetra interface: {url}\n  Open the 'Scenario Lab' to inject your own events. Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
