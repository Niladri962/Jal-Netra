"""Vercel serverless function: GET /api/config?scenario=mandal|pilot  (default Lab config)."""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jalnetra.webapi import config_for  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        key = parse_qs(urlparse(self.path).query).get("scenario", ["mandal"])[0]
        try:
            body, code = json.dumps(config_for(key), default=str).encode("utf-8"), 200
        except ValueError as exc:
            body, code = json.dumps({"error": str(exc)}).encode("utf-8"), 400
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
