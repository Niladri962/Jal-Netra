"""Build the static website for deployment (Vercel / GitHub Pages).

    python build_site.py

Writes public/index.html (10-node Mission Bhagiratha scenario) and public/pilot.html
(3-node school pilot). On Vercel, the Scenario Lab additionally calls the Python
serverless functions in api/ to re-run simulations live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jalnetra.dashboard import TEMPLATE
from jalnetra.webapi import config_for, run_lab

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    PUBLIC.mkdir(exist_ok=True)
    template = TEMPLATE.read_text(encoding="utf-8")
    for key, name in (("mandal", "index.html"), ("pilot", "pilot.html")):
        payload = run_lab(config_for(key))
        blob = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")).replace("</", "<\\/")
        out = PUBLIC / name
        out.write_text(template.replace("/*__JALNETRA_DATA__*/null", blob), encoding="utf-8")
        print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
