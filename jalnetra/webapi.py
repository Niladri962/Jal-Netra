"""Shared web logic for the local app (app.py) and the Vercel serverless functions (api/*.py)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .dashboard import build_payload
from .lab import build_from_config, lab_config
from .scenario import SCENARIOS
from .simulation import run

MAX_EVENTS = 25


def config_for(key: str) -> dict:
    if key not in SCENARIOS:
        raise ValueError(f"unknown scenario '{key}'")
    return lab_config(SCENARIOS[key]())


def run_lab(cfg: dict, out_dir: Path | None = None) -> dict:
    """Validate a Lab config, run the full pipeline, and return the dashboard payload."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")
    if cfg.get("scenario", "mandal") not in SCENARIOS:
        raise ValueError("unknown scenario")
    events = cfg.get("events", [])
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise ValueError(f"at most {MAX_EVENTS} events")
    sc = build_from_config(cfg)
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="jalnetra-") as tmp:   # /tmp is the only writable dir on Vercel
            res = run(sc, Path(tmp), verbose=False)
    else:
        res = run(sc, out_dir, verbose=False)
    payload = build_payload(res)
    lab = lab_config(sc)
    by_id = {e.get("id"): e for e in events}
    for e in lab["events"]:
        src = by_id.get(e["id"])
        if src and src.get("description"):
            e["description"] = src["description"]
    on = {e["id"] for e in lab["events"]}
    lab["events"] += [dict(e, enabled=False) for e in events if not e.get("enabled", True) and e.get("id") not in on]
    payload["lab"] = lab
    ev = res.evaluation
    print(f"  [{time.strftime('%H:%M:%S')}] run {sc.key} seed={sc.seed} days={sc.days} events={len(sc.events)} -> "
          f"detected {ev['events_detected']}/{ev['events_total']}, precision {ev['precision']}, {res.runtime_s}s",
          flush=True)
    return payload
