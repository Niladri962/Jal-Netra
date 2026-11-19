"""Score the AI layer against the injected ground truth."""
from __future__ import annotations

from datetime import timedelta

from .scenario import Scenario

ACTIONABLE_SEV = {"critical", "warning", "maintenance"}
CLASS_EXPECTED = {"leak": "LEAK_CONFIRMED", "theft": "THEFT_SUSPECTED"}


def _first_active(sc: Scenario, ev):
    t = ev.start
    while t < ev.end:
        if ev.active(t):
            return t
        t += sc.step
    return ev.start


def evaluate(sc: Scenario, alerts) -> dict:
    used: set[str] = set()
    rows = []
    for ev in sc.events:
        locs = sc.event_locations(ev) | {ev.target}
        t0 = _first_active(sc, ev)
        hits = [a for a in alerts if a.type in ev.expected_alerts and a.location in locs
                and t0 <= a.time <= ev.end + timedelta(hours=3)
                and (ev.daily_window is None or
                     any(ev.active(a.time - timedelta(minutes=m)) for m in range(0, 181, sc.step_min)))]
        used.update(a.id for a in hits)
        first = min((a.time for a in hits), default=None)
        cls = None
        if ev.type in CLASS_EXPECTED and CLASS_EXPECTED[ev.type] in ev.expected_alerts:
            want = CLASS_EXPECTED[ev.type]
            got = [a for a in hits if a.type == want]
            cls = {"expected": want, "correct": bool(got),
                   "time": min(a.time for a in got).strftime("%Y-%m-%d %H:%M") if got else None,
                   "latency_h": round((min(a.time for a in got) - t0).total_seconds() / 3600, 1) if got else None}
        rows.append({
            "event": ev.id, "type": ev.type, "target": ev.target, "target_name": sc.location_name(ev.target),
            "description": ev.description, "start": t0.strftime("%Y-%m-%d %H:%M"),
            "end": ev.end.strftime("%Y-%m-%d %H:%M"), "expected": list(ev.expected_alerts),
            "detected": first is not None,
            "first_alert": first.strftime("%Y-%m-%d %H:%M") if first else None,
            "latency_min": int((first - t0).total_seconds() / 60) if first else None,
            "n_alerts": len(hits), "alert_ids": [a.id for a in hits], "classification": cls,
        })
    actionable = [a for a in alerts if a.severity in ACTIONABLE_SEV]
    false = [a for a in actionable if a.id not in used]
    det = sum(r["detected"] for r in rows)
    classified = [r for r in rows if r["classification"]]
    return {
        "events": rows,
        "events_total": len(rows), "events_detected": det,
        "recall": round(det / len(rows), 3) if rows else None,
        "alerts_total": len(actionable), "false_alerts": len(false),
        "precision": round(1 - len(false) / len(actionable), 3) if actionable else None,
        "false_alert_list": [a.to_dict() for a in false],
        "classification_accuracy": round(sum(r["classification"]["correct"] for r in classified) / len(classified), 3)
        if classified else None,
        "median_latency_min": sorted(r["latency_min"] for r in rows if r["latency_min"] is not None)[
            det // 2] if det else None,
    }
