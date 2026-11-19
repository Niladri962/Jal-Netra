"""Scenario Lab: turn a browser-edited event list into a runnable Scenario.

The dashboard's Lab drawer edits a JSON config (scenario, seed, days, events with
start / duration / magnitude / target). ``build_from_config`` rebuilds the Scenario
from it, re-deriving which alerts each event *should* trigger so the scoring stays fair.
"""
from __future__ import annotations

from datetime import timedelta

from .scenario import SCENARIOS, EventSpec, Scenario

TYPE_INFO = {
    "leak": {"label": "Pipe leak / burst", "targets": "groups",
             "mag": ["lpm", "Leak size (L/min)", 5, 80, 1],
             "hint": "Runs through a night (00-05 h) to be confirmed as a leak by minimum-night-flow."},
    "theft": {"label": "Illegal tapping (daily window)", "targets": "groups", "window": True,
              "mag": ["lpm", "Draw (L/min)", 20, 200, 5],
              "hint": "Needs >= 2 evenings for the recurring-pattern (theft) classification."},
    "contamination": {"label": "Sewage ingress / contamination", "targets": "nodes",
                      "mag": ["turb_add", "Added turbidity (NTU)", 1, 25, 0.5],
                      "hint": "Chlorine is consumed; risk = low chlorine x high turbidity."},
    "chlorination_failure": {"label": "Chlorination failure at WTP", "targets": "source",
                             "mag": ["dose", "Chlorine left in dose (mg/L)", 0.0, 0.8, 0.02],
                             "hint": "Travels downstream with water age; expect one root-cause alert."},
    "groundwater_mixing": {"label": "Saline borewell mixing", "targets": "notsource",
                           "mag": ["fraction", "Borewell share of supply", 0.1, 0.8, 0.05],
                           "hint": "Raises TDS at the target and everything downstream."},
    "sensor_stuck": {"label": "Frozen sensor (pH)", "targets": "nodes",
                     "hint": "Needs >= 3 h to be told apart from a genuinely steady reading."},
    "node_down": {"label": "Node power loss / vandalism", "targets": "nodes",
                  "hint": "Flagged after 1 h of silence while the gateway is healthy."},
    "gateway_outage": {"label": "Gateway 4G backhaul outage", "targets": "gw",
                       "hint": "Gateway buffers frames; analytics back-fill when it returns."},
    "chronic_inequity": {"label": "Tail-end under-supply", "targets": "taps",
                         "mag": ["supply_factor", "Share of design supply", 0.3, 1.0, 0.05],
                         "hint": "Evaluated once a day at 06:00 (alert below 70% of target)."},
}

DEFAULT_PARAMS = {
    "leak": {"lpm": 25}, "theft": {"lpm": 90},
    "contamination": {"turb_add": 12.0, "cl_factor": 0.05, "tds_add": 80, "ph_shift": -0.25, "attack_rate": 0.04},
    "chlorination_failure": {"dose": 0.12, "attack_rate": 0.002},
    "groundwater_mixing": {"fraction": 0.45, "tds": 1600},
    "sensor_stuck": {"sensor": "ph"},
}

EXPECTED = {"contamination": ("CONTAMINATION_RISK",), "chlorination_failure": ("SOURCE_CHLORINATION_FAILURE",),
            "groundwater_mixing": ("HIGH_TDS",), "sensor_stuck": ("SENSOR_FAULT",), "node_down": ("NODE_OFFLINE",),
            "gateway_outage": ("GATEWAY_OFFLINE",), "chronic_inequity": ("EQUITY_SHORTFALL",)}


def _targets(sc: Scenario, kind: str) -> list[dict]:
    if kind == "groups":
        return [{"id": g.id, "name": g.name} for g in sc.groups]
    if kind == "gw":
        return [{"id": "GW", "name": sc.gateway.name}]
    nodes = sc.nodes
    if kind == "source":
        nodes = [n for n in nodes if n.kind == "source"]
    elif kind == "notsource":
        nodes = [n for n in nodes if n.kind != "source"]
    elif kind == "taps":
        nodes = [n for n in nodes if n.kind == "tap" and n.equity]
    return [{"id": n.id, "name": f"{n.id} {n.name}"} for n in nodes]


def lab_config(sc: Scenario) -> dict:
    h = 3600.0
    events = []
    for ev in sc.events:
        info = TYPE_INFO.get(ev.type, {})
        mag = None
        if "mag" in info:
            k = info["mag"][0]
            mag = sc.node_map[ev.target].supply_factor if ev.type == "chronic_inequity" else ev.params.get(k)
        events.append({"id": ev.id, "type": ev.type, "target": ev.target,
                       "start_h": round((ev.start - sc.start).total_seconds() / h, 2),
                       "duration_h": round((ev.end - ev.start).total_seconds() / h, 2),
                       "window": list(ev.daily_window) if ev.daily_window else None,
                       "mag": mag, "enabled": True, "description": ev.description})
    types = {}
    for t, info in TYPE_INFO.items():
        tg = _targets(sc, info["targets"])
        if tg:
            types[t] = {"label": info["label"], "mag": info.get("mag"), "window": info.get("window", False),
                        "hint": info["hint"], "targets": tg}
    return {"scenario": sc.key, "scenarios": list(SCENARIOS), "seed": sc.seed, "days": sc.days,
            "min_days": 2 if sc.key == "pilot" else 5, "max_days": 7 if sc.key == "pilot" else 28,
            "train_days": sc.train_days, "start": sc.start.isoformat(), "types": types, "events": events}


def _expected(sc: Scenario, ev: EventSpec) -> tuple[str, ...]:
    if ev.type in EXPECTED:
        return EXPECTED[ev.type]
    steps, t = [], ev.start
    while t < ev.end:
        if ev.active(t):
            steps.append(t)
        t += sc.step
    if ev.type == "leak":
        night = sum(1 for s in steps if s.hour < 5)
        return ("WATER_LOSS", "LEAK_CONFIRMED") if night >= 8 else ("WATER_LOSS",)
    if ev.type == "theft":
        return ("WATER_LOSS", "THEFT_SUSPECTED") if len({s.date() for s in steps}) >= 2 else ("WATER_LOSS",)
    return ()


def build_from_config(cfg: dict) -> Scenario:
    key = cfg.get("scenario", "mandal")
    factory = SCENARIOS[key]
    default = factory()
    days = int(max(2, min(28, cfg.get("days", default.days))))
    sc = factory(seed=int(cfg.get("seed", default.seed)), days=days)
    defaults_supply = {n.id: n.supply_factor for n in default.nodes}
    # structural under-supply only exists if an enabled inequity event says so
    for n in sc.nodes:
        if n.kind == "tap" and defaults_supply[n.id] < 0.8:
            n.supply_factor = 1.0
    events = []
    for i, e in enumerate(cfg.get("events", [])):
        if not e.get("enabled", True):
            continue
        t, target = e["type"], e["target"]
        if t not in TYPE_INFO or (target not in sc.node_map and target not in sc.group_map and target != "GW"):
            continue
        start = sc.start + timedelta(hours=float(e.get("start_h", 0)))
        end = min(start + timedelta(hours=max(0.25, float(e.get("duration_h", 1)))), sc.end)
        if start >= sc.end:
            continue
        params = dict(DEFAULT_PARAMS.get(t, {}))
        if sc.key == "pilot" and "attack_rate" in params:
            params["attack_rate"] = 0.0
        info = TYPE_INFO[t]
        mag = e.get("mag")
        if "mag" in info and mag is not None:
            if t == "chronic_inequity":
                sc.node_map[target].supply_factor = float(mag)
            else:
                params[info["mag"][0]] = float(mag)
        window = tuple(int(x) for x in e["window"]) if (t == "theft" and e.get("window")) else None
        propagate = t == "groundwater_mixing" or (t == "contamination" and sc.node_map[target].kind != "tap")
        desc = e.get("description") or ""
        if not desc or e.get("id", "").startswith("U") or e.get("edited"):
            mtxt = f" ({info['mag'][1]}: {mag:g})" if ("mag" in info and mag is not None) else ""
            desc = f"{info['label']} - {sc.location_name(target)}{mtxt}"
        ev = EventSpec(e.get("id") or f"U{i + 1}", t, target, start, end, params, window, desc, (), propagate)
        ev.expected_alerts = _expected(sc, ev)
        events.append(ev)
    sc.events = events
    return sc
