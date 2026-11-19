"""Build the self-contained 'virtual grid' dashboard (single HTML file, data embedded)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .lab import lab_config
from .simulation import RunResult

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"

# Indicative Indian retail prices (INR, late 2025/2026) for one tap node.
BOM = [
    ("ESP32-WROOM-32 dev board", "Controller, deep sleep 10 uA", 450),
    ("SX1276 LoRa module 865 MHz + 2 dBi antenna", "Radio (IN865)", 750),
    ("Analog pH sensor kit + probe (gravity type)", "pH 0-14", 2800),
    ("Analog ORP sensor kit + probe", "Free-chlorine proxy via ORP", 4200),
    ("Analog turbidity sensor", "0-3000 NTU (field-calibrated w/ formazin)", 900),
    ("Analog TDS sensor (temperature-compensated)", "0-1000+ ppm", 900),
    ("DS18B20 waterproof temperature probe", "TDS / ORP compensation", 150),
    ("YF-DN25 hall-effect flow sensor (1\")", "Pulse flow meter for tap branch", 900),
    ("ADS1115 16-bit ADC", "Cleaner analog reads than ESP32 ADC", 350),
    ("6 V 3 W solar panel + CN3791 MPPT charger", "Power harvest", 750),
    ("2 x 18650 Li-ion 3000 mAh + holder", "45 days autonomy without sun", 600),
    ("IP65 enclosure, flow cell, glands, MOSFET, PCB", "Packaging + sensor power gating", 1100),
]


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def build_payload(res: RunResult) -> dict:
    sc = res.sc
    n = res.nodes.copy()
    n["hour"] = pd.to_datetime(n["ts"]).dt.floor("h")
    hours = pd.date_range(sc.start, sc.end - pd.Timedelta(hours=1), freq="h")
    hidx = {h: i for i, h in enumerate(hours)}
    rank = {"green": 0, "offline": 1, "amber": 2, "red": 3}
    unrank = {v: k for k, v in rank.items()}
    n["srank"] = n["status"].map(rank).fillna(0)

    t = res.truth.copy()
    t["hour"] = pd.to_datetime(t["ts"]).dt.floor("h")

    series = {}
    for spec in sc.nodes:
        d = n[n.node == spec.id].groupby("hour")
        agg = d.agg({c: "mean" for c in ["cl", "turb", "tds", "ph", "flow", "mri", "batt_mv"]}).reindex(hours)
        st = d["srank"].max().reindex(hours).fillna(0)
        tr = t[t.node == spec.id].groupby("hour")["cl"].mean().reindex(hours)
        series[spec.id] = {
            "cl": [_f(v) for v in agg["cl"]], "cl_true": [_f(v) for v in tr],
            "turb": [_f(v, 2) for v in agg["turb"]], "tds": [_f(v, 0) for v in agg["tds"]],
            "ph": [_f(v, 2) for v in agg["ph"]], "flow": [_f(v, 1) for v in agg["flow"]],
            "mri": [_f(v, 1) for v in agg["mri"]], "batt": [_f(v, 0) for v in agg["batt_mv"]],
            "status": [unrank[int(v)] for v in st],
        }

    g = res.groups.copy()
    g["hour"] = pd.to_datetime(g["ts"]).dt.floor("h")
    grank = {"ok": 0, "loss": 1, "theft": 2, "leak": 3}
    g["grank"] = g["state"].map(grank).fillna(-1)
    groups = []
    for grp in sc.groups:
        d = g[g.group == grp.id]
        dd = d.groupby("hour")
        ratio = dd["ratio"].mean().reindex(hours)
        state = dd["grank"].max().reindex(hours).fillna(-1)
        lost = d[d.state.isin(["loss", "theft", "leak"])]["excess_lpm"].sum() * sc.step_min / 1000
        mu = res.engine_stats["baselines"][grp.id]["mu"]
        groups.append({"id": grp.id, "name": grp.name, "parent": grp.parent, "children": grp.children,
                       "ratio": [_f(v * 100, 2) for v in ratio], "mu": round(mu * 100, 2),
                       "state": [int(v) for v in state], "lost_kl": round(float(lost), 1)})

    events = []
    for ev in sc.events:
        events.append({"id": ev.id, "type": ev.type, "target": ev.target, "name": sc.location_name(ev.target),
                       "description": ev.description, "locations": sorted(sc.event_locations(ev) | {ev.target}),
                       "from": (ev.start - sc.start).total_seconds() / 3600,
                       "to": (min(ev.end, sc.end) - sc.start).total_seconds() / 3600,
                       "window": list(ev.daily_window) if ev.daily_window else None,
                       "start": ev.start.strftime("%d %b %H:%M"), "end": ev.end.strftime("%d %b %H:%M")})

    alerts = []
    for a in res.alerts:
        d = a.to_dict()
        d["h"] = (a.time - sc.start).total_seconds() / 3600
        d["rh"] = (a.resolved_time - sc.start).total_seconds() / 3600 if a.resolved_time else None
        alerts.append(d)

    eq = res.equity
    equity = {"rows": eq.to_dict("records") if len(eq) else [], "summary": []}
    if len(eq):
        for nid, d in eq[eq.equity].groupby("node"):
            spec = sc.node_map[nid]
            equity["summary"].append({"node": nid, "name": spec.name, "village": spec.village,
                                      "population": spec.population, "lpcd": round(float(d.lpcd.mean()), 1),
                                      "target": spec.lpcd_target, "pct": round(float(d.pct.mean()), 1)})

    return {
        "meta": {"title": sc.title, "region": sc.region, "key": sc.key, "start": sc.start.isoformat(),
                 "days": sc.days, "step_min": sc.step_min, "train_days": sc.train_days, "hours": len(hours),
                 "runtime_s": res.runtime_s},
        "hours": [h.strftime("%Y-%m-%d %H:%M") for h in hours],
        "nodes": [{"id": s.id, "name": s.name, "village": s.village, "kind": s.kind, "lat": s.lat, "lon": s.lon,
                   "parent": s.parent, "population": s.population, "meter": s.meter} for s in sc.nodes],
        "gateway": {"name": sc.gateway.name, "lat": sc.gateway.lat, "lon": sc.gateway.lon,
                    "backhaul": sc.gateway.backhaul},
        "series": series, "groups": groups, "events": events, "alerts": alerts,
        "outbox": res.outbox[:400], "equity": equity, "health": res.health, "evaluation": res.evaluation,
        "radio": res.radio, "power": res.power, "payload": res.sample_payload,
        "engine": {k: v for k, v in res.engine_stats.items() if k != "baselines"},
        "bom": [{"item": a, "role": b, "inr": c} for a, b, c in BOM],
        "lab": lab_config(sc),
    }


def build_dashboard(res: RunResult) -> Path:
    data = build_payload(res)
    html = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__JALNETRA_DATA__*/null", blob)
    out = res.out_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out
