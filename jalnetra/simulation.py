"""Orchestrates one end-to-end run: physics -> nodes -> radio -> backend -> AI -> alerts -> outputs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .alerts import AlertManager
from .analytics import AnalyticsEngine
from .backend import Store
from .codec import SIZE, encode
from .evaluation import evaluate
from .health import health_model
from .lora import RadioNetwork
from .physics import PhysicalNetwork
from .scenario import Scenario
from .sensors import SensorNode


@dataclass
class RunResult:
    sc: Scenario
    out_dir: Path
    truth: pd.DataFrame
    nodes: pd.DataFrame
    groups: pd.DataFrame
    alerts: list
    outbox: list
    equity: pd.DataFrame
    health: dict
    evaluation: dict
    radio: list
    power: list
    sample_payload: dict
    engine_stats: dict
    runtime_s: float


def run(sc: Scenario, out_dir: Path, verbose: bool = True) -> RunResult:
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sc.seed)
    net = PhysicalNetwork(sc, np.random.default_rng(sc.seed + 1))
    nodes = {n.id: SensorNode(n, np.random.default_rng(sc.seed + 100 + i), sc.step_min, sc.cloudiness)
             for i, n in enumerate(sc.nodes)}
    radio = RadioNetwork(sc, np.random.default_rng(sc.seed + 2), SIZE)
    store = Store(out_dir / "jalnetra.db", sc)
    am = AlertManager(sc, np.random.default_rng(sc.seed + 3))
    engine = AnalyticsEngine(sc, am)

    truth_rows, in_flight, sample_payload = [], [], None
    n_alerts_seen = 0
    for k in range(sc.steps):
        t = sc.start + k * sc.step
        active = [ev for ev in sc.events if ev.active(t)]
        truth = net.step(k, t, active)
        for nid, node in nodes.items():
            tr = truth[nid]
            truth_rows.append({"ts": t, "node": nid, **{kk: round(v, 4) for kk, v in tr.items()}})
            raw = node.sample(t, tr, active)
            if raw is None:
                continue
            payload = encode(raw)
            if sample_payload is None and k == 40:
                sample_payload = {"node": nid, "hex": payload.hex(" ").upper(), "fields": raw}
            up = radio.uplink(nid, payload, t)
            if up:
                in_flight.append(up)
        ready = [u for u in in_flight if u.arrival <= t]
        in_flight = [u for u in in_flight if u.arrival > t]
        engine.ingest(store.ingest(ready), t)
        engine.tick(t)
        if verbose and len(am.alerts) > n_alerts_seen:
            for a in am.alerts[n_alerts_seen:]:
                print(f"  {a.time:%d-%b %H:%M}  {a.severity.upper():<11} {a.type:<28} {a.location_name}")
            n_alerts_seen = len(am.alerts)
        if verbose and k % (96 * 2) == 0:
            print(f"  ... simulated {t:%d-%b %H:%M}")

    store.save_alerts(am.alerts, am.outbox)
    store.close()

    ev = evaluate(sc, am.alerts)
    health = health_model(sc, am.alerts, np.random.default_rng(sc.seed + 4))
    res = RunResult(
        sc=sc, out_dir=out_dir, truth=pd.DataFrame(truth_rows), nodes=pd.DataFrame(engine.node_records),
        groups=pd.DataFrame(engine.group_records), alerts=am.alerts, outbox=am.outbox,
        equity=pd.DataFrame(engine.equity_rows), health=health, evaluation=ev, radio=radio.summary(),
        power=[{"node": nid, "min_soc_pct": round(100 * n.min_soc, 1), "final_soc_pct": round(100 * n.soc, 1),
                "avg_current_ma": round(n.avg_current_ma, 2), "autonomy_days_no_sun": round(n.autonomy_days, 1),
                "harvest_mah_day": round(n.energy_in_mah / sc.days, 0),
                "use_mah_day": round(n.energy_out_mah / sc.days, 0)} for nid, n in nodes.items()],
        sample_payload=sample_payload or {},
        engine_stats={"linked_suppressed": engine.linked_suppressed, "cooldown_suppressed": am.suppressed,
                      "baselines": engine.baselines(), "ml_enabled": engine.model is not None},
        runtime_s=round(time.time() - t_start, 1),
    )
    _save_tables(res)
    return res


def _save_tables(res: RunResult) -> None:
    o = res.out_dir
    res.truth.to_csv(o / "ground_truth_physical.csv", index=False)
    res.nodes.to_csv(o / "node_readings_processed.csv", index=False)
    res.groups.to_csv(o / "water_balance.csv", index=False)
    res.equity.to_csv(o / "equity_lpcd.csv", index=False)
    pd.DataFrame([a.to_dict() for a in res.alerts]).to_csv(o / "alerts.csv", index=False)
    with open(o / "alerts_outbox.txt", "w", encoding="utf-8") as f:
        for m in res.outbox:
            f.write(f"--- {m['time']} | {m['channel']} -> {m['recipient']} | {m['status']}"
                    f"{' | ' + str(m['segments']) + ' SMS segment(s)' if m['channel'] == 'SMS' else ''}\n")
            f.write(m["text"] + "\n\n")
    with open(o / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"evaluation": res.evaluation, "health": res.health["events"], "radio": res.radio,
                   "power": res.power, "engine": res.engine_stats}, f, indent=2, default=str)
