"""JalNetra AI layer - streaming analytics engine.

For every 15-minute time slice (processed as soon as data arrives):

1. Data-quality guard    stuck-value and range checks; faulty sensors are excluded
                         from scoring and a maintenance ticket is raised.
2. Microbial Risk Index  0-100 score from chlorine deficit, turbidity excess, and their
                         interaction (low Cl x high turbidity = pathogen ingress), plus pH
                         deviation and an Isolation-Forest anomaly boost.
3. Isolation Forest      multivariate anomaly model trained on the commissioning period.
4. Water balance         for every parent->children segment: loss ratio = (in - sum out)/in,
                         monitored with a one-sided CUSUM. Incidents are classified by
                         minimum-night-flow analysis (leak) vs a recurring time-of-day
                         signature (illegal draw).
5. Spatial reasoning     low chlorine at the WTP -> one SOURCE alert; downstream
                         low-chlorine alerts are linked and suppressed (no alert storm).
6. Equity                daily LPCD per habitation vs target; shortfall alerts.
7. Health of the system  node offline (gateway healthy) vs gateway offline (all silent).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np

from .alerts import AlertManager
from .codec import orp_to_free_chlorine
from .scenario import Scenario

from .iforest import IsolationForest   # pure-NumPy implementation (no scikit-learn needed)

EWMA_ALPHA = 0.4
CUSUM_K, CUSUM_H = 1.0, 10.0
MRI_AMBER, MRI_RED = 30.0, 60.0
CL_MIN = 0.2                 # BIS IS 10500: >= 0.2 mg/L free residual chlorine at consumer end
TDS_ACCEPT, TDS_PERMIT = 500.0, 2000.0
STUCK_N = 12                 # identical raw samples (3 h) => frozen sensor
SENSOR_KEYS = {"ph_x100": "pH", "orp_mv": "ORP (chlorine)", "turb_x10": "turbidity", "tds": "TDS"}


def clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def microbial_risk(cl, turb, ph) -> float:
    cl_def = clip01((0.5 - cl) / 0.5) if cl is not None else 0.0
    turb_ex = clip01((turb - 1.0) / 4.0) if turb is not None else 0.0
    ph_dev = clip01(max(6.5 - ph, ph - 8.5, 0.0)) if ph is not None else 0.0
    return 100 * min(1.0, 0.30 * cl_def + 0.20 * turb_ex + 0.40 * cl_def * turb_ex + 0.10 * ph_dev)


class NodeState:
    def __init__(self):
        self.ewma: dict[str, float] = {}
        self.last_raw: dict[str, int] = {}
        self.same: dict[str, int] = defaultdict(int)
        self.faulty: set[str] = set()
        self.missing = 0
        self.offline = False
        self.cnt = defaultdict(int)          # consecutive-condition counters
        self.last_flow: float | None = None
        self.flow_age = 99
        self.status = "green"


class GroupState:
    def __init__(self):
        self.win: deque = deque(maxlen=4)
        self.train: list[float] = []
        self.mu = 0.0
        self.sigma = 0.02
        self.S = 0.0
        self.onset: datetime | None = None
        self.incident: dict | None = None
        self.history: list[dict] = []
        self.n_incidents = 0


class AnalyticsEngine:
    def __init__(self, sc: Scenario, am: AlertManager):
        self.sc = sc
        self.am = am
        self.ns = {n.id: NodeState() for n in sc.nodes}
        self.gs = {g.id: GroupState() for g in sc.groups}
        self.pending: dict[datetime, dict] = {}
        self.next_ts = sc.start
        self.last_arrival = sc.start
        self.trained = False
        self.ml_rows: list[list[float]] = []
        self.ml_node_rows: dict[str, list[list[float]]] = defaultdict(list)
        self.ml_stats: dict[str, tuple] = {}
        self.flow_hour: dict[tuple[str, int], list[float]] = defaultdict(list)
        self.flow_med: dict[tuple[str, int], float] = {}
        self.model = None
        self.node_records: list[dict] = []
        self.group_records: list[dict] = []
        self.daily_flow: dict[tuple[str, object], list[float]] = defaultdict(list)
        self.equity_rows: list[dict] = []
        self.linked_suppressed = 0
        self.source_id = next(n.id for n in sc.nodes if n.kind == "source")

    # =================================================================== ingest
    def ingest(self, readings: list[dict], now: datetime) -> None:
        if readings:
            self.last_arrival = now
            if self.am.is_open(("GW",)):
                self.am.resolve(("GW",), now)
        for r in readings:
            self.pending.setdefault(r["ts"], {})[r["node_id"]] = r

    def tick(self, now: datetime) -> None:
        silent = (now - self.last_arrival).total_seconds() / 60
        if silent >= 60 and not self.am.is_open(("GW",)):
            self.am.raise_alert(("GW",), "GATEWAY_OFFLINE", now, self.last_arrival, "GW", {"mins": int(silent)})
        if not self.pending:
            return
        latest = max(self.pending)
        while self.next_ts <= latest:
            self.process(self.next_ts, self.pending.pop(self.next_ts, {}), now)
            self.next_ts += self.sc.step

    # ================================================================== process
    def process(self, ts: datetime, readings: dict[str, dict], now: datetime) -> None:
        sc, am = self.sc, self.am
        if not self.trained and ts >= sc.train_end:
            self._train()
        hour = ts.hour + ts.minute / 60
        feats, feat_nodes, vals = [], [], {}

        # ---------------------------------------------------- per-node checks
        for spec in sc.nodes:
            nid, st = spec.id, self.ns[spec.id]
            r = readings.get(nid)
            if r is None:
                st.missing += 1
                st.flow_age += 1
                if st.missing >= 4 and not st.offline:
                    st.offline = True
                    am.raise_alert(("OFF", nid), "NODE_OFFLINE", now, ts, nid, {"mins": st.missing * sc.step_min})
                continue
            if st.offline:
                st.offline = False
                am.resolve(("OFF", nid), now)
            st.missing = 0

            # 1. data quality: frozen sensors
            for key, label in SENSOR_KEYS.items():
                v = r[key]
                st.same[key] = st.same[key] + 1 if st.last_raw.get(key) == v else 0
                st.last_raw[key] = v
                if st.same[key] >= STUCK_N - 1 and key not in st.faulty:
                    st.faulty.add(key)
                    am.raise_alert(("SF", nid, key), "SENSOR_FAULT", now, ts, nid, {"sensor": label, "n": STUCK_N})
                elif st.same[key] == 0 and key in st.faulty:
                    st.faulty.discard(key)
                    am.resolve(("SF", nid, key), now)
            if not (0 < r["ph"] < 14):
                st.faulty.add("ph_x100")

            # 2. engineering values (faulty channels fall back / drop out)
            ph_ok = "ph_x100" not in st.faulty
            ph = r["ph"] if ph_ok else st.ewma.get("ph", spec.base_ph)
            cl = orp_to_free_chlorine(r["orp_mv"], ph) if "orp_mv" not in st.faulty else None
            turb = r["turb"] if "turb_x10" not in st.faulty else None
            tds = float(r["tds"]) if "tds" not in st.faulty else None
            for k, v in (("ph", ph if ph_ok else None), ("cl", cl), ("turb", turb), ("tds", tds)):
                if v is not None:
                    st.ewma[k] = v if k not in st.ewma else EWMA_ALPHA * v + (1 - EWMA_ALPHA) * st.ewma[k]
            st.last_flow, st.flow_age = r["flow"], 0
            self.daily_flow[(nid, ts.date())].append(r["flow"])
            vals[nid] = {"r": r, "ph": ph, "cl": cl, "turb": turb, "tds": tds}

            # 3. ML features (z-scored per node after training)
            hkey = (nid, ts.hour)
            f = [st.ewma.get("cl", 1.0), st.ewma.get("turb", 0.5), st.ewma.get("tds", 300.0),
                 st.ewma.get("ph", 7.4), r["flow"]]
            if not self.trained:
                self.flow_hour[hkey].append(r["flow"])
                self.ml_node_rows[nid].append(f + [ts.hour])
            elif self.model is not None:
                med = self.flow_med.get(hkey, r["flow"])
                f[4] = math.log((r["flow"] + 1) / (med + 1))
                mu, sd = self.ml_stats[nid]
                z = [(f[i] - mu[i]) / sd[i] for i in range(5)]
                if not ph_ok:
                    z[3] = 0.0
                feats.append(z)
                feat_nodes.append(nid)

        ml_scores: dict[str, float] = {}
        if feats and self.model is not None:
            dec = self.model.decision_function(np.array(feats))
            for nid, d in zip(feat_nodes, dec):
                ml_scores[nid] = clip01(-d / 0.12)

        # ---------------------------------------------------- quality alerts
        src_linked = am.recently_open("SOURCE_CHLORINATION_FAILURE", self.source_id, now, timedelta(hours=18))
        for spec in sc.nodes:
            nid, st = spec.id, self.ns[spec.id]
            if nid not in vals:
                self.node_records.append({"ts": ts, "node": nid, "status": "offline" if st.offline else st.status,
                                          "missing": True})
                continue
            v = vals[nid]
            cl_e, tu_e, ph_e, tds_e = (st.ewma.get("cl") if v["cl"] is not None else None,
                                       st.ewma.get("turb") if v["turb"] is not None else None,
                                       st.ewma.get("ph"), st.ewma.get("tds") if v["tds"] is not None else None)
            ml = ml_scores.get(nid, 0.0)
            mri = min(100.0, microbial_risk(cl_e, tu_e, ph_e) + 10 * ml)

            st.cnt["mri"] = st.cnt["mri"] + 1 if mri >= MRI_RED else 0
            st.cnt["mri_ok"] = st.cnt["mri_ok"] + 1 if mri < MRI_AMBER else 0
            if st.cnt["mri"] >= 2:
                am.raise_alert(("CONTAM", nid), "CONTAMINATION_RISK", now, ts, nid,
                               {"mri": mri, "cl": cl_e or 0, "turb": tu_e or 0})
            if st.cnt["mri_ok"] >= 8:
                am.resolve(("CONTAM", nid), now)

            low = cl_e is not None and cl_e < CL_MIN
            st.cnt["lowcl"] = st.cnt["lowcl"] + 1 if low else 0
            st.cnt["cl_ok"] = st.cnt["cl_ok"] + 1 if (cl_e is not None and cl_e >= CL_MIN * 1.5) else 0
            if st.cnt["lowcl"] >= (2 if nid == self.source_id else 4):
                if nid == self.source_id:
                    lag_h = max(s.tau_up_h + s.tau_local_h for s in sc.nodes)
                    am.raise_alert(("SRC",), "SOURCE_CHLORINATION_FAILURE", now, ts, nid,
                                   {"cl": cl_e, "lag": f"~2-{lag_h:.0f} h"})
                elif src_linked or am.is_open(("CONTAM", nid)) or mri >= 45:
                    if st.cnt["lowcl"] == 4:
                        self.linked_suppressed += 1
                elif st.cnt["lowcl"] >= 6:
                    am.raise_alert(("LOWCL", nid), "LOW_CHLORINE", now, ts, nid, {"cl": cl_e})
            if st.cnt["cl_ok"] >= 8:
                am.resolve(("LOWCL", nid), now)
                if nid == self.source_id:
                    am.resolve(("SRC",), now)

            if tds_e is not None:
                st.cnt["tds"] = st.cnt["tds"] + 1 if tds_e > TDS_ACCEPT else 0
                if st.cnt["tds"] >= 8:
                    am.raise_alert(("TDS", nid), "HIGH_TDS", now, ts, nid, {"tds": tds_e},
                                   severity="critical" if tds_e > TDS_PERMIT else None)
                elif tds_e < TDS_ACCEPT * 0.9:
                    am.resolve(("TDS", nid), now)
            if ph_e is not None and "ph_x100" not in st.faulty:
                st.cnt["ph"] = st.cnt["ph"] + 1 if not (6.5 <= ph_e <= 8.5) else 0
                if st.cnt["ph"] >= 4:
                    am.raise_alert(("PH", nid), "PH_OUT_OF_RANGE", now, ts, nid, {"ph": ph_e})

            # node colour for the virtual grid
            if am.is_open(("CONTAM", nid)) or mri >= MRI_RED or (tds_e or 0) > TDS_PERMIT:
                status = "red"
            elif (mri >= MRI_AMBER or low or (tds_e or 0) > TDS_ACCEPT or st.faulty or st.cnt["ph"] >= 4
                  or (f"G-{nid}" in self.gs and self.gs[f"G-{nid}"].incident is not None)):
                status = "amber"
            else:
                status = "green"
            st.status = status
            r = v["r"]
            self.node_records.append({
                "ts": ts, "node": nid, "status": status, "missing": False, "ph": r["ph"],
                "cl": round(v["cl"], 3) if v["cl"] is not None else None, "turb": r["turb"], "tds": r["tds"],
                "flow": r["flow"], "temp": r["temp"], "batt_mv": r["batt_mv"], "mri": round(mri, 1),
                "ml": round(ml, 3), "faulty": ",".join(SENSOR_KEYS[k] for k in sorted(st.faulty)),
                "rssi": r["rssi"], "late_min": int((now - ts).total_seconds() / 60)})

        # ------------------------------------------------------ water balance
        for g in sc.groups:
            self._balance(g, ts, hour, now)

        # ------------------------------------------------------------ equity
        if ts.hour == 6 and ts.minute == 0 and ts.date() > sc.start.date():
            self._equity(ts, now)

    # ============================================================ water balance
    def _balance(self, g, ts: datetime, hour: float, now: datetime) -> None:
        gs, am = self.gs[g.id], self.am
        members = [g.parent] + g.children
        if any(self.ns[m].flow_age > 2 or self.ns[m].last_flow is None for m in members):
            self.group_records.append({"ts": ts, "group": g.id, "ratio": None, "state": "no-data"})
            return
        p = self.ns[g.parent].last_flow
        c = sum(self.ns[m].last_flow for m in g.children)
        nominal_min = 2.0
        if p < nominal_min:
            self.group_records.append({"ts": ts, "group": g.id, "ratio": None, "state": "idle"})
            return
        raw = (p - c) / p
        gs.win.append(raw)
        ratio = float(np.mean(gs.win))          # smoothed, for display and loss estimates
        if not self.trained:
            gs.train.append(raw)
            self.group_records.append({"ts": ts, "group": g.id, "ratio": ratio, "mu": None, "z": 0.0, "S": 0.0,
                                       "excess_lpm": 0.0, "state": "training"})
            return
        z = (raw - gs.mu) / gs.sigma            # CUSUM on independent raw samples (no smoothing bias)
        excess = max(0.0, (ratio - gs.mu) * p)
        prev_S = gs.S
        gs.S = max(0.0, gs.S + z - CUSUM_K)
        if prev_S == 0 and gs.S > 0:
            gs.onset = ts
        night = hour < 5
        inc = gs.incident
        if inc is None and gs.S > CUSUM_H:
            gs.n_incidents += 1
            onset = gs.onset or ts
            inc = gs.incident = {"n": gs.n_incidents, "onset": onset, "night_hits": 0, "ok": 0, "leak": False,
                                 "excess": [], "type": "WATER_LOSS"}
            # recurring signature? same onset hour (+-2 h) within the last 48 h, night was clean
            rec = [h for h in gs.history if not h["leak"] and (onset - h["onset"]) <= timedelta(hours=50)
                   and min(abs(onset.hour - h["onset"].hour), 24 - abs(onset.hour - h["onset"].hour)) <= 2]
            info = {"excess": excess, "pct": 100 * (ratio - gs.mu), "kld": excess * 1.44,
                    "parent": self.sc.node_map[g.parent].name, "hour": onset.strftime("%H:%M"),
                    "count": len(rec) + 1}
            if rec:
                inc["type"] = "THEFT_SUSPECTED"
                am.raise_alert(("WL", g.id, inc["n"]), "THEFT_SUSPECTED", now, ts, g.id, info,
                               severity="critical" if len(rec) + 1 >= 3 else None)
            else:
                am.raise_alert(("WL", g.id, inc["n"]), "WATER_LOSS", now, ts, g.id, info)
        if inc is not None:
            inc["excess"].append(excess)
            if night and z > 3:
                inc["night_hits"] += 1
                if inc["night_hits"] >= 4 and not inc["leak"]:
                    inc["leak"] = True
                    inc["type"] = "LEAK"
                    am.raise_alert(("LEAK", g.id, inc["n"]), "LEAK_CONFIRMED", now, ts, g.id,
                                   {"excess": excess, "kld": excess * 1.44})
            inc["ok"] = inc["ok"] + 1 if z < 1.5 else 0
            if inc["ok"] >= 8:
                inc["end"] = ts
                gs.history.append(inc)
                gs.incident = None
                gs.S = 0.0
                gs.onset = None
                am.resolve(("WL", g.id, inc["n"]), now)
                am.resolve(("LEAK", g.id, inc["n"]), now)
        state = "ok" if gs.incident is None else (
            "leak" if gs.incident["leak"] else "theft" if gs.incident["type"] == "THEFT_SUSPECTED" else "loss")
        self.group_records.append({"ts": ts, "group": g.id, "ratio": ratio, "mu": gs.mu, "z": z, "S": gs.S,
                                   "excess_lpm": excess, "state": state})

    # =================================================================== equity
    def _equity(self, ts: datetime, now: datetime) -> None:
        day = (ts - timedelta(days=1)).date()
        rows = []
        for spec in self.sc.nodes:
            if spec.kind != "tap" or spec.population == 0:
                continue
            fl = self.daily_flow.get((spec.id, day), [])
            if len(fl) < 0.5 * 96:
                continue
            lpcd = float(np.mean(fl)) * 1440 / spec.population
            rows.append({"date": str(day), "node": spec.id, "village": spec.village, "lpcd": round(lpcd, 1),
                         "target": spec.lpcd_target, "pct": round(100 * lpcd / spec.lpcd_target, 1),
                         "equity": spec.equity})
        self.equity_rows.extend(rows)
        eq = [r for r in rows if r["equity"]]
        if not eq:
            return
        median = float(np.median([r["lpcd"] for r in eq]))
        for r in eq:
            prev = self.am.last.get(("EQUITY_SHORTFALL", r["node"]))
            if r["pct"] < 70 and (prev is None or now - prev.time >= timedelta(days=7)):
                self.am.raise_alert(("EQ", r["node"], r["date"]), "EQUITY_SHORTFALL", now, ts, r["node"],
                                    {"lpcd": r["lpcd"], "target": r["target"], "pct": r["pct"], "median": median})

    # ==================================================================== train
    def _train(self) -> None:
        self.trained = True
        for k, v in self.flow_hour.items():
            self.flow_med[k] = float(np.median(v))
        for gid, gs in self.gs.items():
            if len(gs.train) > 20:
                arr = np.array(gs.train)
                gs.mu = float(np.median(arr))
                mad = float(np.median(np.abs(arr - gs.mu))) * 1.4826   # robust sigma
                gs.sigma = max(0.01, mad, float(np.std(arr)) * 0.9)
        rows = []
        for nid, nr in self.ml_node_rows.items():
            a = np.array([row[:4] + [math.log((row[4] + 1) / (self.flow_med[(nid, int(row[5]))] + 1))]
                          for row in nr])
            mu, sd = a.mean(axis=0), np.maximum(a.std(axis=0), [0.05, 0.1, 5.0, 0.03, 0.08])
            self.ml_stats[nid] = (mu.tolist(), sd.tolist())
            rows.extend(((a - mu) / sd).tolist())
        self.model = IsolationForest(n_estimators=150, random_state=0).fit(np.array(rows))

    # ================================================================== summary
    def baselines(self) -> dict:
        return {gid: {"mu": gs.mu, "sigma": gs.sigma, "incidents": gs.n_incidents} for gid, gs in self.gs.items()}
