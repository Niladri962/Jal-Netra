"""Ground-truth physical model of the piped network.

Hydraulics
    Demand-driven tree network. Each end-point tap draws
    ``population x LPCD x diurnal-profile x supply_factor``. A parent node's
    flow is the sum of its children plus any leak / illegal draw in that
    segment, grossed up by the segment's background (physical) loss.

Water quality
    * Free chlorine follows first-order bulk decay  C = C0(t - tau) * exp(-k tau),
      where water age ``tau`` grows when local flow is low (night, tail-ends),
      and k rises with temperature (Arrhenius-style 1.07^(T-25)).
    * pH, turbidity, TDS carry slow AR(1) natural variation.
    * Events (contamination, groundwater mixing, chlorination failure) perturb
      these signals the way they would in the field.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from .scenario import EventSpec, Scenario

_RES = [0.35, 0.30, 0.30, 0.35, 0.60, 1.30, 1.90, 2.00, 1.60, 1.20, 1.00, 0.90,
        0.90, 0.85, 0.80, 0.85, 1.10, 1.50, 1.70, 1.40, 1.00, 0.70, 0.50, 0.40]
_SCH = [0.05] * 7 + [0.6, 2.0, 2.5, 2.5, 3.0, 4.0, 3.5, 2.5, 2.0, 1.0] + [0.05] * 7
PROFILES = {
    "residential": [v / (sum(_RES) / 24) for v in _RES],
    "school": [v / (sum(_SCH) / 24) for v in _SCH],
}


def profile_value(profile: str, t: datetime) -> float:
    p = PROFILES[profile]
    h = t.hour + t.minute / 60
    i = int(h) % 24
    f = h - int(h)
    v = p[i] * (1 - f) + p[(i + 1) % 24] * f
    if profile == "school" and t.weekday() == 6:   # Sunday
        v *= 0.05
    return v


def water_temp(t: datetime) -> float:
    h = t.hour + t.minute / 60
    return 27.0 + 2.5 * math.sin(2 * math.pi * (h - 9) / 24)


class PhysicalNetwork:
    def __init__(self, sc: Scenario, rng: np.random.Generator):
        self.sc = sc
        self.rng = rng
        # post-order: children before parents
        self.order: list[str] = []

        def visit(nid):
            for c in sc.children[nid]:
                visit(c)
            self.order.append(nid)

        for n in sc.nodes:
            if n.parent is None:
                visit(n.id)
        self.nominal: dict[str, float] = {}
        for nid in self.order:
            spec = sc.node_map[nid]
            if spec.kind == "tap":
                self.nominal[nid] = spec.population * spec.lpcd_target / 1440 * spec.supply_factor
            else:
                g = f"G-{nid}"
                self.nominal[nid] = sum(self.nominal[c] for c in sc.children[nid]) / (1 - sc.base_loss.get(g, 0.04))
        self._ar: dict[tuple[str, str], float] = defaultdict(float)
        self._day_factor: dict[tuple[str, object], float] = {}
        self.cl_src_hist: list[float] = []

    def _ar1(self, nid: str, var: str, phi: float = 0.9) -> float:
        x = phi * self._ar[(nid, var)] + math.sqrt(1 - phi * phi) * self.rng.normal()
        self._ar[(nid, var)] = x
        return x

    def _dayf(self, nid: str, t: datetime) -> float:
        key = (nid, t.date())
        if key not in self._day_factor:
            self._day_factor[key] = 1 + 0.05 * self.rng.normal()
        return self._day_factor[key]

    def step(self, k: int, t: datetime, active: list[EventSpec]) -> dict[str, dict]:
        sc = self.sc
        h = t.hour + t.minute / 60
        night = h < 5

        # ---------------------------------------------------------- hydraulics
        event_loss: dict[str, float] = defaultdict(float)
        for ev in active:
            if ev.type == "leak":   # pressure-driven: leaks run harder at night (higher pressure)
                event_loss[ev.target] += ev.params["lpm"] * (1.15 if night else 1.0)
            elif ev.type == "theft":
                event_loss[ev.target] += ev.params["lpm"] * (0.9 + 0.2 * self.rng.random())

        flows: dict[str, float] = {}
        for nid in self.order:
            spec = sc.node_map[nid]
            if spec.kind == "tap":
                base = spec.population * spec.lpcd_target / 1440 * spec.supply_factor
                f = base * profile_value(spec.profile, t) * self._dayf(nid, t) * (1 + 0.04 * self._ar1(nid, "q", 0.8))
                flows[nid] = max(f, 0.0)
            else:
                g = f"G-{nid}"
                loss = sc.base_loss.get(g, 0.04) + 0.002 * self._ar1(nid, "loss", 0.95)
                flows[nid] = (sum(flows[c] for c in sc.children[nid]) + event_loss[g]) / (1 - loss)

        # ------------------------------------------------------- source dosing
        dose = sc.cl_dose * (1 + 0.03 * self._ar1("SRC", "dose", 0.9))
        for ev in active:
            if ev.type == "chlorination_failure":
                dose = ev.params["dose"] * (1 + 0.1 * self.rng.normal())
        self.cl_src_hist.append(max(dose, 0.0))

        temp = water_temp(t)
        kb = sc.decay_k * 1.07 ** (temp - 25)

        # --------------------------------------------------------- per node WQ
        truth: dict[str, dict] = {}
        for spec in sc.nodes:
            nid = spec.id
            flow = flows[nid]
            fr = min(2.5, max(0.6, self.nominal[nid] / max(flow, 1e-6)))
            tau = spec.tau_up_h + spec.tau_local_h * fr
            lag = int(round(tau * 60 / sc.step_min))
            c0 = self.cl_src_hist[max(0, k - lag)]
            cl = c0 * math.exp(-kb * tau)
            ph = spec.base_ph + 0.04 * self._ar1(nid, "ph", 0.95)
            turb = max(0.05, spec.base_turb * (1 + 0.12 * self._ar1(nid, "turb", 0.9)))
            tds = spec.base_tds * (1 + 0.01 * self._ar1(nid, "tds", 0.97))

            for ev in active:
                if nid not in sc.event_locations(ev) and ev.target != nid:
                    continue
                if ev.type == "contamination":
                    ramp = min(1.0, (t - ev.start).total_seconds() / 3600)
                    p = ev.params
                    cl *= 1 - (1 - p["cl_factor"]) * ramp
                    turb += p["turb_add"] * ramp * (0.85 + 0.15 * math.sin(k / 3) + 0.08 * self.rng.normal())
                    tds += p.get("tds_add", 0) * ramp
                    ph += p.get("ph_shift", 0) * ramp
                elif ev.type == "groundwater_mixing":
                    fr_mix = ev.params["fraction"]
                    tds = (1 - fr_mix) * tds + fr_mix * ev.params["tds"]
                    ph += 0.15
                    turb += 0.3

            truth[nid] = {"flow": flow, "cl": max(cl, 0.0), "ph": ph, "turb": max(turb, 0.0),
                          "tds": tds, "temp": temp, "water_age_h": tau}
        truth["_groups"] = {g: {"event_loss_lpm": event_loss[g]} for g in sc.group_map}
        return truth
