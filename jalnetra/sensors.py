"""Emulation of one JalNetra field node (ESP32 + sensors + solar power).

Measurement chain per wake-up (every 15 min, ~35 s awake):
    power sensors via MOSFET -> settle -> sample pH, ORP, turbidity, TDS, temp,
    flow (pulse count) -> 12-bit ADC quantisation -> pack 17 bytes -> LoRa TX -> deep sleep.

Modelled imperfections: sensor noise, quantisation, slow pH calibration drift,
the flow meter's minimum detectable flow, stuck/frozen sensors, power loss.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from .codec import FLAG_BOOT, FLAG_LOW_BATT, free_chlorine_to_orp
from .scenario import EventSpec, NodeSpec

# ---- power budget (see README "Power budget") -----------------------------------------
BATT_MAH = 6000          # 2 x 18650 in parallel
AWAKE_S = 35             # warm-up + sampling + LoRa TX
AWAKE_MA = 140           # ESP32 active + sensors + SX1276 TX burst (time-averaged)
SLEEP_MA = 0.12          # ESP32 deep sleep + regulator quiescent
PANEL_W = 3.0            # 6 V / 3 W panel
CHARGE_EFF = 0.75        # MPPT charger + battery round-trip
V_BATT = 3.7


class SensorNode:
    def __init__(self, spec: NodeSpec, rng: np.random.Generator, step_min: int,
                 cloud_range: tuple[float, float]):
        self.spec = spec
        self.rng = rng
        self.step_min = step_min
        self.cloud_range = cloud_range
        self.seq = 0
        self.soc = 0.8 + 0.15 * rng.random()
        self.ph_drift_per_day = rng.normal(0, 0.004)
        self.t0: datetime | None = None
        self.stuck: dict[str, int] = {}
        self._cloud: dict = {}
        self.min_soc = self.soc
        self.energy_in_mah = 0.0
        self.energy_out_mah = 0.0

    # --------------------------------------------------------------------- power
    def _power(self, t: datetime) -> None:
        step_h = self.step_min / 60
        d = t.date()
        if d not in self._cloud:
            self._cloud[d] = self.rng.uniform(*self.cloud_range)
        h = t.hour + t.minute / 60
        irr = max(0.0, math.sin(math.pi * (h - 6) / 12))
        harvest = PANEL_W * irr * self._cloud[d] * CHARGE_EFF / V_BATT * 1000 * step_h
        use = (AWAKE_S * AWAKE_MA + (self.step_min * 60 - AWAKE_S) * SLEEP_MA) / 3600
        self.energy_in_mah += harvest
        self.energy_out_mah += use
        self.soc = min(1.0, max(0.0, self.soc + (harvest - use) / BATT_MAH))
        self.min_soc = min(self.min_soc, self.soc)

    # ----------------------------------------------------------------- sampling
    def sample(self, t: datetime, truth: dict, active: list[EventSpec]) -> dict | None:
        """Return raw integer fields for the payload, or None if the node is silent."""
        if self.t0 is None:
            self.t0 = t
        self._power(t)
        for ev in active:
            if ev.type == "node_down" and ev.target == self.spec.id:
                return None
        if self.soc < 0.05:          # brown-out protection: stay asleep
            return None

        rng = self.rng
        days = (t - self.t0).total_seconds() / 86400
        ph = truth["ph"] + rng.normal(0, 0.03) + self.ph_drift_per_day * days
        orp = free_chlorine_to_orp(truth["cl"], truth["ph"]) + rng.normal(0, 5)
        turb = max(0.0, truth["turb"] * (1 + rng.normal(0, 0.05)) + rng.normal(0, 0.08))
        tds = max(0.0, truth["tds"] * (1 + rng.normal(0, 0.015)))
        flow = truth["flow"] * (1 + rng.normal(0, 0.02))
        if flow < 0.5:               # below hall-sensor start-up flow
            flow = 0.0
        temp = truth["temp"] + rng.normal(0, 0.2)

        raw = {
            "node": self.spec.num,
            "seq": self.seq,
            "flags": (FLAG_LOW_BATT if self.soc < 0.2 else 0) | (FLAG_BOOT if self.seq == 0 else 0),
            "ph_x100": int(round(min(14, max(0, ph)) * 100)),
            "orp_mv": int(round(min(2000, max(-2000, orp)))),
            "turb_x10": int(round(min(6553, turb) * 10)),
            "tds": int(round(min(65535, tds))),
            "flow_x10": int(round(min(6553, flow) * 10)),
            "temp_x10": int(round(temp * 10)),
            "batt": int(round((3300 + 900 * self.soc - 2500) / 10)),
        }
        # stuck / frozen sensor faults: the field repeats its first faulty value
        for ev in active:
            if ev.type == "sensor_stuck" and ev.target == self.spec.id:
                key = {"ph": "ph_x100", "orp": "orp_mv", "turb": "turb_x10", "tds": "tds"}[ev.params["sensor"]]
                self.stuck.setdefault(key, raw[key])
                raw[key] = self.stuck[key]
        if not any(ev.type == "sensor_stuck" and ev.target == self.spec.id for ev in active):
            self.stuck.clear()
        self.seq = (self.seq + 1) & 0xFFFF
        return raw

    @property
    def avg_current_ma(self) -> float:
        return (AWAKE_S * AWAKE_MA + (self.step_min * 60 - AWAKE_S) * SLEEP_MA) / (self.step_min * 60)

    @property
    def autonomy_days(self) -> float:
        """Days the node runs from a full battery with zero sun."""
        return BATT_MAH / (self.avg_current_ma * 24)
