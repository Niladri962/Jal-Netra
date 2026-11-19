"""Public-health link: synthetic diarrhoeal cases reported by ASHA workers, and a counterfactual.

Two worlds are compared for every contamination-type event:

* status quo  - nobody knows the water is unsafe until a case cluster is noticed by the
                health system (ASHA daily report); repair follows ~24 h after that.
* JalNetra    - a boil-water advisory goes out ~2 h after the alert (assumed to cut
                exposure by 70%) and the line is repaired at the scheduled event end.

Cases = attack-rate x exposed population x exposure-days, spread over a log-normal
incubation period (median 30 h). The numbers are ILLUSTRATIVE: they show the mechanism
(alert lead time -> fewer cases), not an epidemiological forecast.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from .scenario import Scenario

INCUB_MEDIAN_H, INCUB_SIGMA = 30.0, 0.5
BASELINE_PER_1000_DAY = 0.12
ADVISORY_DELAY_H, ADVISORY_EFFECT = 2.0, 0.7
REPORT_HOUR = 20
RESPONSE_H = 24


def _lognorm_cdf(h: float) -> float:
    if h <= 0:
        return 0.0
    return 0.5 * (1 + math.erf((math.log(h) - math.log(INCUB_MEDIAN_H)) / (INCUB_SIGMA * math.sqrt(2))))


def _expected_daily(exposure: list[tuple[datetime, float]], attack: float, pop: int, day0: datetime, ndays: int,
                    step_h: float) -> np.ndarray:
    out = np.zeros(ndays)
    for t, inten in exposure:
        if inten <= 0:
            continue
        cases = attack * pop * inten * step_h / 24
        for d in range(ndays):
            a = (day0 + timedelta(days=d) - t).total_seconds() / 3600
            b = a + 24
            out[d] += cases * (_lognorm_cdf(b) - _lognorm_cdf(a))
    return out


def health_model(sc: Scenario, alerts, rng: np.random.Generator) -> dict:
    events = [e for e in sc.events if e.params.get("attack_rate", 0) > 0]
    villages = defaultdict(int)
    for n in sc.nodes:
        if n.kind == "tap" and n.profile == "residential":
            villages[n.village] += n.population
    if not events or not villages:
        return {"daily": [], "events": []}
    day0 = datetime.combine(sc.start.date(), datetime.min.time())
    ndays = sc.days + 4
    step_h = sc.step_min / 60
    exp_jn = {v: np.full(ndays, villages[v] / 1000 * BASELINE_PER_1000_DAY) for v in villages}
    exp_sq = {v: arr.copy() for v, arr in exp_jn.items()}
    summary = []
    for ev in events:
        locs = sc.event_locations(ev) if ev.type != "chlorination_failure" else set(sc.descendants(ev.target))
        pops = defaultdict(int)
        for nid in locs:
            n = sc.node_map[nid]
            if n.kind == "tap" and n.profile == "residential":
                pops[n.village] += n.population
        hits = sorted(a.time for a in alerts if a.type in ev.expected_alerts and ev.start <= a.time <= ev.end
                      and (a.location in locs or a.location == ev.target))
        t_alert = hits[0] if hits else None
        steps = int((ev.end - ev.start).total_seconds() / 3600 / step_h)
        times = [ev.start + timedelta(hours=i * step_h) for i in range(steps)]

        def jn_int(t):
            if t_alert is None or t < t_alert + timedelta(hours=ADVISORY_DELAY_H):
                return 1.0
            return 1 - ADVISORY_EFFECT
        attack = ev.params["attack_rate"]
        # status quo: exposure continues until the case cluster is noticed + response time
        long_times = [ev.start + timedelta(hours=i * step_h) for i in range(int(10 * 24 / step_h))]
        sq_end, surge_t = ev.end, None
        for v, p in pops.items():
            trial = _expected_daily([(t, 1.0) for t in long_times], attack, p, day0, ndays, step_h)
            base = p / 1000 * BASELINE_PER_1000_DAY
            for d in range(ndays):
                if trial[d] + base >= max(4.0, 3 * base):
                    st = day0 + timedelta(days=d, hours=REPORT_HOUR)
                    if surge_t is None or st < surge_t:
                        surge_t = st
                    break
        if surge_t is not None:
            sq_end = max(ev.end, surge_t + timedelta(hours=RESPONSE_H))
        sq_times = [ev.start + timedelta(hours=i * step_h)
                    for i in range(int((sq_end - ev.start).total_seconds() / 3600 / step_h))]
        tot_jn = tot_sq = 0.0
        first_case = None
        for v, p in pops.items():
            e_jn = _expected_daily([(t, jn_int(t)) for t in times], attack, p, day0, ndays, step_h)
            e_sq = _expected_daily([(t, 1.0) for t in sq_times], attack, p, day0, ndays, step_h)
            exp_jn[v] += e_jn
            exp_sq[v] += e_sq
            tot_jn += e_jn.sum()
            tot_sq += e_sq.sum()
            # time at which the first symptomatic case is expected (cumulative expected >= 1)
            h = 0.0
            while h < 240:
                cum_cases = sum(attack * p * step_h / 24 * _lognorm_cdf((ev.start + timedelta(hours=h) - t).total_seconds() / 3600)
                                for t in times[: int(h / step_h) + 1])
                if cum_cases >= 1:
                    fc = ev.start + timedelta(hours=h)
                    first_case = fc if first_case is None or fc < first_case else first_case
                    break
                h += 1
        summary.append({
            "event": ev.id, "type": ev.type, "description": ev.description,
            "exposed_population": int(sum(pops.values())),
            "alert_time": t_alert.strftime("%Y-%m-%d %H:%M") if t_alert else None,
            "first_case_expected": first_case.strftime("%Y-%m-%d %H:%M") if first_case else None,
            "status_quo_detection": surge_t.strftime("%Y-%m-%d %H:%M") if surge_t else "not detected",
            "lead_time_h": round((surge_t - t_alert).total_seconds() / 3600, 1) if (surge_t and t_alert) else None,
            "alert_before_first_case_h": round((first_case - t_alert).total_seconds() / 3600, 1)
            if (first_case and t_alert) else None,
            "cases_status_quo": round(tot_sq, 1), "cases_jalnetra": round(tot_jn, 1),
            "cases_averted": round(tot_sq - tot_jn, 1),
        })
    daily = []
    for v in villages:
        for d in range(sc.days):
            daily.append({"date": str((day0 + timedelta(days=d)).date()), "village": v,
                          "cases_jalnetra": int(rng.poisson(exp_jn[v][d])),
                          "cases_status_quo": int(rng.poisson(exp_sq[v][d])),
                          "expected_jalnetra": round(float(exp_jn[v][d]), 2),
                          "expected_status_quo": round(float(exp_sq[v][d]), 2)})
    return {"daily": daily, "events": summary}
