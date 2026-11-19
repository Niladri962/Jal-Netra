"""Test suite: unit tests for each layer + end-to-end detection checks.

    pip install -r requirements.txt pytest
    pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jalnetra.analytics import microbial_risk  # noqa: E402
from jalnetra.codec import SIZE, decode, encode, free_chlorine_to_orp, orp_to_free_chlorine  # noqa: E402
from jalnetra.iforest import IsolationForest  # noqa: E402
from jalnetra.lab import build_from_config, lab_config  # noqa: E402
from jalnetra.lora import airtime_ms, path_loss_db  # noqa: E402
from jalnetra.scenario import mandal_scenario, pilot_scenario  # noqa: E402
from jalnetra.simulation import run  # noqa: E402
from jalnetra.webapi import run_lab  # noqa: E402


# ------------------------------------------------------------------ codec / payload
def test_payload_is_17_bytes_and_round_trips():
    raw = {"node": 10, "seq": 65535, "flags": 5, "ph_x100": 742, "orp_mv": -120, "turb_x10": 123,
           "tds": 1450, "flow_x10": 5438, "temp_x10": -15, "batt": 169}
    b = encode(raw)
    assert SIZE == 17 and len(b) == 17
    d = decode(b)
    for k, v in raw.items():
        assert d[k] == v
    assert d["ph"] == pytest.approx(7.42) and d["batt_mv"] == 4190


@pytest.mark.parametrize("cl,ph", [(0.05, 7.0), (0.2, 7.4), (1.0, 7.8), (2.0, 6.9)])
def test_orp_chlorine_conversion_is_inverse(cl, ph):
    assert orp_to_free_chlorine(free_chlorine_to_orp(cl, ph), ph) == pytest.approx(cl, rel=1e-9)


# ------------------------------------------------------------------ radio
def test_lora_airtime_matches_semtech_formula():
    assert airtime_ms(30, 7) == pytest.approx(71.9, abs=0.1)          # 17 B payload + 13 B LoRaWAN header
    times = [airtime_ms(30, sf) for sf in range(7, 13)]
    assert times == sorted(times)                                     # higher SF -> longer on air


def test_path_loss_grows_with_distance():
    assert path_loss_db(1) < path_loss_db(5) < path_loss_db(10)


# ------------------------------------------------------------------ AI building blocks
def test_microbial_risk_interaction():
    assert microbial_risk(1.0, 0.5, 7.4) < 5                          # clean water
    assert 20 < microbial_risk(0.05, 0.5, 7.4) < 30                   # chlorine gone, water clear -> watch
    assert microbial_risk(0.05, 12.0, 7.2) >= 60                      # low Cl x high turbidity -> unsafe
    assert microbial_risk(None, None, None) == 0                      # faulty sensors never fabricate risk


def test_isolation_forest_separates_outliers():
    rng = np.random.default_rng(0)
    model = IsolationForest(100, random_state=1).fit(rng.normal(size=(1500, 5)))
    inliers = model.decision_function(rng.normal(size=(50, 5)))
    outliers = model.decision_function(rng.normal(size=(50, 5)) + 5)
    assert inliers.mean() > 0 > outliers.mean()
    assert (outliers < inliers.min()).mean() > 0.9


# ------------------------------------------------------------------ scenario lab
def test_lab_config_round_trips_to_same_events():
    sc = mandal_scenario()
    rebuilt = build_from_config(lab_config(sc))
    assert [e.id for e in rebuilt.events] == [e.id for e in sc.events]
    assert [e.expected_alerts for e in rebuilt.events] == [e.expected_alerts for e in sc.events]


def test_lab_disable_and_daytime_leak_expectations():
    cfg = lab_config(mandal_scenario())
    for e in cfg["events"]:
        e["enabled"] = e["id"] == "E1"
    cfg["events"][0].update(start_h=4 * 24 + 9, duration_h=4)       # daytime-only leak
    sc = build_from_config(cfg)
    assert [e.id for e in sc.events] == ["E1"]
    assert sc.events[0].expected_alerts == ("WATER_LOSS",)           # cannot be night-confirmed
    assert sc.node_map["N10"].supply_factor == 1.0                   # inequity switched off


def test_run_lab_rejects_bad_input():
    with pytest.raises(ValueError):
        run_lab({"scenario": "atlantis"})
    with pytest.raises(ValueError):
        run_lab({"scenario": "pilot", "events": [{}] * 100})


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def mandal_result(tmp_path_factory):
    return run(mandal_scenario(), tmp_path_factory.mktemp("mandal"), verbose=False)


def test_mandal_detects_every_event_without_false_alerts(mandal_result):
    ev = mandal_result.evaluation
    assert ev["events_detected"] == ev["events_total"] == 9
    assert ev["false_alerts"] == 0
    assert ev["classification_accuracy"] == 1.0


def test_contamination_alert_beats_illness(mandal_result):
    h = {e["event"]: e for e in mandal_result.health["events"]}["E4"]
    assert h["alert_before_first_case_h"] > 6
    assert h["cases_jalnetra"] < h["cases_status_quo"]


def test_root_cause_linking_suppresses_alert_storm(mandal_result):
    types = [a.type for a in mandal_result.alerts]
    assert types.count("SOURCE_CHLORINATION_FAILURE") == 1
    assert "LOW_CHLORINE" not in types
    assert mandal_result.engine_stats["linked_suppressed"] > 0


def test_pilot_bench_tests(tmp_path):
    ev = run(pilot_scenario(), tmp_path, verbose=False).evaluation
    assert ev["recall"] == 1.0 and ev["false_alerts"] == 0


def test_no_events_means_no_alerts(tmp_path):
    cfg = lab_config(mandal_scenario())
    cfg["events"] = []
    cfg["days"] = 7
    res = run(build_from_config(cfg), tmp_path, verbose=False)
    assert [a.type for a in res.alerts] == []
