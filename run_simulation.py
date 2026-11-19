"""JalNetra - run the complete project simulation.

    python run_simulation.py                    # both scenarios (mandal + school pilot)
    python run_simulation.py --scenario mandal  # 10-node Mission Bhagiratha segment, 14 days
    python run_simulation.py --scenario pilot   # 3-node school prototype, 4 days
    python run_simulation.py --seed 42 --days 21

Outputs land in output/<scenario>/ : dashboard.html, REPORT.md, figures/, CSVs,
SQLite database, and the SMS/WhatsApp outbox.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jalnetra.dashboard import build_dashboard
from jalnetra.report import build_report
from jalnetra.scenario import SCENARIOS
from jalnetra.simulation import run

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--benchmark", type=int, metavar="N", help="run the mandal scenario on N random seeds and score")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.benchmark:
        return benchmark(args.benchmark)

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for key in keys:
        kw = {}
        if args.seed is not None:
            kw["seed"] = args.seed
        if args.days is not None:
            kw["days"] = args.days
        sc = SCENARIOS[key](**kw)
        out = ROOT / "output" / key
        print(f"\n=== JalNetra simulation: {sc.title} ({sc.days} days, {len(sc.nodes)} nodes) ===")
        res = run(sc, out, verbose=not args.quiet)
        build_report(res)
        dash = build_dashboard(res)
        e = res.evaluation
        print(f"\n  Events detected : {e['events_detected']}/{e['events_total']}   "
              f"alert precision: {e['precision']:.0%}   false alerts: {e['false_alerts']}")
        if e["classification_accuracy"] is not None:
            print(f"  Leak/theft classification accuracy: {e['classification_accuracy']:.0%}")
        for h in res.health["events"]:
            print(f"  Health [{h['event']}]: alert {h['alert_time']} | status-quo detection "
                  f"{h['status_quo_detection']} | lead {h['lead_time_h']} h | cases "
                  f"{h['cases_status_quo']} -> {h['cases_jalnetra']}")
        print(f"  Runtime {res.runtime_s}s")
        print(f"  Dashboard : {dash}")
        print(f"  Report    : {out / 'REPORT.md'}")
    return 0


def benchmark(n: int) -> int:
    import pandas as pd
    import tempfile
    rows = []
    for seed in range(100, 100 + n):
        sc = SCENARIOS["mandal"](seed=seed)
        with tempfile.TemporaryDirectory() as tmp:
            res = run(sc, Path(tmp), verbose=False)
        e = res.evaluation
        row = {"seed": seed, "recall": e["recall"], "precision": e["precision"], "false_alerts": e["false_alerts"],
               "classification": e["classification_accuracy"], "median_latency_min": e["median_latency_min"]}
        for r in e["events"]:
            row[f"{r['event']}_{r['type']}_latency_min"] = r["latency_min"]
        rows.append(row)
        print(f"  seed {seed}: recall {e['recall']:.0%}  precision {e['precision']:.0%}  "
              f"classification {e['classification_accuracy']:.0%}  median latency {e['median_latency_min']} min")
    df = pd.DataFrame(rows)
    out = ROOT / "output" / "benchmark.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Mean recall {df.recall.mean():.1%} | mean precision {df.precision.mean():.1%} | "
          f"classification {df.classification.mean():.1%} | median latency {df.median_latency_min.median():.0f} min")
    print(f"  Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
