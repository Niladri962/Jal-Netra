"""Static figures (PNG) + REPORT.md for each run - ready to drop into a paper or policy brief."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .dashboard import BOM
from .simulation import RunResult

try:                      # figures are optional (pip install -r requirements-dev.txt)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
TEXT, TEXT2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e0", "#f6d5d5"

if plt is not None:
    plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": TEXT,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 140, "lines.linewidth": 1.8,
})


def _bands(ax, sc, match):
    for ev in sc.events:
        if not match(ev):
            continue
        if ev.daily_window:
            d = ev.start.replace(hour=0, minute=0)
            while d < ev.end:
                a, b = d + timedelta(hours=ev.daily_window[0]), d + timedelta(hours=ev.daily_window[1])
                ax.axvspan(max(a, ev.start), min(b, ev.end), color=BAND, lw=0, zorder=0)
                d += timedelta(days=1)
        else:
            ax.axvspan(ev.start, min(ev.end, sc.end), color=BAND, lw=0, zorder=0)


def _fmt_x(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def _alert_lines(ax, res, types, locs=None):
    for a in res.alerts:
        if a.type in types and (locs is None or a.location in locs):
            ax.axvline(a.time, color=TEXT, lw=1, ls=(0, (2, 2)))


def figures(res: RunResult) -> list[tuple[str, str]]:
    if plt is None:
        print("  (matplotlib not installed - skipping figures)")
        return []
    sc, out = res.sc, res.out_dir / "figures"
    out.mkdir(exist_ok=True)
    made = []
    nodes = res.nodes[~res.nodes.missing].copy()
    nodes["ts"] = pd.to_datetime(nodes["ts"])
    groups = res.groups.copy()
    groups["ts"] = pd.to_datetime(groups["ts"])

    # 1 - water balance per segment
    gids = [g.id for g in sc.groups]
    fig, axes = plt.subplots(len(gids), 1, figsize=(9, 1.9 * len(gids) + 0.6), sharex=True, squeeze=False)
    for ax, gid in zip(axes[:, 0], gids):
        d = groups[(groups.group == gid) & groups.ratio.notna()].set_index("ts")["ratio"].resample("1h").mean() * 100
        _bands(ax, sc, lambda e, gid=gid: e.target == gid)
        ax.plot(d.index, d.values, color=S1)
        mu = res.engine_stats["baselines"][gid]["mu"] * 100
        ax.axhline(mu, color=TEXT2, lw=1, ls="--")
        _alert_lines(ax, res, {"WATER_LOSS", "LEAK_CONFIRMED", "THEFT_SUSPECTED"}, {gid})
        ax.set_title(sc.group_map[gid].name, fontsize=9.5)
        ax.set_ylabel("loss %")
    _fmt_x(axes[-1, 0])
    fig.suptitle("Water balance per segment  (shaded = injected leak/theft, dotted = alert, dashed = learned baseline)",
                 x=0.01, ha="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(out / "fig1_water_balance.png")
    plt.close(fig)
    made.append(("fig1_water_balance.png", "Water-balance loss ratio per segment"))

    # 2 - contamination close-up
    cont = next((e for e in sc.events if e.type == "contamination"), None)
    if cont:
        nid = cont.target
        d = nodes[nodes.node == nid].set_index("ts")
        w0, w1 = cont.start - timedelta(hours=18), cont.end + timedelta(hours=18)
        d = d[(d.index >= w0) & (d.index <= w1)]
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        for ax, (col, lab, th) in zip(axes, [("cl", "Free chlorine (mg/L)", [0.2]), ("turb", "Turbidity (NTU)", [1, 5]),
                                             ("mri", "Microbial risk index", [30, 60])]):
            ax.axvspan(cont.start, cont.end, color=BAND, lw=0, zorder=0)
            ax.plot(d.index, d[col], color=S1)
            for t in th:
                ax.axhline(t, color=TEXT2, lw=1, ls="--")
            _alert_lines(ax, res, {"CONTAMINATION_RISK"}, {nid})
            ax.set_title(lab, fontsize=9.5)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        fig.suptitle(f"Contamination at {sc.location_name(nid)}: low chlorine x high turbidity -> red alert",
                     x=0.01, ha="left", fontsize=10, color=TEXT)
        fig.tight_layout()
        fig.savefig(out / "fig2_contamination.png")
        plt.close(fig)
        made.append(("fig2_contamination.png", "Contamination event: sensor signals and risk score"))

    # 3 - chlorination failure propagating downstream
    cf = next((e for e in sc.events if e.type == "chlorination_failure"), None)
    if cf:
        src = cf.target
        tails = [n.id for n in sc.nodes if n.kind == "tap"]
        pick = [src] + sorted(tails, key=lambda x: -sc.node_map[x].tau_up_h - sc.node_map[x].tau_local_h)[:3][::-1]
        w0, w1 = cf.start - timedelta(hours=8), cf.end + timedelta(hours=24)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.axvspan(cf.start, cf.end, color=BAND, lw=0, zorder=0)
        for c, nid in zip([S1, S2, S3, S4], pick):
            d = nodes[(nodes.node == nid) & (nodes.ts >= w0) & (nodes.ts <= w1)].set_index("ts")["cl"]
            ax.plot(d.index, d.values, color=c, label=f"{nid} {sc.node_map[nid].name}")
        ax.axhline(0.2, color=TEXT2, lw=1, ls="--")
        _alert_lines(ax, res, {"SOURCE_CHLORINATION_FAILURE"})
        ax.set_ylabel("free Cl mg/L")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax.set_title("Chlorination failure at source travels downstream with water age; one root-cause alert")
        fig.tight_layout()
        fig.savefig(out / "fig3_chlorination_failure.png")
        plt.close(fig)
        made.append(("fig3_chlorination_failure.png", "Chlorination failure propagation"))

    # 4 - health counterfactual
    if res.health["daily"]:
        h = pd.DataFrame(res.health["daily"]).groupby("date")[["cases_status_quo", "cases_jalnetra"]].sum()
        fig, ax = plt.subplots(figsize=(9, 3.2))
        x = range(len(h))
        ax.bar([i - 0.2 for i in x], h.cases_status_quo, width=0.38, color=S2, label="Without JalNetra (status quo)")
        ax.bar([i + 0.2 for i in x], h.cases_jalnetra, width=0.38, color=S1, label="With JalNetra")
        ax.set_xticks(list(x), [d[5:] for d in h.index], fontsize=8)
        ax.set_ylabel("cases / day")
        ax.legend(loc="upper left")
        ax.set_title("Simulated diarrhoeal cases (ASHA reports, all villages) - illustrative")
        fig.tight_layout()
        fig.savefig(out / "fig4_health.png")
        plt.close(fig)
        made.append(("fig4_health.png", "Health counterfactual"))

    # 5 - equity
    eq = res.equity
    if len(eq) and eq.equity.any():
        s = eq[eq.equity].groupby("node").lpcd.mean()
        fig, ax = plt.subplots(figsize=(9, 3.2))
        labels = [sc.node_map[n].name.replace(" habitation", "") for n in s.index]
        ax.bar(labels, s.values, color=S1, width=0.55)
        ax.axhline(100, color=TEXT2, ls="--", lw=1)
        ax.axhline(70, color="#d03b3b", ls=":", lw=1)
        for i, v in enumerate(s.values):
            ax.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=8.5, color=TEXT)
        ax.set_ylabel("LPCD")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title("Average litres per person per day (target 100, alert below 70)")
        fig.tight_layout()
        fig.savefig(out / "fig5_equity.png")
        plt.close(fig)
        made.append(("fig5_equity.png", "Supply equity by habitation"))
    return made


def build_report(res: RunResult) -> None:
    figs = figures(res)
    sc, e = res.sc, res.evaluation
    L = [f"# JalNetra simulation report - {sc.title}", "",
         f"*Region:* {sc.region}  ",
         f"*Period:* {sc.start:%d %b %Y} - {sc.end:%d %b %Y} ({sc.days} days, {sc.step_min}-min telemetry, "
         f"{len(sc.nodes)} nodes, first {sc.train_days} day(s) used as AI commissioning baseline)  ",
         f"*Seed:* {sc.seed} - runtime {res.runtime_s}s", "",
         "> All data is simulated. Village names and coordinates are illustrative; health figures are a mechanism",
         "> demonstration, not an epidemiological forecast.", "",
         "## Headline results", "",
         "| Metric | Value |", "|---|---|",
         f"| Injected events detected | **{e['events_detected']} / {e['events_total']}** |",
         f"| Alert precision (actionable alerts that matched a real event) | **{e['precision']:.0%}** "
         f"({e['false_alerts']} false of {e['alerts_total']}) |",
         f"| Median detection latency | {e['median_latency_min']} min |"]
    if e["classification_accuracy"] is not None:
        L.append(f"| Leak vs theft classification | {e['classification_accuracy']:.0%} |")
    L.append(f"| Downstream alerts linked to a root cause (alert storm avoided) | {res.engine_stats['linked_suppressed']} |")
    for h in res.health["events"]:
        L.append(f"| {h['event']} early-warning lead time vs status quo | {h['lead_time_h']} h "
                 f"(cases {h['cases_status_quo']} -> {h['cases_jalnetra']}) |")
    L += ["", "## Ground truth vs detection", "",
          "| Event | Injected problem | Start | First alert | Latency | Classification |", "|---|---|---|---|---|---|"]
    for r in e["events"]:
        lat = "-" if r["latency_min"] is None else (f"{r['latency_min']} min" if r["latency_min"] < 120
                                                     else f"{r['latency_min'] / 60:.1f} h")
        cls = "-" if not r["classification"] else (
            f"{'OK' if r['classification']['correct'] else 'WRONG'} {r['classification']['expected']}")
        L.append(f"| {r['event']} | {r['description']} | {r['start'][5:]} | {(r['first_alert'] or 'MISSED')[5:]} "
                 f"| {lat} | {cls} |")
    L += ["", "## Figures", ""]
    for f, cap in figs:
        L += [f"![{cap}](figures/{f})", ""]
    if res.health["events"]:
        L += ["## Public-health link", "",
              "| Event | Alert | First case expected | Status-quo detection | Lead time | Cases SQ -> JN |",
              "|---|---|---|---|---|---|"]
        for h in res.health["events"]:
            L.append(f"| {h['event']} {h['type']} | {h['alert_time']} | {h['first_case_expected']} | "
                     f"{h['status_quo_detection']} | {h['lead_time_h']} h | {h['cases_status_quo']} -> {h['cases_jalnetra']} |")
        L.append("")
    L += ["## Alerts dispatched", "", "| ID | Time | Severity | Type | Location | Recipients |", "|---|---|---|---|---|---|"]
    for a in res.alerts:
        L.append(f"| {a.id} | {a.time:%d %b %H:%M} | {a.severity} | {a.type} | {a.location_name} | {len(a.recipients)} |")
    L += ["", f"Messages in outbox: {len(res.outbox)} (see `alerts_outbox.txt`).", "",
          "## Radio and power", "", "| Node | km | SF | Antenna | RSSI dBm | Delivery | Airtime ms |",
          "|---|---|---|---|---|---|---|"]
    for r in res.radio:
        L.append(f"| {r['node']} | {r['distance_km']} | {r['sf']}{' x2' if r['nb_trans'] > 1 else ''} | {r['antenna']} "
                 f"| {r['rssi_dbm']} | {r['pdr']:.1%} | {r['airtime_ms']} |")
    p = res.power[0]
    L += ["", f"Average node current {p['avg_current_ma']} mA -> {p['use_mah_day']:.0f} mAh/day; "
          f"autonomy without sun {p['autonomy_days_no_sun']} days. Lowest battery seen across nodes: "
          f"{min(x['min_soc_pct'] for x in res.power)}%.", "",
          f"Indicative cost per tap node: Rs {sum(b[2] for b in BOM):,}.", "",
          "## Files", "",
          "- `dashboard.html` - interactive virtual-grid dashboard (open in a browser)",
          "- `jalnetra.db` - SQLite: uplinks (raw hex), readings, alerts, outbox",
          "- `node_readings_processed.csv`, `water_balance.csv`, `equity_lpcd.csv`, `alerts.csv`",
          "- `ground_truth_physical.csv` - what really happened in the pipes (for validation)",
          "- `alerts_outbox.txt` - every SMS / WhatsApp message, English + Telugu",
          "- `evaluation.json` - all scores", ""]
    (res.out_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
