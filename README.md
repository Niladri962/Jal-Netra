# JalNetra — IoT Virtual Water Grid (complete project simulation)

JalNetra is an IoT-enabled virtual water grid for Telangana’s rural drinking water network. Low-cost solar sensor nodes track pH, chlorine, turbidity, TDS and flow, connecting over LoRa or GSM to a GIS dashboard. AI flags leaks, theft and contamination risk early, sending SMS alerts to panchayats and health workers before outbreaks.

An end-to-end, software-only simulation of **JalNetra**: low-cost solar IoT nodes on Telangana's
Mission Bhagiratha drinking-water network, a LoRa radio network, a cloud backend, an AI layer and
a GIS "virtual grid" dashboard. The simulation measures **quantity, quality and integrity** together.

Everything that can go wrong is **injected as ground truth**: leaks, illegal tapping, sewage
ingress, chlorination failure, saline borewell mixing, a frozen sensor, a vandalised node, a gateway
outage and a chronically under-served tail-end hamlet. The AI layer only sees what real sensors
would send, and its detections are scored against that truth.

```
physical network ──> sensor nodes ──> LoRa radio ──> gateway ──> cloud backend ──> AI engine ──> alerts + dashboard
(hydraulics,          (noise, drift,    (link budget,   (store &     (SQLite,          (risk score,     (SMS/WhatsApp
 chlorine decay,       faults, solar     SF/ADR,         forward      17-byte decode)   CUSUM balance,   EN + Telugu,
 water quality)        power, 17 B)      packet loss)    buffer)                        Isolation Forest) virtual grid)
```

## Live deployment (Vercel)

The repo deploys to Vercel with no configuration changes:

* **`public/`** holds the pre-built interface: `index.html` (10-node scenario) and `pilot.html`
  (school pilot). Rebuild it with `python build_site.py` after changing the simulator.
* **`api/`** holds the Python serverless functions that power the Scenario Lab:
  `POST /api/run`, `GET /api/config`, `GET /api/ping`. They need only NumPy and pandas;
  the Isolation Forest is implemented in NumPy (`jalnetra/iforest.py`).
* **`vercel.json`** serves `public/` and allows up to 60 s per simulation.

To deploy, push the repo to GitHub, then in Vercel choose **Add New → Project → Import** the
repository and press **Deploy**. Every later `git push` redeploys automatically.

## Run it

```bash
pip install -r requirements-dev.txt     # numpy, pandas (+ matplotlib for report figures)
python run_simulation.py                 # both scenarios, ~10 s
python run_simulation.py --scenario mandal --seed 42 --days 21
python run_simulation.py --benchmark 10  # robustness across 10 random seeds
```

Then open **`output/mandal/dashboard.html`** in a browser. The map tiles need internet; every
other panel works offline.

### Interactive app + Scenario Lab

```bash
python app.py            # opens http://localhost:8050
```

This serves the same interface plus a **Scenario Lab** (press `L` or click ⚗). In the Lab you can
switch injected problems on or off, move them in time, resize them (leak L/min, contamination NTU,
borewell share, ...), add your own, change the seed and the number of days, and then
**re-run the whole pipeline** (physics → sensors → LoRa → AI → alerts) from the browser.

Presenting it:

* **🎬 Guided tour** (`G`) walks through every injected problem. At each step it shows what
  happened, when JalNetra alerted, and *how the AI knew*. Ideal for a panel or interview demo.
* **🔗 Share** copies a deep link that reopens the exact time, node and tab
  (e.g. `#t=181&n=N10&tab=health`; add `&tour` to start the tour).
* **🛈 How it works** shows an animated architecture diagram with live numbers from the run.
* **⬇ CSV** exports a node's hourly data or the full alert log.

Interface features:

* Time replay with play/speed controls and an alert rail above the slider.
* A map with nodes that pulse while unsafe, and pipes that change colour by water-balance state.
* A clickable status timeline for every node and pipe.
* Synced chart crosshairs with click-to-seek, plus a "compare two nodes" option.
* Alert filters, search and acknowledge, with an English / Telugu / both toggle.
* Light and dark themes.
* Keyboard shortcuts: `Space` to play, `←/→` to step an hour, `Shift` with an arrow to step a day.

## Scenarios

| Scenario | What it is | Nodes | Duration |
|---|---|---|---|
| `mandal` | Mission Bhagiratha segment: WTP outlet → trunk main → 3 village OHSRs → 6 end-point taps (school, main habitations, tail-end ward, tail-end thanda). 9,650 people. Monsoon (July). | 10 | 14 days |
| `pilot` | The physical prototype: school sump outlet + drinking-water tap row + mid-day-meal kitchen tap, with the bench tests the student team would run. | 3 | 4 school days |

Village names and coordinates are **illustrative** (placed in Siddipet district) and do not
describe any real village's water.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q          # 17 tests: payload codec, LoRa airtime, risk index, Isolation Forest, Lab, end-to-end detection
```

GitHub Actions (`.github/workflows/ci.yml`) runs the tests, a 5-seed detection benchmark and the
site build on every push.

## Results (mandal scenario, default seed 7)

| | |
|---|---|
| Injected events detected | **9 / 9** |
| Alert precision | **100 %** (0 false of 19 actionable alerts) |
| Leak vs theft classification | **100 %** |
| Median detection latency | 90 min (leak: within the first 15-min sample) |
| Contamination early warning | alert **18.5 h before the first expected case**, **37.5 h before** the case cluster a status-quo system would notice |
| Illustrative cases (contamination event) | 68 → 13 with a boil-water advisory after the alert |
| Downstream low-chlorine alerts folded into one root-cause alert | 10 |
| 10-seed benchmark | recall 100 %, precision 98.5 % (mean), classification 100 % |

The health numbers come from a simple mechanism model (attack rate × exposure × incubation). They
show *why* lead time matters; they are not an epidemiological forecast.

## What each module does

| File | Role |
|---|---|
| `jalnetra/scenario.py` | Topology, populations, LPCD targets, injected events, contact roles |
| `jalnetra/physics.py` | Ground truth: diurnal demand, demand-driven tree hydraulics, background loss, leaks/theft, first-order chlorine decay with water age and temperature, pH/turbidity/TDS dynamics, contamination and mixing |
| `jalnetra/sensors.py` | Node emulation: sensor noise, 12-bit quantisation, pH drift, flow-meter start-up threshold, frozen sensors, 3 W solar + 2×18650 power budget, brown-out |
| `jalnetra/codec.py` | 17-byte big-endian payload (identical to the firmware), ORP + pH → free-chlorine conversion |
| `jalnetra/lora.py` | IN865 link budget, ADR-style SF choice, Semtech time-on-air formula, per-packet fading, Yagi upgrade + NbTrans for weak sites, gateway store-and-forward during backhaul outages |
| `jalnetra/backend.py` | SQLite ingest: raw uplinks (hex), decoded readings, alerts, outbox |
| `jalnetra/analytics.py` | The AI layer (below) |
| `jalnetra/alerts.py` | Routing by role, cool-downs, bilingual SMS (UCS-2 segment counting) + WhatsApp |
| `jalnetra/health.py` | ASHA case reports + status-quo vs JalNetra counterfactual |
| `jalnetra/evaluation.py` | Scores detections against injected ground truth |
| `jalnetra/dashboard.py` + `templates/dashboard.html` | Virtual-grid dashboard (Leaflet map + Chart.js, time-replay) |
| `jalnetra/report.py` | PNG figures + `REPORT.md` per run |
| `firmware/jalnetra_node/jalnetra_node.ino` | ESP32 firmware for the real node: same payload, deep-sleep duty cycle |

### The AI layer (`analytics.py`)

1. **Data-quality guard.** A sensor that repeats the same raw value for 12 samples (3 h) is
   frozen. It is excluded from scoring, and a maintenance ticket goes to the technician rather than
   a false health alert.
2. **Microbial Risk Index (0–100).** Built from the chlorine deficit and turbidity excess. The
   key term is their **interaction**: low chlorine *and* high turbidity together signal pathogen
   ingress. pH deviation and an Isolation-Forest anomaly score add smaller contributions.
   Green < 30, amber 30–60, red ≥ 60.
3. **Isolation Forest.** A multivariate anomaly model trained on the 3-day commissioning period,
   with features z-scored per node.
4. **Water balance + CUSUM.** For every segment: loss = (inflow − Σ outflows) / inflow, tested
   with a one-sided CUSUM on independent 15-min samples. This shows *which pipe segment* is losing
   water.
5. **Leak vs theft.** A loss that persists through 00:00–05:00 (minimum night flow) is a
   **physical leak**. A loss that recurs at the same evening hour while nights stay clean is an
   **illegal draw**; it escalates to critical on the 3rd recurrence.
6. **Spatial root cause.** Low chlorine at the WTP produces one SOURCE alert, and the downstream
   low-chlorine alerts that follow over the next hours are linked to it instead of flooding phones.
7. **Equity.** A daily per-capita supply (LPCD) check for each habitation against the 100 LPCD
   target. Supply below 70 % of target raises an alert.
8. **Network health.** "Node offline" (one node silent while the gateway is fine) is told apart
   from "gateway offline" (everything silent). After a gateway outage, the buffered data is
   back-filled and processed in order.

## Outputs (`output/<scenario>/`)

`dashboard.html` · `REPORT.md` · `figures/*.png` · `jalnetra.db` (SQLite) ·
`node_readings_processed.csv` · `water_balance.csv` · `equity_lpcd.csv` · `alerts.csv` ·
`alerts_outbox.txt` (every message, English + Telugu) · `ground_truth_physical.csv` ·
`evaluation.json`.

## Hardware notes

* **Indicative cost: ~₹13,850 per tap node.** The bill of materials is in the dashboard's
  Network & hardware tab. Reservoir and WTP nodes swap the hall-effect meter for a pulse-output
  Woltmann or clamp-on ultrasonic meter.
* **Power.** About 35 s awake every 15 min gives a 5.6 mA average draw (133 mAh/day). The
  6 V / 3 W panel harvests about 2 Ah/day in monsoon cloud, and the node lasts about **45 days
  with no sun**.
* **Chlorine.** Free chlorine is estimated from **ORP + pH**, which costs much less than an
  amperometric chlorine probe. It must be calibrated per site against a DPD colorimeter.
* **Telugu templates.** The Telugu alert templates are a first draft. They should be reviewed by
  a native speaker and the district team before use.

## Limitations (be upfront about these)

* All data is synthetic. The hydraulics are a demand-driven tree model, not EPANET, and the
  thresholds still need field tuning.
* The ORP→chlorine relation is site-specific and drifts as probes foul.
* Water-balance detection needs meters at both ends of a segment. Accuracy is limited by meter
  error, about ±2 % here.
* The health model is a mechanism demo with assumed attack rates.
