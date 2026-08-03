# IoT-IDS — Edge Intrusion Detection for IoT Networks

A machine-learning intrusion detection system for IoT networks, built around
**SFAF (Semantic Feature Alignment Framework)** and deployable as a real-time
sensor on a Raspberry Pi.

The project has two halves that share one idea — *a small, flow-based model can
detect network attacks in real time on cheap hardware*:

1. **SFAF research model** — trains a single model that generalises across
   three different public IDS datasets (CICIDS2017, UNSW-NB15, TON-IoT) by
   mapping their incompatible schemas into one 12-feature space. A CICIDS-only
   model scores 98% on CICIDS but collapses to **36%** on UNSW; the unified
   SFAF model recovers this to **93%** — a **+56.6 pp** generalisation gain.

2. **Live edge IDS** — a self-contained, real-time sensor. It sniffs traffic,
   aggregates packets into bidirectional flows, and classifies each flow with a
   ~55 KB ONNX model (feature scaler baked in) in **microseconds per flow**. It
   detects **6 attack classes** — port scan, SYN flood, ICMP flood, UDP flood,
   SSH brute-force, and slowloris — and reports aggregated, per-source alerts
   the way a real sensor does.

```
   ┌──────────────────────── shared detection core ────────────────────────┐
   │  packets → bidirectional flow table → 21-feature vector → ONNX → verdict │
   └────────────────────────────────────────────────────────────────────────┘
        ▲  Mac / dev : synthetic labeled pcaps        (root-free demo)
        ▲  Pi  / live: scapy sniff on eth0 / wlan0    (systemd service)
```

---

## Quickstart (dev machine, no root)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. build the labeled corpus from synthesized traffic
python attacks/build_corpus.py --scenarios 25

# 2. train the edge model + export ONNX (scaler baked in)
python src/train_live_model.py

# 3. run the full demonstration (generate traffic → detect → validate)
bash demo/run_demo.sh
```

`demo/run_demo.sh` replays a mixed benign+attack capture through the IDS with
live-style aggregated alerts, then scores held-out unseen scenarios.

## Deploy on a Raspberry Pi

The Pi runtime needs only `onnxruntime + numpy + scapy`. Full walkthrough
(flash → SSH → copy → install → run → demonstrate) is in
**[deploy/README_PI.md](deploy/README_PI.md)**.

```bash
# on the Pi
sudo bash deploy/setup_pi.sh eth0     # installs venv + systemd service
sudo systemctl start iot-ids
journalctl -u iot-ids -f              # watch live detections
```

Then launch attacks from another LAN host (`nmap`, `hping3`, `hydra`, …) — see
**[attacks/README.md](attacks/README.md)**.

---

## Repository layout

```
src/
  flow_features.py     shared packet→flow→feature extractor (train == serve)
  train_live_model.py  train the live edge model, export live_ids.onnx
  ids_daemon.py        the IDS: --pcap / --replay / --iface  (+ live sniff)
attacks/
  traffic_gen.py       scapy generators: benign + 6 attack classes (root-free)
  build_corpus.py      synth traffic → labeled flow dataset (flows.parquet)
  README.md            synthetic pcaps + real-tool (nmap/hping3/…) equivalents
demo/
  run_demo.sh          one-command end-to-end demonstration
  validate.py          held-out validation on unseen scenarios
  results/             confusion matrices, per-flow CSVs, reports
deploy/
  setup_pi.sh          one-shot Pi installer (venv + systemd)
  iot-ids.service      systemd unit
  requirements-pi.txt  minimal edge runtime deps
  README_PI.md         Raspberry Pi deployment guide
code/
  01_Dataset_EDA.ipynb        exploratory data analysis
  02_SFAF_Unified_Model.ipynb SFAF training notebook
  02_train_sfaf.py            headless SFAF reproduction (regenerates artifacts)
  03_Edge_Deployment.ipynb    edge benchmark notebook
  download_datasets.py        dataset fetch + instructions
models/
  live_ids.onnx        live edge model (scaler baked in, ~55 KB)  ← deployable
  live_meta.json       feature order, label map, metrics
  xgb_unified.json     SFAF unified model (12-feature)
  ...                  reports, plots, benchmarks
Literature/            reference papers
```

---

## Results

### Live edge IDS (this repo, reproducible with `demo/run_demo.sh`)

Held-out validation on **unseen-seed** scenarios (new IPs/ports/timings the
model never trained on):

| Metric | Value |
|---|---|
| Attack detection rate (recall) | 100% |
| Benign false-positive rate | 0% |
| Multiclass accuracy (7 classes) | ~100% |
| Model size (ONNX, scaler baked in) | ~55 KB |
| Inference | ~3–4 µs / flow |

> **Honest caveat.** These numbers are on *synthetic* traffic generated from
> attack templates, which is inherently separable — they validate that the
> **engineering pipeline** (feature extraction → model → live daemon) is
> correct and end-to-end, not real-world generalisation. The credible
> cross-domain generalisation claim is the SFAF result below, measured on
> three independent public datasets.

### SFAF research model (public datasets)

| Test dataset | Accuracy | F1 | AUC | Features |
|---|---|---|---|---|
| CICIDS2017 | 0.9833 | 0.9584 | 0.9985 | 12 |
| UNSW-NB15  | 0.9268 | 0.9432 | 0.9844 | 12 |
| TON-IoT    | 0.9941 | 0.9961 | 0.9997 | 12 |

Generalisation gap: a CICIDS-only baseline scores **98.4%** on CICIDS but only
**36.1%** on unseen UNSW-NB15; SFAF lifts UNSW to **92.7%** (**+56.6 pp**).
Edge model: **45 KB** ONNX (88.9% smaller than the 404 KB full model),
~0.01 ms/flow. Full numbers in `models/report_numbers.md`.

---

## Reproducing the SFAF research model

The three datasets are large and gated behind registration, so they are **not**
in the repo. To regenerate `scaler_unified_4dataset.pkl` and `xgb_edge.onnx`
and reproduce the table above:

```bash
python code/download_datasets.py    # Kaggle token, or manual instructions
python code/02_train_sfaf.py        # trains + exports artifacts + metrics
```

The ONNX export path is unit-tested; the only external requirement is dataset
access. Expected layout is documented in `code/download_datasets.py`.

---

## How detection works

Each bidirectional flow is summarised by **21 features** (see
`src/flow_features.py`): protocol, duration, packet/byte counts and rates,
packet-size statistics, inter-arrival statistics, TCP flag ratios, forward/
backward asymmetry, plus **host-context** features (distinct destination ports
and IPs per source in a rolling window). The host-context features are what let
a flow-level model separate reconnaissance (a port scan touches hundreds of
ports) from a flood (thousands of flows to one port), which are otherwise
identical at the single-flow level.

The daemon aggregates per-flow verdicts into **incidents** keyed by
`(source, attack-type)`, so a 500-port scan is one alert, not 500 lines.

> **Authorized-use only.** The attack generators and tool commands produce
> hostile traffic; confine them to hardware you own (your Pi + host).
