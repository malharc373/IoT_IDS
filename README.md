# IoT-IDS — Edge Intrusion Detection & Prevention for IoT Networks

A machine-learning IDS/IPS for IoT networks, built around **SFAF (Semantic
Feature Alignment Framework)** and deployable as a real-time sensor on hardware
from a Raspberry Pi down to an ESP32-class microcontroller.

Two halves share one idea — *a small, flow-based model can detect and stop
network attacks in real time on cheap hardware*:

1. **SFAF research model** — one model that generalises across five public IDS
   datasets (CICIDS2017, UNSW-NB15, TON-IoT, CIC-IoT-2023, Bot-IoT) by mapping
   their incompatible schemas into one 12-feature space. A CICIDS-only model
   scores 98% on CICIDS but collapses to **36%** on unseen UNSW; the unified
   SFAF model recovers this to **93%** — a **+56.6 pp** generalisation gain.

2. **Live edge IDS/IPS** — a self-contained real-time sensor. It sniffs traffic,
   aggregates packets into bidirectional flows, and classifies each flow with a
   ~90 KB ONNX model (scaler baked in) in **microseconds per flow**. It detects
   **9 attack types across 4 categories**, reports aggregated per-source
   incidents, and can **actively block** offenders (IPS mode). The same model
   also compiles to a **dependency-free C header** for microcontrollers.

```
   ┌────────────────────────── shared detection core ──────────────────────────┐
   │ packets → bidirectional flow table → 22-feature vector → model → verdict    │
   │                                                          ↳ IPS block/limit  │
   └────────────────────────────────────────────────────────────────────────────┘
       ▲ Mac / dev : synthetic labeled pcaps        (root-free demo)
       ▲ Pi  / live: scapy sniff on eth0 / wlan0    (systemd service, IPS)
       ▲ MCU       : models/live_ids.h              (no runtime, ~42 KB const)
```

### Attack taxonomy (hierarchical)

| Category | Types |
|---|---|
| **recon** | portscan, xmas_scan (Xmas/NULL/FIN stealth scans) |
| **dos** | synflood, udpflood, icmpflood, mqtt_flood, slowloris |
| **botnet** | mirai (telnet/ssh propagation) |
| **bruteforce** | ssh_bruteforce |

Alerts read `category/type`, e.g. `⚠ ATTACK recon/portscan src=… 593 dst-ports`.

---

## Quickstart (dev machine, no root)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python attacks/build_corpus.py --scenarios 30   # synth traffic -> labeled flows
python src/train_live_model.py                  # train + export ONNX (+ booster)
bash demo/run_demo.sh                            # generate traffic -> detect -> validate
```

Detect + **prevent** (dry-run IPS shows what it *would* block):

```bash
python src/ids_daemon.py --replay data/pcaps/demo_mixed.pcap --ips
```

Live **web dashboard** (reads the alert feed; stdlib-only, Pi-friendly):

```bash
python src/ids_daemon.py --replay data/pcaps/demo_mixed.pcap --ips   # writes alerts
python src/dashboard.py                                              # http://localhost:8080
```

Compile the model for a microcontroller:

```bash
python src/export_c.py --verify     # writes models/live_ids.h, checks parity
```

## Deploy on a Raspberry Pi (IDS or IPS)

Runtime needs only `onnxruntime + numpy + scapy`. Full walkthrough in
**[deploy/README_PI.md](deploy/README_PI.md)**.

```bash
sudo bash deploy/setup_pi.sh eth0        # venv + systemd service
sudo systemctl start iot-ids
# IPS mode (actually blocks via nftables/iptables):
sudo .venv/bin/python src/ids_daemon.py --iface eth0 --prevent --allow 192.168.1.0/24
```

Launch attacks from another LAN host — see **[attacks/README.md](attacks/README.md)**.

---

## Results

### Live edge IDS (reproducible with `demo/run_demo.sh`)

Held-out validation on **unseen-seed** scenarios (new IPs/ports/timings),
including realistic packet-size noise and *hard-benign* traffic that resembles
attacks (bursty transfers with flood-like rates, multi-endpoint telemetry):

| Metric | Value |
|---|---|
| Multiclass accuracy (10 classes) | **99.9%** |
| Attack detection rate (recall) | 99.9% |
| Benign false-positive rate | **0.0%** |
| Hardest class | `ssh_bruteforce` (recall ~94.5%) |
| Model size (ONNX / C const-data) | ~90 KB / ~42 KB |
| Inference | ~6 µs/flow |

> **Honest caveat.** This is *synthetic* traffic. Adding size noise and
> attack-resembling benign traffic dropped the score from a suspicious 100% to a
> realistic 99.9% with an identifiable weak spot (brute-force vs benign is
> genuinely subtle at flow level) — evidence the model is learning behaviour,
> not memorising. But real-world accuracy can only be claimed on real labeled
> traffic: that is the SFAF result below and the purpose of the 5-dataset
> pipeline (`code/02_train_sfaf.py`).

### SFAF research model (public datasets)

| Test dataset | Accuracy | F1 | AUC | Features |
|---|---|---|---|---|
| CICIDS2017 | 0.9833 | 0.9584 | 0.9985 | 12 |
| UNSW-NB15  | 0.9268 | 0.9432 | 0.9844 | 12 |
| TON-IoT    | 0.9941 | 0.9961 | 0.9997 | 12 |

Generalisation gap: CICIDS-only baseline = 98.4% on CICIDS but 36.1% on unseen
UNSW-NB15; SFAF lifts UNSW to 92.7% (**+56.6 pp**). Bot-IoT and CIC-IoT-2023
alignment is wired in (`code/dataset_maps.py`) and runs once the datasets are
provided. Full numbers in `models/report_numbers.md`.

---

## Repository layout

```
src/
  flow_features.py     packet→flow→22-feature extractor (train == serve)
  train_live_model.py  train the edge model; export ONNX + booster + meta
  ids_daemon.py        the IDS/IPS: --pcap / --replay / --iface, --ips/--prevent
  ips_response.py      active response: block/rate-limit (nftables/iptables)
  dashboard.py         live web dashboard (stdlib http.server, reads alert feed)
  export_c.py          compile the model to a dependency-free C header (MCUs)
attacks/
  traffic_gen.py       scapy generators: benign family + 9 attack types
  build_corpus.py      synth traffic → labeled flow dataset
  README.md            synthetic pcaps + real-tool (nmap/hping3/…) equivalents
demo/
  run_demo.sh          one-command end-to-end demonstration
  validate.py          held-out validation on unseen scenarios
  results/             confusion matrices, reports
deploy/
  setup_pi.sh, iot-ids.service, requirements-pi.txt, README_PI.md
code/
  02_train_sfaf.py     headless 5-dataset SFAF reproduction (regenerates artifacts)
  dataset_maps.py      SFAF alignment for all 5 datasets (unit-tested)
  download_datasets.py dataset fetch + instructions
  *.ipynb              EDA / SFAF / edge-deployment notebooks
models/
  live_ids.onnx        deployable edge model (scaler baked in, ~90 KB)
  live_ids.h           dependency-free C model for microcontrollers
  live_meta.json       feature order, labels, categories, scaler, metrics
tests/
  smoke_test.py        21 checks over every module/script
```

---

## How detection works

Each bidirectional flow is summarised by **22 features** (`src/flow_features.py`):
protocol, duration, packet/byte counts and rates, packet-size and inter-arrival
statistics, TCP flag ratios, forward/backward asymmetry, the target **service
port**, plus **host-context** features (distinct destination ports and IPs per
source in a rolling window). Host-context is what separates a port scan (many
ports, one host) from a Mirai spread (one port, many hosts) from a flood
(one host+port, huge volume) — all indistinguishable at the single-flow level.
Verdicts are aggregated into per-`(source, type)` **incidents**, so a 500-port
scan is one alert.

The **IPS layer** (`--ips` dry-run, `--prevent` enforce) blocks high-confidence
sources via nftables/iptables with a confidence gate, allowlist, and auto-expiry
— it degrades safely to dry-run when it can't enforce.

> **Authorized-use only.** The attack generators and tool commands produce
> hostile traffic; confine them to hardware you own (your Pi + host).
