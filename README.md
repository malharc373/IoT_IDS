# IoT-IDS — Edge Intrusion Detection & Prevention for IoT Networks

A machine-learning IDS/IPS for IoT networks, built around **SFAF (Semantic
Feature Alignment Framework)** and deployable as a real-time sensor on hardware
from a Raspberry Pi down to an ESP32-class microcontroller.

Two halves share one idea — *a small, flow-based model can detect and stop
network attacks in real time on cheap hardware*:

1. **Live edge IDS/IPS** — a self-contained real-time sensor. It sniffs traffic,
   aggregates packets into bidirectional flows, and classifies each flow with a
   ~96 KB ONNX model in **microseconds per flow**. It detects
   **9 attack types across 4 categories**, reports aggregated per-source
   incidents, and can **actively block** offenders (IPS mode). The same model
   also compiles to a **dependency-free C header** for microcontrollers.

2. **SFAF cross-dataset study** — eleven public IDS datasets (CICIDS2017,
   UNSW-NB15, TON-IoT, Bot-IoT, CIC-IoT-2023, CICDDoS2019, IoTID20, X-IIoTID,
   MQTT-IoT-IDS2020, WUSTL-IIoT, IoT-23) aligned into one 12-feature space to measure —
   not assert — how well flow behaviour transfers across labs/devices/tools. The
   last audited off-domain run found severe domain shift, but its diagonal was
   later found to be resubstitution (train and test were the same rows). The
   corrected held-out protocol is implemented and its exact headline numbers
   are intentionally withheld until the external datasets are remounted and the
   full study is rerun. See
   [`demo/results/CROSS_DATASET_FINDINGS.md`](demo/results/CROSS_DATASET_FINDINGS.md).

```
   ┌────────────────────────── live detection core ────────────────────────────┐
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
python src/dashboard.py                                              # http://127.0.0.1:8080
```

It binds loopback by default. The page exposes attacking hosts, blocked hosts
and the segment's addressing, so serving it to a network needs a token
(`--token generate`) or an explicit `--insecure`; over an untrusted network
prefer `ssh -L 8080:127.0.0.1:8080 pi@<host>`.

Compile the model for a microcontroller:

```bash
python src/export_c.py --verify     # writes models/live_ids.h, checks parity
```

Benchmark the whole system (accuracy, latency, throughput, footprint):

```bash
python demo/benchmark.py            # writes demo/results/BENCHMARK.md
```

## Configuration

Optional settings live in a git-ignored `.env` (copy the template):

```bash
cp .env.example .env                # e.g. IOTIDS_DATASETS_ROOT, IOTIDS_IFACE
```

No secrets belong in the repo. The Kaggle API token (for dataset downloads) goes
at `~/.kaggle/kaggle.json` — see `code/download_datasets.py`.

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
| Multiclass accuracy (10 classes) | **~100%** |
| Attack detection rate (recall) | 100% |
| Benign false-positive rate | **0.0%** |
| Model size (ONNX / C const-data) | ~96 KB / ~46 KB |
| Inference | ~6 µs/flow |

> **Read this number as a property of the generators, not of the detector.**
> Three things were checked to find out what the ~100% actually means, and all
> three came back clean:
>
> - **Split by scenario, not by row.** Flows from one scenario share an attacker
>   and identical host-context values, so a random row split leaks. Grouping on
>   scenario (test attackers never seen in training) moves the score by 0.0001.
> - **Benign background mixed into every attack scenario**, so the three
>   host-context features are measured against a realistic backdrop instead of
>   an empty capture. No change.
> - **`dst_port` ablation on every training run.** Removing the target port
>   entirely costs 0.0000 accuracy and no class loses >0.02 F1 — the model is
>   not a port lookup table.
>
> What remains is the honest explanation: **the synthetic corpus is trivially
> separable**, along several redundant axes at once. That is why no correction
> moves the number. It cannot be quoted as real-world accuracy — for that, see
> the cross-dataset study below, and the open work to validate on real labelled
> IoT-23 captures.

### Cross-dataset generalization (the honest real-world number)

Eleven datasets align to the 12-feature space. The corrected protocol trains on
a fixed 80% split of each source, tests the diagonal on its untouched 20%, and
tests off-diagonal cells on independent datasets (`code/cross_dataset_eval.py`). Reported with
ROC-AUC and MCC rather than F1, because each test set is ~50/50 balanced and a
classifier that answers "attack" to everything scores F1 = 0.667:

The previous run's mean off-diagonal ROC-AUC was 0.509 over 110 ordered pairs,
which remains evidence of severe domain shift. Its 0.995 diagonal, however, was
measured on training rows, so the quoted 0.487 gap and every downstream exact
claim have been withdrawn. Row-retention, Bot-IoT sampling, IoT-23 cache
invalidation, and small-budget calibration were corrected at the same time.

The external dataset mount is currently unavailable, so publishing replacement
numbers would be fabrication. The affected artifacts are quarantined under
`legacy/resubstitution-results/`; the live status and reproduction gate are in
[`demo/results/CROSS_DATASET_FINDINGS.md`](demo/results/CROSS_DATASET_FINDINGS.md).

### System benchmark (Apple M4; `python demo/benchmark.py`)

| | |
|---|---|
| ONNX inference | 10 µs/flow single, **381k flows/s** batched |
| Native C model | **1.15 µs/flow**, ~130 B RAM, zero deps |
| Feature extraction | ~248k packets/s |
| Daemon memory | **56 MB** (onnxruntime + numpy) |
| Model size | 90 KB ONNX / 42 KB C const |

Real-time on a Pi 4 with headroom — sniffing/aggregation, not inference, is the
limit. Full report: [`demo/results/BENCHMARK.md`](demo/results/BENCHMARK.md).

### Reproducing the SFAF datasets

The datasets are large and gated; fetch them with a Kaggle token (auto-detected)
and auth-free direct URLs:

```bash
python code/download_datasets.py --direct   # Kaggle + IoT-23/WUSTL, or prints links
python code/multidataset.py                 # verify alignment of all present datasets
python code/cross_dataset_eval.py           # run the generalization study
python code/threshold_transfer.py           # how cheap is target-domain calibration
```

IoT-23 ships as `iot_23_datasets_small.tar.gz`; extract it under
`Datasets/IoT23/` before use. Its sampled frame is cached beside the data, so
only the first run pays the ~27 GB parse.

---

## Repository layout

```
src/
  flow_features.py     packet→flow→22-feature extractor (train == serve)
  train_live_model.py  train the edge model (scenario-level split, dst_port
                       ablation); export ONNX + booster + meta, verified
  ids_daemon.py        the IDS/IPS: --pcap / --replay / --iface, --ips/--prevent
  ips_response.py      active response ladder: monitor/throttle/block, scoped
                       to INPUT (host) or INPUT+FORWARD (inline network)
  dashboard.py         live web dashboard (stdlib http.server, reads alert feed)
  export_c.py          compile the model to a dependency-free C header (MCUs)
attacks/
  traffic_gen.py       scapy generators: benign family + 9 attack types
  build_corpus.py      synth traffic → labeled flow dataset (attacks mixed
                       with benign background; scenario provenance recorded)
  README.md            synthetic pcaps + real-tool (nmap/hping3/…) equivalents
demo/
  run_demo.sh          one-command end-to-end demonstration
  validate.py          held-out validation on unseen scenarios
  benchmark.py         accuracy / latency / throughput / footprint benchmark
  results/             confusion matrices, cross-dataset study, benchmark reports
deploy/
  setup_pi.sh          Pi installer: venv + iot-ids + iot-ids-dashboard services
  iot-ids*.service, requirements-pi.txt, README_PI.md, README_MCU.md
code/
  multidataset.py      load + SFAF-align 11 flow datasets; the single source of
                       truth for feature alignment (units, coverage, NaN policy)
  cross_dataset_eval.py train-on-one/test-on-others generalization matrix
  transfer_experiment.py deployable feature transforms to close the transfer gap
  threshold_transfer.py how many labelled target flows fix the operating point
  02_train_sfaf.py     headless SFAF reproduction (regenerates thesis artifacts)
  download_datasets.py dataset fetch (Kaggle + auth-free direct URLs)
legacy/                superseded work, kept for the record — see
                       legacy/README.md. Nothing here is current: the
                       pre-2026-08-19 result artifacts (invalidated by the
                       feature-alignment and metric findings) and the three
                       original notebooks.
models/
  live_ids.onnx        deployable edge model (~96 KB, raw features — trees are
                       scale-invariant, so there is no scaler to drift)
  live_ids.h           dependency-free C model for microcontrollers
  live_meta.json       purpose, feature contract, labels, metrics, evidence scope
  README.md            artifact manifest + research/runtime separation
tests/
  smoke_test.py        regression checks over every module/script
.env.example           optional configuration template (copy to .env)
```

---

## How detection works

Each bidirectional flow is summarised by **22 features** (`src/flow_features.py`
— IPv4 and IPv6, VLAN/QinQ-aware, TCP teardown-aware, reads both pcap and
pcapng):
protocol, duration, packet/byte counts and rates, packet-size and inter-arrival
statistics, TCP flag ratios, forward/backward asymmetry, the target **service
port**, plus **host-context** features (distinct destination ports and IPs per
source in a rolling window). Host-context is what separates a port scan (many
ports, one host) from a Mirai spread (one port, many hosts) from a flood
(one host+port, huge volume) — all indistinguishable at the single-flow level.
Verdicts are aggregated into per-`(source, type)` **incidents**, so a 500-port
scan is one alert.

The **IPS layer** (`--ips` dry-run, `--prevent` enforce) responds on a ladder —
*monitor → throttle → block* — via nftables/iptables, with an allowlist and
auto-expiry, degrading safely to dry-run when it can't enforce.

Two flags matter for correctness:

- `--ips-scope host` (default) installs INPUT rules and protects **only the
  sensor**. A passive sensor on a mirror port cannot stop an attack on another
  device. Use `--ips-scope network` (INPUT + FORWARD) when the Pi is inline.
- `--ips-strikes 3` requires that many corroborating incidents within
  `--ips-strike-window` seconds before blocking, because the model's softmax
  confidence is **not calibrated** — a reported 0.99 is not a 99% guarantee.
  Below the strike count the source is rate-limited rather than blackholed.

> **Authorized-use only.** The attack generators and tool commands produce
> hostile traffic; confine them to hardware you own (your Pi + host).
