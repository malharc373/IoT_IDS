# IoT-IDS — Edge Intrusion Detection & Prevention for IoT Networks

A machine-learning IDS/IPS prototype for IoT networks with two deliberately
separate artifacts: a 22-feature live detector and an **SFAF (Semantic Feature
Alignment Framework)** cross-dataset research model.

Two halves test one idea — *whether a small, flow-based model can detect and
help stop network attacks on constrained edge hardware*:

1. **Live edge IDS/IPS** — a streaming sensor prototype. It sniffs traffic,
   aggregates packets into bidirectional flows, and classifies each flow with a
   91.8 KB ONNX model. On the audited Apple M4 host, single-flow inference took
   8.1 microseconds in the current benchmark snapshot; Raspberry Pi and MCU end-to-end performance remain
   unvalidated. It detects
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
       ▲ MCU       : models/live_ids.h              (no runtime, ~43 KB const)
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
pip install -r requirements-dev.txt

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

Dependency inputs produce exact transitive Python 3.10 locks with `pip-compile`.
Use `requirements-dev.txt` on macOS/default development hosts and
`requirements-dev-linux-x86_64.txt` on Ubuntu x86_64; their runtime-only
counterparts omit test/lint tools. The separate Linux lock captures XGBoost's
conditional NCCL dependency without making macOS install CUDA packages. The Pi
uses `deploy/requirements-pi.txt`. Upgrade intentionally through the `.in`
files; CI regenerates both host families and rejects stale locks.

### Current report and presentation

The April 2026 report and image-only deck are preserved under
`legacy/stale-publication-artifacts/` because they contain withdrawn SFAF
metrics and an unvalidated Raspberry Pi conclusion. Use the corrected,
evidence-scoped replacements:

- editable report source: [`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md)
- generated PDF: [`output/pdf/IOT_IDS_Corrected_Technical_Report.pdf`](output/pdf/IOT_IDS_Corrected_Technical_Report.pdf)
- editable deck: [`output/presentation/IOT_IDS_Corrected_Project_Review.pptx`](output/presentation/IOT_IDS_Corrected_Project_Review.pptx)

Regenerate the report with the bundled ReportLab-capable Python runtime (or any
environment with ReportLab) using `python reports/build_report.py`.

### Governance and responsible use

- [`SECURITY.md`](SECURITY.md) — private vulnerability reporting and operational cautions
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — verification and evidence rules
- [`models/README.md`](models/README.md) — live/research model cards and artifact contract
- [`docs/DATA_CARD.md`](docs/DATA_CARD.md) — provenance, limitations, privacy and missing acceptance data
- [`docs/REAL_TRAFFIC_ACCEPTANCE.md`](docs/REAL_TRAFFIC_ACCEPTANCE.md) — real-capture evidence gate
- [`deploy/PI_ACCEPTANCE.md`](deploy/PI_ACCEPTANCE.md) — target identity, benchmark and soak protocol

No open-source license has been selected yet. The absence of a license means no
permission to copy, modify, or redistribute is granted; the owner must make an
explicit license decision before outside contributions or reuse.

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
| Multiclass accuracy (10 classes) | **99.65%** |
| Macro F1 | 0.9961 |
| Attack detection rate (recall) | 100.00% |
| Benign false-positive rate | **0.0%** |
| Mirai recall | 94.2% |
| Model size (ONNX / C const-data) | 91.8 KB / ~43 KB |
| Host ONNX inference | 8.1 µs/flow (p99 11.2 µs) |

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
> the cross-dataset study below, and the open work to validate on a real
> labelled packet capture compatible with the 22-feature extractor.

### Cross-dataset generalization (protocol corrected; rerun pending)

Eleven datasets align to the 12-feature space. The corrected protocol trains on
a fixed 80% split of each source, tests the diagonal on its untouched 20%, and
tests off-diagonal cells on independent datasets (`code/cross_dataset_eval.py`). Reported with
ROC-AUC and MCC rather than F1, because each test set is ~50/50 balanced and a
classifier that answers "attack" to everything scores F1 = 0.667:

The historical run's mean off-diagonal ROC-AUC was 0.509 over 110 ordered
pairs, but its 0.995 diagonal was measured on training rows. The quoted gap and
every downstream exact claim are therefore withdrawn; 0.509 is retained only
as historical evidence that motivated the corrected rerun. Row-retention,
Bot-IoT sampling, IoT-23 cache invalidation, and small-budget calibration were
corrected at the same time.

The external dataset mount is currently unavailable, so publishing replacement
numbers would be fabrication. The affected artifacts are quarantined under
`legacy/resubstitution-results/`; the live status and reproduction gate are in
[`demo/results/CROSS_DATASET_FINDINGS.md`](demo/results/CROSS_DATASET_FINDINGS.md).

### System benchmark (Apple M4; `python demo/benchmark.py`)

| | |
|---|---|
| ONNX inference | 8.1 µs/flow single (p99 11.2), **403,058 flows/s** at batch 512 |
| Native C model | **1.136 µs/flow**, 880,460 flows/s, ~130 B stack, zero deps |
| Feature extraction | 215,601 packets/s; 56,455 flows/s |
| End to end | 52,249 flows/s |
| Daemon memory | **55.1 MB** (onnxruntime + numpy) |
| Model size | 91.8 KB ONNX / ~43 KB C const |

The host benchmark is encouraging, but Raspberry Pi throughput remains an
unvalidated projection until the hardware acceptance run is captured. Full
report: [`demo/results/BENCHMARK.md`](demo/results/BENCHMARK.md).

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

Downloads use normal TLS verification and archives are extracted with path,
link, and special-file checks. Direct-source archives are downloaded but not
extracted unless a trusted digest is supplied, for example
`--sha256 IoT23=<64-hex-digest>`, or the risk is explicitly accepted with
`--allow-unverified`. The script prints the observed SHA-256 for independent
verification. Kaggle archives are also routed through the safe extractor.

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
  live_ids.onnx        deployable edge model (91.8 KB, raw features — trees are
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
