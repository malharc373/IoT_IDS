# IoT-IDS — Edge Intrusion Detection & Prevention for IoT Networks

A machine-learning IDS/IPS prototype for IoT networks with two deliberately
separate artifacts: a 22-feature live detector and an **SFAF (Semantic Feature
Alignment Framework)** cross-dataset research model.

Two halves test one idea — *whether a small, flow-based model can detect and
help stop network attacks on constrained edge hardware*:

1. **Live edge IDS/IPS** — a streaming sensor prototype. It sniffs traffic,
   aggregates packets into bidirectional flows, and classifies each flow with a
   91.8 KB ONNX model. The deployment platform is a **Raspberry Pi 2 Model B
   (32-bit)**, where inference, parsing, live attacks, benign false alarms,
   reboot recovery and a 14.4-hour soak were measured. It detects
   **9 attack types across 4 categories**, reports aggregated per-source
   incidents, and includes an experimental response module. Active blocking is
   deliberately disabled on real networks because the first live baseline
   produced unsafe false positives. The same model also compiles to a
   **dependency-free C header**; that header is the Pi 2's inference engine,
   and a minimal ESP32 sketch shows it compiling into microcontroller code
   (inference example only; see *Scope and status*).

2. **SFAF cross-dataset study** — eleven public IDS datasets (CICIDS2017,
   UNSW-NB15, TON-IoT, Bot-IoT, CIC-IoT-2023, CICDDoS2019, IoTID20, X-IIoTID,
   MQTT-IoT-IDS2020, WUSTL-IIoT, IoT-23) aligned into one 12-feature space to measure —
   not assert — how well flow behaviour transfers across labs/devices/tools. An
   earlier run's diagonal turned out to be resubstitution (train and test were
   the same rows); the protocol-correct rerun of 10 November 2026 measures
   held-out in-domain ROC-AUC **0.979** against cross-domain **0.497** (chance).
   See [`demo/results/CROSS_DATASET_FINDINGS.md`](demo/results/CROSS_DATASET_FINDINGS.md).

```
   ┌────────────────────────── live detection core ────────────────────────────┐
   │ packets → bidirectional flow table → 22-feature vector → model → verdict    │
   │                                      ↳ dry-run response; enforcement gated  │
   └────────────────────────────────────────────────────────────────────────────┘
       ▲ Mac / dev : synthetic labeled pcaps        (root-free demo)
       ▲ Pi  / live: scapy sniff on eth0 / wlan0    (systemd service; IPS dry-run)
       ▲ MCU       : models/live_ids.h              (inference only, ~49 KB const)
```

## Scope and status

What the project has shown, and what it has not.

### Raspberry Pi: the Pi 2 is the deployment platform

- **Final platform: Raspberry Pi 2 Model B on 32-bit Raspberry Pi OS
  (armv7l).** All hardware results in this repository come from this board.
  onnxruntime publishes no 32-bit ARM wheels, so two 32-bit runtimes were
  added for the **same trained model** (it was not retrained):
  1. **C backend (default on 32-bit):** the model's C export
     (`models/live_ids.h`) is compiled on the Pi and called through ctypes
     (`src/c_backend.py`). The installer selects it automatically on
     armv6l/armv7l.
  2. **onnxruntime 1.23.2 for armv7l**, cross-compiled from source under
     emulation ([`deploy/onnxruntime-armv7/`](deploy/onnxruntime-armv7/README.md)).
     The wheel is built on demand, not committed.
- **Same answers on both.** Against 64-bit onnxruntime, both 32-bit runtimes
  gave identical labels on 5,000 inputs (largest probability difference
  1.07e-6). On the Pi 2 the C backend is about 2× faster in batch; the armv7
  onnxruntime wheel is faster for a single flow. C stays the default because it
  also needs no custom wheel.
- **64-bit path (supported, not hardware-tested).** On a 64-bit Pi OS
  (aarch64) the installer uses onnxruntime's official wheels and the locked
  dependencies. That path runs on 64-bit hosts and in CI, but no 64-bit Pi was
  used in this project, so no hardware result is claimed for it.

### IPS (prevention): a tested mechanism, not a safe protection

Works:
- The response ladder (monitor → throttle → block) with a strike gate,
  allowlist, per-class thresholds and automatic expiry through nftables sets
  with kernel timeouts. It is covered by unit tests and was run once in enforce
  mode on the Pi 2: the attacking laptop was blocked about 5 s into a port scan,
  and the block lifted itself 21 s after the attack stopped.
- After that run, sources that cannot be a meaningful attacker (unspecified,
  loopback, multicast, link-local, broadcast) were excluded from enforcement,
  and firewall commands were resolved to absolute paths for systemd. These
  changes have unit tests but have not yet been re-run in enforce mode on
  hardware.

Does not work yet:
- **Safety on a real network.** In the enforce run, false alarms became
  actions against 12 innocent sources, including the router, and one innocent
  IPv6 device was blocked. Replaying those recorded actions through the new
  exclusion rule removes 7 of the 12 (including the block), but the router
  and four other IPv4 hosts would still be throttled. The root cause is the
  detector's false-alarm rate on ordinary LAN traffic (about 1.9 incidents per
  minute), which the response layer cannot fix by itself.
- **Protecting other devices.** The default `--ips-scope host` protects only
  the sensor. Inline bridge mode (`--ips-scope network`) is implemented but has
  never been tested.
- **Enforcement from the installed service.** The enforce test ran the daemon
  by hand as root, not through the systemd unit.

The installed service therefore runs IDS-only. `--ips` shows what the responder
*would* do without touching the firewall. `--prevent` belongs only on an
isolated test network.

### C export and the ESP32 inference example

The deliverable here is the **dependency-free C model**, not an ESP32
intrusion detector:

- The model exports to a C99 header (`models/live_ids.h`).
  `src/export_c.py --verify` checks its decisions against the XGBoost model.
  It is the default inference engine on the Raspberry Pi 2, so the C code has
  run on real ARM hardware.
- Footprint of an `-Os` build on the development host: about 49 KB of
  constant data (a 43 KB node table plus a 6 KB per-tree index), about 240 B
  of code, and no heap. The constant data would be the same size on an ESP32,
  well within the chip's flash and RAM.
- [`deploy/esp32_iot_ids/`](deploy/esp32_iot_ids/esp32_iot_ids.ino) is a
  minimal **inference example**: it calls the model on a fixed placeholder
  vector and prints the class over Serial. It is not a detector, has not been
  flashed to a board, and nothing is claimed about ESP32 accuracy or speed.

An ESP32 detector would additionally need the 22-feature flow table ported to
C and an inline capture position (for example as the Wi-Fi access point); both
are out of scope. See [`deploy/README_MCU.md`](deploy/README_MCU.md).

### Attack taxonomy (hierarchical)

| Category | Types |
|---|---|
| **recon** | portscan, xmas_scan (Xmas/NULL/FIN stealth scans) |
| **dos** | synflood, udpflood, icmpflood, mqtt_flood, slowloris |
| **botnet** | mirai (telnet/ssh propagation) |
| **bruteforce** | ssh_bruteforce |

Alerts read `category/type`, e.g. `⚠ ATTACK recon/portscan src=… 593 dst-ports`.

One command drives a running sensor with benign background traffic and every
attack class in turn, for a live demo — `sudo python attacks/live_demo.py`
(loopback by default; see [`attacks/README.md`](attacks/README.md)).

---

## Quickstart (dev machine, no root)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python attacks/build_corpus.py --scenarios 30   # synth traffic -> labeled flows
python src/train_live_model.py                  # train + export ONNX (+ booster)
python demo/preflight.py                         # verify deps, artifacts, contract, inference
bash demo/run_demo.sh                            # generate traffic -> detect -> validate
```

Detect + preview the experimental response policy (dry-run; no firewall changes):

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

Run every laptop-available capstone gate in one command (lint, types, tests,
demo, benchmark, and report generation):

```bash
bash demo/verify_capstone.sh
```

This intentionally does not simulate or replace the separate real-labelled-PCAP
or external-dataset acceptance gates. Raspberry Pi 2 evidence is tracked under
[`reports/pi2-acceptance-2026-09-30/`](reports/pi2-acceptance-2026-09-30/)
and [`reports/pi2-live-2026-09-30/`](reports/pi2-live-2026-09-30/).

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
explicit license decision before outside contributions or reuse. The scoped
recommendation and rightsholder checklist are in
[`docs/LICENSE_DECISION.md`](docs/LICENSE_DECISION.md).

## Configuration

Optional settings live in a git-ignored `.env` (copy the template):

```bash
cp .env.example .env                # e.g. IOTIDS_DATASETS_ROOT, IOTIDS_IFACE
```

No secrets belong in the repo. The Kaggle API token (for dataset downloads) goes
at `~/.kaggle/kaggle.json` — see `code/download_datasets.py`.

## Deploy on a Raspberry Pi (IDS; experimental response available)

Runtime needs only `onnxruntime + numpy + scapy` on a 64-bit Pi; on a 32-bit Pi
the installer uses the C backend instead of onnxruntime. Full walkthrough in
**[deploy/README_PI.md](deploy/README_PI.md)**.

```bash
sudo bash deploy/setup_pi.sh eth0        # venv + systemd service
sudo systemctl start iot-ids
# Experimental enforcement: isolated authorized tests only. The current model
# produced unsafe false positives on a home LAN; normal deployments stay IDS-only.
sudo .venv/bin/python src/ids_daemon.py --iface eth0 --prevent --allow 192.168.1.0/24
```

Launch attacks from another LAN host — see **[attacks/README.md](attacks/README.md)**.

For the live presentation, follow the one-page
**[deploy/DEMO_RUNBOOK.md](deploy/DEMO_RUNBOOK.md)**.

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
| Model size (ONNX / C const-data) | 91.8 KB / ~49 KB |
| Host ONNX inference (x86 reference host) | 9.7 µs/flow (p99 23.9 µs) |

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

### Raspberry Pi 2 live validation (controlled trial, 30 September 2026)

The deployment board is a Raspberry Pi 2 Model B running 32-bit Raspberry Pi
OS. The trial used ordinary home-LAN
traffic plus seven controlled attacks from devices owned by the operator. It is
real execution evidence, but seven attacks are not a statistically representative
real-world accuracy study.

| Measure | Pi 2 result |
|---|---|
| Model/runtime | Same 10-class model; C backend by default; armv7 ONNX Runtime 1.23.2 also verified |
| Controlled attacks | 7/7 raised an incident; 5/7 had the correct label at confidence ≥ 0.9 |
| Benign baseline | 19 false-alarm incidents in 10 min (1.9/min); 13/19 survived a ≥ 0.9 confidence gate |
| Single-flow C daemon path | 535 µs (p99 799 µs) |
| Batched C inference | 17,506 flows/s at batch 1024 |
| Packet parsing / end to end | 6,758 packets/s / 1,316 flows/s |
| Flood capture | About 12–14% of offered ~2k pps loopback flood traffic processed |
| Runtime footprint | 39–70 MB RSS; 50–56 °C; no throttling observed |
| Reboot recovery | SSH in 80 s; fresh sensor heartbeat in 99 s; `NRestarts=0` |
| Soak | 14.4 h partial (stopped by hand): 0 restarts, 0 drops, RSS 53.5–63.0 MB |

The UDP flood was detected as an attack but labelled `xmas_scan`; the ICMP
flood was weak (0.52 confidence), and SSH brute force was late and represented
only 2 of roughly 40 connections. Confidence alone did not remove benign false
alarms. An enforcement experiment blocked the attacker but also affected
innocent devices and the router, so automatic blocking is **not safe to enable**
with the current model. Full commands, raw evidence and limitations:
[`reports/pi2-live-2026-09-30/README.md`](reports/pi2-live-2026-09-30/README.md).

### Cross-dataset generalization (corrected rerun, 10 November 2026)

Eleven datasets align to the 12-feature space. The corrected protocol trains on
a fixed 80% split of each source, tests the diagonal on its untouched 20%, and
tests off-diagonal cells on independent datasets (`code/cross_dataset_eval.py`). Reported with
ROC-AUC and MCC rather than F1, because each test set is ~50/50 balanced and a
classifier that answers "attack" to everything scores F1 = 0.667.

Rerun with `python code/run_sfaf_rerun.py` (50,000 flows per dataset, seed 42;
gate passed, 121/121 cells):

| | In-domain (held-out 20%, n = 11) | Cross-domain (n = 110) | Gap |
|---|---|---|---|
| ROC-AUC, mean | **0.979** | **0.497** | 0.482 |
| MCC, mean | 0.898 | −0.014 | 0.911 |

- 50 of 110 cross-domain cells are *below* 0.5 (down to 0.016): rankings
  invert between labs, they do not just get noisier.
- Pooled training on five datasets transfers to 2 of 6 unseen ones
  (MQTT-IoT-IDS2020 0.979, CICDDoS2019 0.930) and fails on the rest (0.347–0.594).
- Ten labelled target flows fix the threshold on those 2 datasets only.
- No MCU-exportable feature transform beats plain standard scaling in
  leave-one-dataset-out training (mean ROC-AUC 0.612).
- Bot-IoT has only 11 benign flows in its held-out split, so its numbers are
  indicative; excluding it gives 0.977 vs 0.489.

The earlier run (diagonal 0.995, off-diagonal 0.509) is kept under
`legacy/resubstitution-results/` as withdrawn historical evidence. Full
results, run manifest and gate checklist:
[`demo/results/CROSS_DATASET_FINDINGS.md`](demo/results/CROSS_DATASET_FINDINGS.md).

### System benchmark (x86 reference host; `python demo/benchmark.py`)

Intel Xeon @ 2.30 GHz cloud VM, Linux, Python 3.10, 10 November 2026. Every
section completed; the accuracy section reproduces the table above exactly.

| | |
|---|---|
| ONNX inference | 9.7 µs/flow single (p99 23.9), **408,321 flows/s** at batch 1024 |
| Native C model | **1.597 µs/flow**, 626,009 flows/s, ~130 B stack, zero deps |
| Feature extraction | 129,003 packets/s; 33,779 flows/s |
| End to end | 35,575 flows/s |
| Daemon memory | **57.8 MB** (onnxruntime + numpy) |
| Model size | 91.8 KB ONNX / ~49 KB C const |

The host benchmark is not a Pi projection; measured Pi 2 results are reported
separately above. A full 24-hour soak is still open (14.4 h was recorded).
Full host report: [`demo/results/BENCHMARK.md`](demo/results/BENCHMARK.md).

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

### Reproducible release evidence

Build the same deterministic capstone bundle produced by the release workflow:

```bash
python tools/build_release.py
(cd dist && shasum -a 256 -c SHA256SUMS)
```

Manual and tagged release workflows upload the bundle and checksum manifest and
issue a GitHub build-provenance attestation. After downloading the artifact:

```bash
gh attestation verify dist/iot-ids-capstone.tar.gz \
  --repo malharc373/IoT_IDS
```

Tagged runs additionally create the matching GitHub release. The workflow pins
every third-party Action to a full commit SHA.

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
  live_demo.py         one-command live demo: benign + every attack class
                       against a running sensor (loopback-safe by default)
  slowloris.py         bundled held-open-HTTP attack used by the live demo
  build_corpus.py      synth traffic → labeled flow dataset (attacks mixed
                       with benign background; scenario provenance recorded)
  README.md            synthetic pcaps + real-tool (nmap/hping3/…) equivalents
demo/
  run_demo.sh          one-command end-to-end demonstration
  preflight.py         demo/deployment readiness and inference check
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
tools/
  build_release.py     deterministic evidence bundle + SHA-256 manifest
.env.example           optional configuration template (copy to .env)
```

---

## How detection works

Each bidirectional flow is summarised by **22 features** (`src/flow_features.py`
— IPv4 and IPv6, VLAN/QinQ-aware, TCP teardown-aware, reads both pcap and
pcapng, including Ethernet, Linux cooked SLL/SLL2, and radiotap data frames):
protocol, duration, packet/byte counts and rates, packet-size and inter-arrival
statistics, TCP flag ratios, forward/backward asymmetry, the target **service
port**, plus **host-context** features (distinct destination ports and IPs per
source in a rolling window). Host-context is what separates a port scan (many
ports, one host) from a Mirai spread (one port, many hosts) from a flood
(one host+port, huge volume) — all indistinguishable at the single-flow level.
Verdicts are aggregated into per-`(source, type)` **incidents**, so a 500-port
scan is one alert.

The **IPS layer** (`--ips` dry-run, `--prevent` enforce; experimental, see
*Scope and status*) responds on a ladder —
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
- `--class-min-conf mirai=0.8` overrides the alert gate for one class;
  `--ips-class-min-conf mirai=0.98` independently raises its enforcement gate.
  Both options are repeatable, so policy can reflect different error costs.
- `--abstain-conf 0.6` labels lower-score predictions `unknown` instead of
  forcing a known class. This max-score heuristic is experimental: without
  deployment-like calibration it is an ambiguity signal, not validated
  unknown-attack detection. Unknown verdicts are monitor-only for IPS unless
  the operator explicitly supplies `--ips-class-min-conf unknown=...`, and IPS
  is still dry-run unless `--prevent` is supplied.

Example, with a conservative Mirai enforcement policy and visible ambiguous
flows:

```bash
python src/ids_daemon.py --replay data/pcaps/demo_mixed.pcap --ips \
  --abstain-conf 0.6 --class-min-conf unknown=0.2 \
  --ips-class-min-conf mirai=0.98
```

> **Authorized-use only.** The attack generators and tool commands produce
> hostile traffic; confine them to hardware you own (your Pi + host).
