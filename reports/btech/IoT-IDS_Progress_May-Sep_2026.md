# IoT-IDS: Progress report, May – Sep 2026

**Project:** IoT-Based Intrusion Detection System Using Machine Learning
**Team:** Mahi Patel (612303107), Malhar Falke (612303108), Yugandhar Pise (612303140), Vaibhav Tayade (612303184)
**Guide:** Prof. S K Gaikwad, Dept. of Computer Engineering, COEP Technological University
**Repo:** github.com/malharc373/IoT_IDS
**Status as of:** 30 Sep 2026, the day before the mid-sem evaluation

> **For whoever writes the report from this file:** Section 7 lists what we can and can't claim. Please read it before using any number. The Raspberry Pi deployment is **in progress** (Section 6). Don't invent Pi results. If those slots are still blank, write "not yet measured".

---

## 1. Where we started (end of last semester, April 2026)

- EDA and baseline models (Random Forest, XGBoost + SMOTE) on public IDS datasets.
- **SFAF (Semantic Feature Alignment Framework):** maps different datasets (CICIDS2017, UNSW-NB15, TON-IoT) into one shared 12-feature space so a single XGBoost model can train across them.
- An ONNX edge model plus a laptop latency benchmark, and the Semester 6 report.

The main question for this period: **turn that into a real, working IDS that runs on cheap edge hardware, and check whether the research claims hold up.**

### Timeline (May – Sep 2026)

| Period | What happened |
|---|---|
| 16 Apr | Semester 6 report and presentation submitted |
| May – Jul | _(fill in: no commits in the repo for this period. Add any offline work, e.g. literature review, dataset collection, internship, or say "semester break")_ |
| 3 Aug | Live edge IDS built: flow feature extractor, 10-class model, real-time daemon, attack generators, Pi installer, IPS, MCU export, web dashboard |
| 4 Aug | Expanded to 11 public datasets, cross-dataset generalisation study, full system benchmark |
| 19–22 Aug | Two full code/research reviews. 26 findings addressed. Withdrew the flawed earlier numbers. Corrected technical report (22 Aug) |
| 31 Aug | Fixes merged to the review branch. CI, CodeQL and Dependabot set up |
| 5 Sep | Final laptop-side hardening: per-class thresholds, abstention, ESP32 sketch, type checking, fresh benchmark |
| 30 Sep | Raspberry Pi deployment in progress (see Section 6) |

---

## 2. Work done (May – Sep 2026)

### 2.1 Live edge IDS (new, main deliverable)
- **Real-time sensor** that reads raw network packets, groups them into flows, and classifies each flow.
  - Three input modes: offline pcap file, timed replay, and live network interface.
- **One feature extractor** (`src/flow_features.py`) turns packets into a **22-feature flow vector**.
  - Features: protocol, duration, packet/byte counts and rates, packet-length and inter-arrival stats, TCP-flag ratios, direction ratios, host-context counts, destination port.
  - Training and live detection use the exact same code, so there's no train/serve mismatch.
  - The feature set is versioned (contract v2), and the daemon refuses a model built for a different feature layout.
- **Packet parser:** Ethernet, VLAN/QinQ, IPv4, IPv6, TCP, UDP, ICMP, pcap and pcapng.
  - Handles IP fragments safely.
  - Rejects capture formats it doesn't support instead of misreading them.
- **Model:** 10-class XGBoost exported to **ONNX** (`models/live_ids.onnx`, ~92 KB).
  - Classes: benign, portscan, SYN flood, ICMP flood, UDP flood, SSH brute force, slowloris, Mirai, Xmas scan, MQTT flood.
  - The classes are grouped into categories: recon / DoS / botnet / brute force.
- **Incident aggregation:** flow verdicts are merged per attacker and attack type, so the operator sees one incident instead of thousands of per-flow lines. Memory use stays bounded.

### 2.2 Attack simulation and training data
- Traffic generators (`attacks/`) create each attack plus realistic benign background traffic.
  - They also create "hard benign" cases, such as bursts and multi-host traffic, that look like attacks but aren't.
- Train/test split is **by scenario**, so no scenario appears in both train and test.
  - 280 train / 70 test scenarios, ~44.8k training flows, ~12.2k held-out flows.

### 2.3 Intrusion *Prevention* (IPS)
- `src/ips_response.py` adds a response ladder: **monitor → throttle → block**, using nftables or iptables. It handles IPv4 and IPv6.
- Safe by default:
  - dry-run mode
  - allowlists
  - confidence thresholds, which can be set per attack class
  - an optional "unknown" abstention for ambiguous predictions
- Works either on the Pi itself ("host" mode) or as an inline bridge protecting the whole network ("network" mode).

### 2.4 Web dashboard
- Live operator dashboard showing incidents and alerts (`src/dashboard.py`). It runs as a systemd service next to the sensor.
- Secured:
  - It only listens locally by default.
  - Network access requires a token, which is stored in a mode-600 file.
  - Logs rotate.

### 2.5 SIEM integration
- Incidents can be sent to a SIEM (Splunk, QRadar, Sentinel, Elastic) over syslog in **CEF** or JSON format.
- A SIEM that is down doesn't crash the sensor.

### 2.6 Microcontroller export
- The model is exported to **dependency-free C99** (`models/live_ids.h`), so it can run on an MCU.
- About 130 bytes of stack and no heap. Its output was checked against the original model.
- Includes a demo **ESP32** sketch.

### 2.7 Raspberry Pi deployment tooling
- One-command installer: `deploy/setup_pi.sh` (Python venv, systemd services for sensor and dashboard, preflight check).
- Runbook: `deploy/README_PI.md`, covering passive vs inline placement.
- Acceptance checklist: `deploy/PI_ACCEPTANCE.md`, covering the benchmark, packet drops, memory, temperature, and a 24-hour soak.

### 2.8 Research track: cross-dataset study (SFAF, continued)
- Expanded to **11 public datasets**, including IoT-23, Bot-IoT, WUSTL-IIoT, CICDDoS2019, MQTT and others, each with its own loader.
- **Found and fixed an IoT-23 bug:** the dataset was being read as 100% benign because its label columns are space-separated. The loader was also rewritten to handle the 27 GB corpus in bounded memory.
- Added a **threshold-transfer study**: how many labelled target-network flows fix a model moved to a new network?
  - The earlier claim that roughly ten labelled flows were sufficient was withdrawn because single-class calibration samples were skipped.
  - The corrected script samples unconditionally and refuses to report "wins" that are really just the trivial "call everything an attack" classifier, but its exact results remain pending a full rerun.
- **Corrected the evaluation protocol:** proper 80/20 held-out splits per dataset, missing values tracked instead of silently dropped, memory-bounded random sampling, and versioned caches.

### 2.9 Two full code and research reviews, plus fixes
- We ran two review passes and addressed **26 findings (F01–F26)**. Most were fixed. A few suspected problems were measured and turned out not to be real, and one was mitigated. Highlights:
  - SFAF feature mappings were semantically wrong, and the cross-domain F1 metric counted degenerate classifiers.
  - The edge ONNX model was exported with the wrong scaler.
  - Live mode had a thread data race and re-scored the whole flow table on every tick.
  - The IPS rate limiting was documented but not actually implemented.
  - The dashboard was open to the network with no authentication.
  - Flow direction semantics were wrong.
  - Dataset archive extraction was unsafe (path traversal). It now fails closed.
- **Most important research correction:** in last semester's in-domain results, the model was **tested on the same rows it was trained on** (resubstitution). Those numbers, and the "generalisation gain" built on them, have been **withdrawn**. They're kept under `legacy/` with a retraction notice.

### 2.10 Engineering quality
- **67 automated tests**: parsing, flow state, thread safety, IPS rules, dashboard auth, dataset safety, model export parity, end-to-end demo.
- **GitHub Actions CI** runs ruff lint, the tests, ONNX checks, C compile, the demo, and the benchmark.
- Also added:
  - mypy type checking on the live core
  - locked dependencies (dev and Pi)
  - Dependabot and CodeQL
  - a security policy, data card and model card
  - a corrected technical report (Aug 22 edition, PDF in `output/pdf/`)

---

## 3. Key numbers, with what each one actually proves

| Metric | Value | Evidence scope |
|---|---|---|
| Held-out accuracy, 10 classes (unseen-seed synthetic traffic) | 99.65% (macro-F1 0.996) | **Synthetic traffic only** |
| Attack detection rate / benign false positives | 100% / 0% | Synthetic only |
| Hardest class | Mirai, 94.2% recall | Synthetic only |
| Accuracy with destination port removed | Unchanged | Shows the model isn't just a port lookup |
| ONNX model size | 91.8 KB | Artifact |
| C header for MCU | 102.7 KB (~43 KB const data), ~130 B stack, 0 heap | Artifact |
| Single-flow ONNX latency | 10.5 µs (p99 16.2 µs) | Apple Silicon laptop |
| Peak ONNX throughput | ~407k flows/s (batch 1024) | Laptop |
| Native C inference | 1.16 µs/flow | Laptop |
| Packet parse + flow aggregation | ~214k packets/s | Laptop |
| End-to-end (pcap → verdicts) | ~52.7k flows/s | Laptop |
| Daemon memory (RSS) | 55.4 MB | Laptop |
| Tests | 67 | CI / local |
| Raspberry Pi performance | **Not yet measured** | See Section 6 |
| Cross-dataset (SFAF) exact metrics | **Withdrawn, rerun pending** | See Section 7 |

Source: `demo/results/BENCHMARK.md` (5 Sep 2026 run).

---

## 4. How to demo it

```bash
bash demo/run_demo.sh          # end-to-end: generate traffic → detect → report
python demo/benchmark.py       # full benchmark
python src/dashboard.py        # dashboard
```

---

## 5. Architecture (one line)

`packets (pcap / replay / live) → parser → flow table → 22 features → ONNX model → per-source incidents → alert log + dashboard + SIEM + optional IPS block/throttle`

The research pipeline is separate: `11 public datasets → loaders → 12-feature SFAF alignment → cross-dataset evaluation`.

---

## 6. Deployment status (Raspberry Pi): IN PROGRESS

The software, installer and acceptance checklist are ready. The on-device run was being attempted on 30 Sep, the day before the evaluation. Fill in only what was actually measured:

| Item | Result |
|---|---|
| Pi model / OS | _(fill in)_ |
| Installer (`setup_pi.sh`) ran cleanly | _(yes / no)_ |
| Sensor + dashboard services running | _(yes / no)_ |
| Live attack demo detected on Pi | _(which attacks)_ |
| Pi benchmark: latency / throughput | _(from `demo/benchmark.py` on the Pi)_ |
| Memory / temperature / packet drops | _(fill in)_ |
| 24-hour soak | _(not done / done)_ |

**If a row is blank, write "not yet measured". Don't estimate Pi numbers by scaling the laptop numbers.**

---

## 7. Rules for writing the report (what we can and can't claim)

1. **Synthetic ≈100% accuracy is not real-world accuracy.** It shows the pipeline works and the generated attacks are separable. Always say "on synthetic held-out traffic".
2. **Laptop speed is not Pi speed.** Label every speed and memory number "measured on Apple Silicon laptop".
3. **Don't cite last semester's SFAF headline numbers**, i.e. the in-domain accuracy and the "generalisation gain" figure. They were withdrawn because of the train/test overlap described in 2.9. The only safe qualitative statement: *earlier cross-dataset tests showed severe domain shift; a corrected rerun is pending.*
4. **Don't cite a labelled-flow budget for threshold transfer.** The earlier approximately-ten-flow claim was produced by conditional sampling and is withdrawn; the corrected experiment has not been rerun.
5. **Present the correction as a result, not an embarrassment.** We found an evaluation flaw in our own work, withdrew the numbers, fixed the protocol and added tests so it can't come back.
6. IPS blocking, abstention and per-class thresholds are implemented and tested, but their operating values **haven't been validated on real deployment traffic**.

---

## 8. Remaining work: under a month to the final showcase (late Oct 2026)

The final goal is to **show the complete system running live**: attacks launched from a laptop get detected on the Pi and show up on the dashboard, with optional blocking.

| Week | Focus | Done when |
|---|---|---|
| 1 (1–7 Oct) | Pi deployment working end to end, then a live demo: attack laptop → Pi sensor → dashboard → IPS block | The demo runs reliably from a fresh boot |
| 2 (8–14 Oct) | Pi benchmark + soak. Real labelled traffic (IoT-23 pcaps) through the live extractor | Section 6 filled in. Per-class results on real traffic |
| 3 (15–21 Oct) | Corrected 11-dataset cross-dataset rerun. Merge all branches to `main` | New research numbers with no train/test overlap. One clean `main` |
| 4 (22–31 Oct) | Final report, slides, recorded backup demo video, rehearsal | Report submitted. Demo rehearsed |

If time runs short, cut from the bottom. The live Pi demo and honest numbers matter most. Drift/adversarial evaluation and licensing (below) are post-showcase.

| Priority | Task | Done when |
|---|---|---|
| P0 | Raspberry Pi deployment + benchmark + 24h soak | Section 6 filled with real numbers |
| P0 | Run real labelled IoT traffic (e.g. IoT-23 pcaps) through the live extractor | Per-class results on real traffic |
| P0 | Rerun the corrected cross-dataset study on all 11 datasets | New results with no train/test overlap |
| P1 | Decide whether the research (12-feature) and live (22-feature) models should converge | Written decision |
| P2 | Drift / adversarial robustness / calibration evaluation | Independent test set |
| P2 | Pick a license; signed releases | License file |

---

## Correction — 2026-09-30

The threshold-transfer paragraph previously repeated the withdrawn claim that
some datasets need roughly ten labelled flows. It now states the corrected
scientific boundary: unconditional sampling is implemented, but the exact
labelled-flow budget remains unknown until the protocol-correct rerun.
