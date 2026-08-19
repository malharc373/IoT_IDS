---
title: Project Overview
tags: [moc, iot-ids]
date: 2026-08-19
---

# Project Overview

**IoT-IDS** is a B.Tech project in two halves, sharing one thesis: *a small,
flow-based model can detect and stop network attacks in real time on cheap
hardware.*

## Half 1 — the live edge IDS/IPS

A self-contained real-time sensor. It sniffs traffic, aggregates packets into
bidirectional flows, and classifies each flow with a ~96 KB ONNX model in
microseconds. It detects **9 attack types across 4 categories**, reports
aggregated per-source incidents, and can actively throttle or block offenders.
The same model compiles to a dependency-free C header for ESP32-class
microcontrollers.

Runs three ways off one detection core: `--pcap` (batch), `--replay` (a pcap
through the live windowed logic, root-free, for the laptop demo) and `--iface`
(real sniffing on a Pi, under systemd).

| Category | Types |
|---|---|
| recon | portscan, xmas_scan |
| dos | synflood, udpflood, icmpflood, mqtt_flood, slowloris |
| botnet | mirai |
| bruteforce | ssh_bruteforce |

**Where it stands:** ~100% on held-out synthetic scenarios — and three separate
checks confirm that number describes the *generators*, not the detector. See
[[F13 - Live model in-domain metrics are leaky]]. Validating on real labelled
captures is the top open item in [[Future Work]].

## Half 2 — the SFAF cross-dataset study

Ten public IDS datasets aligned into one 12-feature space to **measure**, not
assert, how well flow behaviour transfers across labs, devices and tools.

**The finding:** between arbitrary dataset pairs a supervised detector transfers
**at chance** — mean ROC-AUC 0.514 against a 0.500 baseline, MCC −0.002, with
80% of the 90 ordered pairs no better than answering "attack" to everything.
No fixed feature transform closes it. Full result in
[[EXP02 - Corrected alignment rerun]].

One exception is the most actionable result in the study: held out on
CICDDoS2019, the pooled model reaches **AUC 0.927 but F1 0.561** — the ranking
transfers and only the decision threshold fails.

## How to read this vault

- [[Review 2026-08-19]] — the full audit that produced everything below
- [[Remediation Log]] — the 18 findings, their status, and the commits
- [[Architecture]] — module map, data flow, invariants worth preserving
- [[Feature Spaces]] — the two feature spaces and the alignment contract
- [[Dataset Notes]] — per-dataset units, quirks and traps
- [[Future Work]] — what to do next, prioritized

## Repository

`/Users/malharfalke/IOT-IDS` · remote `github.com/malharc373/IoT_IDS`

Datasets live outside the repo (`Datasets/` symlink or `IOTIDS_DATASETS_ROOT`).
Nothing deployable is a pickle — ONNX or C header only.

## Related

[[Home]]
