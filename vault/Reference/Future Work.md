---
title: Future Work
tags: [reference, roadmap]
date: 2026-08-19
---

# Future Work

Prioritized out of the [[Review 2026-08-19]] and what the remediation revealed.
Highest value first within each track.

## Research

### 1. Validate the live model on real labelled traffic — *the top item*

The live model scores ~100% and three separate checks
([[F13 - Live model in-domain metrics are leaky]],
[[F14 - Host-context features shift between train and deploy]],
[[F15 - Attack classes separable by dst_port alone]]) confirm the number is
**a property of the synthetic generators**, not of the detector. Until real
captures go through `src/flow_features.py`, there is no defensible accuracy
claim for the live half at all.

IoT-23 is the obvious target: real IoT malware, real labels, and it ships raw
pcaps. Extract `iot_23_datasets_small.tar.gz`, run the pcaps through the same
22-feature pipeline, and report per-family recall. This single experiment
converts "99.99% on synthetic" into a number that survives a viva.

### 2. Threshold transfer on the CICDDoS2019 regime

The most actionable result in [[EXP02 - Corrected alignment rerun]]: pooled
training reaches **AUC 0.927 but F1 0.561** on held-out CICDDoS2019. The
ranking transfers almost perfectly; only the operating point fails.

Measure how many labelled target-domain flows are needed to re-fit a threshold
and recover the AUC-implied performance. If the answer is "a few hundred", that
is a cheap, deployable domain-adaptation story — and one the old F1-only view
could not even see.

### 3. Trace the WUSTL-IIoT polarity inversion

Pooled-trained AUC **0.314** — reliably *worse* than chance, FPR 0.94. That is
not absence of signal, it is a systematic flip. Per-feature SHAP against a
transferring dataset should localise it. A named cause here is a genuine
contribution.

### 4. Real domain adaptation

[[EXP02 - Corrected alignment rerun]] shows no *fixed* transform helps (all
seven land between AUC 0.46 and 0.57 against chance 0.50). The next tier:

- **CORAL** — align second-order statistics; ~5 lines, drops straight into
  `TRANSFORMS`
- **Test-time per-domain normalisation** — quantile-normalise the target
  domain using unlabelled target data
- **Adversarial domain-invariant encoder** — a gradient-reversal head over the
  12 features

`code/transfer_experiment.py` already has the LODO harness and the metric set,
so each is a contained experiment.

### 5. Unknown-attack handling

The model is **closed-set**: a novel attack is forced into one of 10 classes.
Add abstention — max-softmax threshold, or an isolation forest over the feature
space — so the sensor can report "anomalous, unclassified". This is what a real
IDS needs and the first thing a reviewer will ask about.

### 6. Calibrate the model

Blocked on #1 and #5. `CalibratedClassifierCV` (isotonic or Platt) on a
held-out split so a reported 0.9 is empirically 0.9, then set `min_conf` from a
target false-positive rate instead of intuition —
[[F09 - IPS gate uses uncalibrated confidence]]. Calibrating against a trivially
separable synthetic task first would produce confidently useless probabilities,
so real data has to come first.

## Engineering

### 7. Real Raspberry Pi measurements

`demo/benchmark.py` still projects Pi figures as `PI_FACTOR = 12.0` × host. The
hardware path is built and the services install; one real run replaces a guessed
constant in the results table with a measured one.

### 8. Inline (bridge) deployment guide

[[F06 - IPS only protects the sensor not the network]] added `--ips-scope
network` (INPUT + FORWARD), but making the Pi actually inline — `br0`,
`net.ipv4.ip_forward`, failing open on daemon death — is a deployment
procedure that belongs in `deploy/README_PI.md`. Without it, `scope=network` is
correct code with no instructions.

### 9. Confidence on the MCU path

`ids_predict()` returns an arg-max class id and no probability, so the C model
cannot honour any confidence gate ([[F16 - Moderate issues roundup]] item 7).
Emitting the margin, or a fixed-point softmax, would let the MCU apply the same
policy as the Python path.

### 10. CI + pytest

`tests/smoke_test.py` is a hand-rolled runner (37 checks, all passing).
Converting to pytest and adding a GitHub Action would catch regressions
automatically — including the alignment contract, the export parity guards and
the credential-literal scan, which are exactly the checks that only help if
they run on every push.

### 11. Drop the scaler from the SFAF edge model too

`code/02_train_sfaf.py` still trains through a `StandardScaler`. Its export
verifies at 100%, so this is not urgent — but the same scale-invariance
argument from [[F18 - Pipeline ONNX export silently ships a broken model]]
applies, and removing it would eliminate the same class of fragility.

### 12. Nice-to-have

- syslog / CEF export so the sensor can talk to a real SIEM
- pcapng reader (modern `tshark`/`dumpcap` emit it by default; only classic
  pcap is supported)
- per-class alert thresholds instead of one global `--min-conf`
- an ESP32 flash-and-run example in `deploy/README_MCU.md`

## Related

[[Home]] · [[Remediation Log]] · [[Review 2026-08-19]]
