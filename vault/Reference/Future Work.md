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

### ~~2. Threshold transfer on the CICDDoS2019 regime~~ — **done**

[[EXP03 - Threshold transfer]] (`code/threshold_transfer.py`). The answer was
The earlier claim that **ten** labelled flows recover 89% of the gap was based
on means that skipped single-class calibration samples. The implementation is
now unconditional, but the exact budget must be rerun after the dataset mount
is restored; see [[Remediation 2026-08-22]].

It works wherever AUC is high and nowhere else — on the low-AUC domains the
best achievable threshold *is* the all-attack classifier. That yields a
practical rule: measure AUC on the target first; high AUC is a ten-label
calibration problem, low or inverted AUC is a representation problem.

Remaining follow-up: calibrate a *probability* (Platt/isotonic on the target
sample) rather than a threshold, which would also address
[[F09 - IPS gate uses uncalibrated confidence]].

### 3. Trace the polarity inversion — now on two datasets

Pooled-trained AUC **0.314** on WUSTL-IIoT and **0.195** on IoT-23 — both
reliably *worse* than chance (FPR 0.94 and 0.92). Inverting the decision would
give 0.686 and 0.805, so the information is present with reversed polarity.
Two independent datasets make this a pattern, not a quirk. Per-feature SHAP
against a transferring dataset should localise it; a named cause is a genuine
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

`code/transfer_experiment.py` already has the LODO harness and the metric set
(now including fold-to-fold spread, which is what showed the signed-log "win"
to be within noise), so each is a contained experiment.

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

### ~~8. Inline (bridge) deployment guide~~ — **done**

`deploy/README_PI.md` §6 now covers both topologies: what a mirror-port sensor
can and cannot stop, a full systemd-networkd bridge setup with
`br_netfilter` / `bridge-nf-call-iptables`, the fail-open caveat, and
verification via `nft list table inet iot_ids`.

### 9. Confidence on the MCU path

`ids_predict()` returns an arg-max class id and no probability, so the C model
cannot honour any confidence gate ([[F16 - Moderate issues roundup]] item 7).
Emitting the margin, or a fixed-point softmax, would let the MCU apply the same
policy as the Python path.

### ~~10. CI + pytest~~ — **done**, and extended 2026-08-21

`tests/test_suite.py` parametrizes pytest over the same registry the standalone
runner uses, and `.github/workflows/ci.yml` runs the suite, an artifact-contract
assertion, a gcc compile-and-run of the C model, and the end-to-end demo on
every push.

Extended after [[Review 2026-08-21]]: `ruff check .` (pyflakes rules only — it
is what found [[F22 - Benchmark published a report with a hole in it]]), a
benchmark run that fails the build on a failed section, hermeticity assertions
(no `Datasets/` in the checkout, no `/Volumes/` path in tracked code), and
`concurrency: cancel-in-progress`.

Still missing: a type checker, and a scheduled run so bit-rot surfaces without
a push.

### ~~11. Drop the scaler from the SFAF edge model too~~ — **done**

Done — it trains on raw features and converts the classifier directly, like the
live model.

### 12. Nice-to-have

- ~~syslog / CEF export~~ — **done**: `--syslog HOST[:PORT]`,
  `--syslog-format cef|json`, best-effort delivery
- ~~pcapng reader~~ — **done**: the old reader *mis-parsed* pcapng rather than
  rejecting it; `read_pcap` now dispatches and raises on unknown formats
- per-class alert thresholds instead of one global `--min-conf`
- an ESP32 flash-and-run example in `deploy/README_MCU.md`

### 13. Assert the auxiliary state, not just the outputs

[[F20 - FlowTable generation map grows without bound]] was invisible to a
40-check suite because every check asserted on flow counts and feature values,
never on the size of the bookkeeping maps. A long-running-sensor soak check —
run N thousand connections through `FlowTable`, prune, assert every map is
bounded — would catch the whole class. The single test added covers `_gen`;
the general property does not yet have a harness.

### 14. Real Pi measurement now has a second reason

Beyond replacing `PI_FACTOR = 12.0` (item 7), a multi-hour run on real hardware
is the only thing that would have surfaced F20 without reading the code. A soak
test is the cheap proxy; the Pi is the real one.

## Related

[[Home]] · [[Remediation Log]] · [[Review 2026-08-19]] · [[Review 2026-08-21]]
