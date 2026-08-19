# Cross-dataset generalization study

**Question:** does a flow-based IDS trained on one dataset actually detect
attacks in *other* datasets (different labs, devices, tools, attack
implementations)? This is the honest test of the overfitting concern — and the
whole motivation for a unified feature space.

**Method.** Ten public datasets were SFAF-aligned to one 12-feature flow space
(`code/multidataset.py`) with labels normalized to a shared taxonomy. We then
trained a binary XGBoost on each dataset and tested it on every dataset — no
random split of a merged corpus (which would leak near-duplicates), so every
off-diagonal cell is genuine cross-domain transfer. 50k balanced rows/dataset.

> **This document was rewritten on 2026-08-19.** The previous version reported
> a cross-domain F1 of 0.45 produced by (a) a feature alignment that mapped port
> numbers into packet-length slots and left flow duration in microseconds for
> some datasets and seconds for others, and (b) a metric that cannot tell
> transfer apart from a degenerate all-attack predictor. Both are fixed. The
> corrected result is *stronger*, not weaker. The superseded numbers are kept in
> `vault/Experiments/EXP01 - Cross-dataset study baseline.md`.

## Why not F1

Each test set is ~50/50 attack/benign, so **predicting "attack" for every row
scores F1 = 0.667** — higher than most cross-domain cells. F1 is also measured
at a fixed 0.5 threshold, and under domain shift a model's ranking often
survives while its calibration drifts, which F1 reports as total failure.

Every cell therefore reports ROC-AUC (chance 0.5) and MCC (chance 0.0) as the
primary metrics, with F1 kept alongside the trivial baseline it must beat.

## Headline numbers

| metric | in-domain (diagonal) | cross-domain (off-diagonal) | gap | chance |
|---|---|---|---|---|
| **ROC-AUC** | 0.996 | **0.514** | 0.482 | 0.500 |
| Average precision | 0.996 | 0.627 | 0.369 | = prevalence |
| **MCC** | 0.959 | **−0.002** | 0.961 | 0.000 |
| Balanced accuracy | 0.979 | 0.496 | 0.483 | 0.500 |
| Binary F1 | 0.979 | 0.440 | 0.539 | 2p/(1+p) |

```
off-diagonal cells at or below the trivial all-attack baseline: 72/90 (80%)
off-diagonal cells with ROC-AUC <= 0.55 (no usable signal)     : 50/90 (56%)
```

**Between arbitrary pairs of public flow-IDS datasets, a supervised binary
detector transfers at chance.** Not "degraded" — chance. Mean ROC-AUC 0.514 and
MCC −0.002 over 90 ordered pairs, with 80% of pairs no better than predicting
"attack" for everything.

## What the matrices show

Read `cross_dataset_auc_matrix.png` first — it is the one that separates *"no
signal transfers"* from *"signal transfers but the decision threshold moved"*.
`cross_dataset_matrix.png` (F1) and `cross_dataset_lift_matrix.csv` (F1 minus
the trivial baseline) are secondary. `cross_dataset_metrics_long.csv` has every
metric for every train×test pair.

- **CICIDS2017 still overfits hard.** It scores 0.98 in-domain and collapses on
  most other datasets — it has memorized CICFlowMeter/CICIDS-specific artifacts,
  not attack behavior. This reproduces the classic CICIDS→UNSW collapse the
  thesis is built on.
- **The `bot_iot` test column is an artifact, not a success.** Bot-IoT is 99.9%
  attack, so any attack-biased model scores high F1 there. Its trivial-baseline
  F1 is 1.000, which the lift matrix now makes explicit.
- **There is no "best generalizer".** The previous version of this document
  named IoTID20 as generalizing best (0.64–0.69). Those were all-attack
  predictions landing on the 0.667 trivial floor. Under AUC, no dataset holds
  that status.

## The result the old metric hid

Pooled training on CICIDS2017 + UNSW + TON-IoT + Bot-IoT + CIC-IoT-2023, tested
on each held-out dataset, shows **three distinct regimes** that F1 alone
collapses into one number:

| held out | AUC | MCC | F1 | trivial F1 | FPR | regime |
|---|---|---|---|---|---|---|
| mqtt_iot_ids2020 | 0.987 | 0.897 | 0.943 | 0.667 | 0.000 | genuine transfer |
| **cicddos2019** | **0.927** | 0.428 | **0.561** | 0.667 | 0.049 | **ranking transfers, threshold does not** |
| x_iiotid | 0.706 | 0.321 | 0.667 | 0.667 | 0.359 | weak signal, degenerate threshold |
| iotid20 | 0.606 | 0.080 | 0.135 | 0.667 | 0.038 | barely above chance |
| **wustl_iiot** | **0.314** | −0.583 | 0.337 | 0.667 | 0.943 | **inverted** (worse than chance) |

`cicddos2019` is the important one. AUC 0.927 means the pooled model ranks that
dataset's attacks almost perfectly; the only thing failing is where the 0.5 cut
lands (recall 0.41). **A small labelled calibration sample from the target
domain would recover most of that performance** — a cheap, concrete deployment
strategy that an F1-only view reports as failure.

`wustl_iiot` at AUC 0.314 is *below* chance: the model is reliably wrong there,
implying a systematic feature-polarity flip rather than absent signal.

## Can a deployable transform close the gap? No.

Network-flow features are heavy-tailed, so a scaler fit on one domain mismatches
another. We tested fixed feature transforms under **leave-one-dataset-out**
(train on 9, test on the 10th, averaged — `code/transfer_experiment.py`):

| Transform | AUC | AP | MCC | bal acc | F1 | F1 vs trivial |
|---|---|---|---|---|---|---|
| signed-log + quantile→normal | 0.569 | 0.650 | 0.042 | 0.533 | 0.375 | −0.325 |
| signed-log + StandardScaler | 0.567 | 0.648 | 0.014 | 0.519 | 0.369 | −0.331 |
| raw + StandardScaler (baseline) | 0.558 | 0.666 | 0.082 | 0.532 | 0.415 | −0.285 |
| quantile→normal | 0.542 | 0.656 | 0.064 | 0.518 | 0.393 | −0.307 |
| ratios + signed-log | 0.533 | 0.676 | 0.065 | 0.521 | 0.444 | −0.256 |
| signed-log + RobustScaler | 0.529 | 0.638 | 0.061 | 0.535 | 0.387 | −0.313 |
| dimensionless ratios | 0.456 | 0.598 | −0.012 | 0.479 | 0.330 | −0.370 |

Every transform lands between AUC 0.46 and 0.57 against a chance baseline of
0.50, with MCC ≤ 0.08, and **every one is below the trivial F1 baseline**. The
best-versus-baseline lift is +0.011 AUC — inside run-to-run noise.

The previous version of this document claimed log+quantile delivered "a real,
free improvement" of +0.045 F1. That was noise in a degenerate regime. The
honest conclusion is the one it also drew, now actually supported:
**a fixed feature transform does not close the gap at all.** Genuine domain
adaptation — CORAL, test-time per-domain normalisation, an adversarial
domain-invariant encoder, or target-domain threshold calibration — is the real
open problem.

## Caveats

- Two datasets are structurally lossy under alignment: MQTT-IoT-IDS2020 has no
  flow duration (approximated from per-direction mean IAT) and CIC-IoT-2023 is
  packet-stat based with no forward/backward direction. Those features are now
  emitted as **NaN** and handled natively by XGBoost, rather than zero-filled —
  zero-filling created a constant column that acted as a dataset fingerprint.
- Bot-IoT's class imbalance inflates its column; the lift matrix corrects for it.
- Zeek-derived datasets (TON-IoT, IoT-23, X-IIoTID) cannot supply packet-length
  min/max/std at all. `multidataset.coverage()` records this per dataset.
- Edge-IIoTSet and N-BaIoT are excluded — different feature paradigms that do
  not map to the flow space (documented in `code/multidataset.py`).

Reproduce:
```bash
python code/cross_dataset_eval.py --cap 50000
python code/transfer_experiment.py --cap 25000
```
