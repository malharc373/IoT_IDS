# Cross-dataset generalization study

**Question:** does a flow-based IDS trained on one dataset actually detect
attacks in *other* datasets (different labs, devices, tools, attack
implementations)? This is the honest test of the overfitting concern — and the
whole motivation for a unified feature space.

**Method.** Eleven public datasets were SFAF-aligned to one 12-feature flow space
(`code/multidataset.py`) with labels normalized to a shared taxonomy. We then
trained a binary XGBoost on each dataset and tested it on every dataset — no
random split of a merged corpus (which would leak near-duplicates), so every
off-diagonal cell is genuine cross-domain transfer. 50k balanced rows/dataset.

> **Rewritten 2026-08-19; extended 2026-08-20** with IoT-23 (an eleventh
> dataset whose labels had been silently parsed as all-benign) and the
> threshold-transfer result.
>
> **The 2026-08-19 rewrite:** The previous version reported
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
| **ROC-AUC** | 0.995 | **0.509** | 0.487 | 0.500 |
| Average precision | 0.996 | 0.620 | 0.376 | = prevalence |
| **MCC** | 0.958 | **−0.007** | 0.965 | 0.000 |
| Balanced accuracy | 0.978 | 0.494 | 0.484 | 0.500 |
| Binary F1 | 0.979 | 0.438 | 0.540 | 2p/(1+p) |

```
off-diagonal cells at or below the trivial all-attack baseline: 89/110 (81%)
off-diagonal cells with ROC-AUC <= 0.55 (no usable signal)     : 60/110 (55%)
```

**Between arbitrary pairs of public flow-IDS datasets, a supervised binary
detector transfers at chance.** Not "degraded" — chance. Mean ROC-AUC 0.509 and
MCC −0.007 over 110 ordered pairs, with 81% of pairs no better than predicting
"attack" for everything. (The ten-dataset run gave 0.514 / −0.002 over 90 pairs;
adding IoT-23 moved nothing, so the result is stable to the choice of corpora.)

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
| **iot_23** | **0.195** | 0.151 | 0.678 | 0.667 | 0.919 | **inverted** |

`cicddos2019` is the important one. AUC 0.927 means the pooled model ranks that
dataset's attacks almost perfectly; the only thing failing is where the 0.5 cut
lands (recall 0.41). **A small labelled calibration sample from the target
domain would recover most of that performance** — a cheap, concrete deployment
strategy that an F1-only view reports as failure.

`wustl_iiot` (0.314) and `iot_23` (0.195) are *below* chance: the model is
reliably wrong on both. Inverting the decision would give AUC 0.686 and 0.805 —
the information is present with reversed polarity. Two independent datasets make
this a pattern rather than a quirk.

**And the calibration case is cheap to fix.** `code/threshold_transfer.py`
measures it: on CICDDoS2019, **ten labelled target flows** move F1 from 0.561 to
0.868 — 89% of the way to the 0.905 oracle, with the curve flat beyond fifty.
MQTT behaves the same (0.943 → 0.982 on ten labels). It works on exactly the
datasets where AUC is high, and not at all where it is not: on `iotid20`,
`wustl_iiot` and `iot_23` the *best achievable* threshold **is** the all-attack
classifier, so there is nothing to calibrate toward.

> Practical guidance: **measure AUC on the target domain first.** High AUC means
> a cheap calibration problem — fixable with ten labels. Low or inverted AUC
> means a representation problem, and no operating point rescues it.

## Can a deployable transform close the gap? No.

Network-flow features are heavy-tailed, so a scaler fit on one domain mismatches
another. We tested fixed feature transforms under **leave-one-dataset-out**
(train on 10, test on the 11th, averaged — `code/transfer_experiment.py`):

| Transform | AUC | ±sd | worst fold | folds >0.55 | MCC | F1 vs trivial |
|---|---|---|---|---|---|---|
| signed-log + StandardScaler | 0.640 | 0.227 | 0.152 | 8/11 | 0.118 | −0.224 |
| signed-log + RobustScaler | 0.566 | 0.249 | 0.124 | 6/11 | 0.056 | −0.254 |
| ratios + signed-log | 0.548 | 0.216 | 0.144 | 6/11 | 0.087 | −0.255 |
| raw + StandardScaler (baseline) | 0.523 | 0.238 | 0.133 | 7/11 | 0.040 | −0.296 |
| dimensionless ratios | 0.505 | 0.194 | 0.106 | 6/11 | 0.011 | −0.349 |
| quantile→normal | 0.502 | 0.238 | 0.142 | 6/11 | 0.031 | −0.300 |
| signed-log + quantile→normal | 0.496 | 0.245 | 0.089 | 5/11 | −0.003 | −0.312 |

Signed-log looks like a winner at +0.118 AUC over the baseline — until the
spread is read. **Fold-to-fold sd is 0.227**, the worst fold is 0.152, and only
8 of 11 folds clear AUC 0.55. The lift is smaller than the variation between
held-out datasets, so it is not distinguishable from which corpora happen to be
in the fold. Every transform remains below the trivial F1 baseline and MCC never
exceeds 0.12.

The previous version of this document claimed log+quantile delivered "a real,
free improvement" of +0.045 F1. That was noise in a degenerate regime. The
honest conclusion is the one it also drew, now actually supported:
**no fixed feature transform closes the gap.** Genuine domain adaptation —
CORAL, test-time per-domain normalisation, or an adversarial domain-invariant
encoder — remains the open problem for the low-AUC regime. For the *high*-AUC
regime, threshold calibration already solves it for ten labels.

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
python code/cross_dataset_eval.py --cap 50000     # 11 datasets, 110 pairs
python code/transfer_experiment.py --cap 25000    # LODO transform sweep
python code/threshold_transfer.py --cap 50000     # labels needed to calibrate
```
