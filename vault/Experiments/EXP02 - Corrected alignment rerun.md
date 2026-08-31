---
title: EXP02 — Corrected alignment rerun
tags: [experiment, sfaf, results]
date: 2026-08-19
supersedes: EXP01
status: superseded
superseded_by: protocol-correct rerun pending in Remediation 2026-08-22
---

# EXP02 — Cross-dataset study, corrected alignment + honest metrics

> [!danger] Superseded on 2026-08-22
> The off-diagonal cells were genuine independent-dataset tests, but the
> diagonal trained and evaluated on identical rows. Therefore the 0.995
> in-domain value and 0.487 gap below are resubstitution artifacts. Additional
> row-retention, sampling, cache and calibration corrections also require exact
> numbers to be rerun. Preserved as history; do not cite. See
> [[Remediation 2026-08-22]] and `demo/results/CROSS_DATASET_FINDINGS.md`.

Rerun of the full SFAF cross-dataset study after [[F01 - SFAF feature mappings are semantically wrong]]
(correct feature semantics + units) and [[F02 - Cross-domain F1 includes degenerate classifiers]]
(threshold-free, prevalence-robust metrics).

Supersedes [[EXP01 - Cross-dataset study baseline]].

## Setup

```bash
python code/cross_dataset_eval.py --cap 50000     # 11 datasets, 110 ordered pairs
python code/transfer_experiment.py --cap 25000    # LODO over 7 transforms
```

- **11 datasets** (IoT-23 extracted 2026-08-20 — see [[F19 - IoT-23 labels parsed as all-benign]])
- 50k rows/dataset, class-balanced where the source allows
- XGBoost, 120 trees, depth 6, lr 0.3 — identical to the pre-fix run
- NaN preserved end-to-end for structurally absent features

## Headline

Eleven datasets, 110 ordered train×test pairs:

| metric | in-domain | cross-domain | gap | chance |
|---|---|---|---|---|
| **ROC-AUC** | 0.995 | **0.509** | 0.487 | 0.500 |
| AP | 0.996 | 0.620 | 0.376 | = prevalence |
| **MCC** | 0.958 | **−0.007** | 0.965 | 0.000 |
| balanced acc | 0.978 | 0.494 | 0.484 | 0.500 |
| F1 | 0.979 | 0.438 | 0.540 | 2p/(1+p) |

```
off-diagonal cells at or below the trivial all-attack baseline: 89/110 (81%)
off-diagonal cells with ROC-AUC <= 0.55 (no usable signal)     : 60/110 (55%)
```

> The ten-dataset run gave AUC 0.514 / MCC −0.002 over 90 pairs. Adding IoT-23
> moved nothing: 0.509 / −0.007 over 110. The finding is stable to the sample
> of datasets, which is what makes it a claim about the problem rather than
> about a particular corpus.

## What changed versus EXP01, and what didn't

**Didn't change: the gap is real.** Fixing the alignment did *not* rescue
transfer. This is the important outcome — the finding survives its own audit.
Before the fix you could not tell whether the gap was domain shift or a bug in
`multidataset.py`. Now you can: it is domain shift.

**Changed: the gap is deeper than reported.** EXP01 said cross-domain F1 = 0.45
versus 0.98 in-domain. That framing implies a degraded-but-present detector.
The corrected measurement says cross-domain **MCC = −0.007, AUC = 0.509** —
there is no detector at all between arbitrary dataset pairs. Much of the old
0.45 was composed of degenerate all-attack predictors banking F1 ≈ 0.667.

**Changed: "IoTID20 generalizes best" was an artifact.** That row was an
all-attack predictor (see [[F02 - Cross-domain F1 includes degenerate classifiers]]).
In the corrected AUC matrix it carries no special status.

**Changed: the transform result reverses.** EXP01 concluded that
`log_quantile` gave "a real, free improvement" lifting LODO F1 from 0.544 to
0.589. Measured threshold-free, that improvement is noise:

Leave-one-dataset-out over all eleven, with the fold-to-fold spread — because a
mean of eleven wildly different folds is not a point estimate:

| transform | AUC | ±sd | worst fold | folds >0.55 | MCC | F1 | F1 lift |
|---|---|---|---|---|---|---|---|
| log_standard | 0.640 | 0.227 | 0.152 | 8/11 | 0.118 | 0.473 | −0.224 |
| log_robust | 0.566 | 0.249 | 0.124 | 6/11 | 0.056 | 0.443 | −0.254 |
| ratios_log | 0.548 | 0.216 | 0.144 | 6/11 | 0.087 | 0.442 | −0.255 |
| raw_standard (baseline) | 0.523 | 0.238 | 0.133 | 7/11 | 0.040 | 0.401 | −0.296 |
| ratios_standard | 0.505 | 0.194 | 0.106 | 6/11 | 0.011 | 0.348 | −0.349 |
| quantile | 0.502 | 0.238 | 0.142 | 6/11 | 0.031 | 0.397 | −0.300 |
| log_quantile | 0.496 | 0.245 | 0.089 | 5/11 | −0.003 | 0.385 | −0.312 |

`log_standard` looks like a real winner at +0.118 AUC over the baseline — until
the spread is read: **sd 0.227 across folds, worst fold 0.152**, and only 8 of
11 folds clear AUC 0.55. The lift is smaller than the fold-to-fold variation,
so it is not distinguishable from *which datasets happen to be held out*. The
script now says so automatically:

```
NOTE: the lift is smaller than the fold-to-fold spread — it is not
      distinguishable from which datasets happen to be held out.
```

Every transform is still below the trivial F1 baseline, and MCC never exceeds
0.12.

> This is the same lesson as [[F02 - Cross-domain F1 includes degenerate classifiers]]
> in a different costume: on the ten-dataset run the best-vs-baseline lift was
> +0.011 and looked like noise; on eleven it was +0.118 and looked like a
> result. Neither reading was safe without the variance.

This is a cleaner conclusion than EXP01's, and it is the one the write-up
already wanted to reach: *a fixed, deployable feature transform does not close
the cross-dataset gap.* EXP01 asserted that while simultaneously reporting a
+0.045 F1 "improvement" that pointed the other way. Now the evidence and the
conclusion agree.

## The finding the old metric hid

Pooled training (CICIDS2017 + UNSW + TON-IoT + Bot-IoT + CIC-IoT-2023), tested
on each held-out dataset:

| held out | AUC | AP | MCC | bal acc | F1 | trivial F1 | lift | FPR |
|---|---|---|---|---|---|---|---|---|
| mqtt_iot_ids2020 | 0.987 | 0.993 | 0.897 | 0.946 | 0.943 | 0.667 | **+0.276** | 0.000 |
| **cicddos2019** | **0.927** | 0.873 | 0.428 | 0.680 | **0.561** | 0.667 | −0.106 | 0.049 |
| x_iiotid | 0.706 | 0.764 | 0.321 | 0.660 | 0.667 | 0.667 | +0.000 | 0.359 |
| iotid20 | 0.606 | 0.597 | 0.080 | 0.518 | 0.135 | 0.667 | −0.532 | 0.038 |
| **wustl_iiot** | **0.314** | 0.566 | −0.583 | 0.225 | 0.337 | 0.667 | −0.330 | 0.943 |
| **iot_23** | **0.195** | 0.411 | 0.151 | 0.532 | 0.678 | 0.667 | +0.011 | 0.919 |

Three distinct regimes, indistinguishable under F1 alone:

1. **Genuine transfer** — `mqtt_iot_ids2020`, AUC 0.99 / F1 0.94 / FPR 0.00.
2. **Ranking transfers, threshold does not** — `cicddos2019`, AUC **0.927** but
   F1 only 0.561 at a 0.5 cut (recall 0.41). The model orders that domain's
   attacks nearly perfectly; only the operating point is wrong. **Ten** labelled
   target flows recover 89% of that gap — measured in
   [[EXP03 - Threshold transfer]]. **This is the single most actionable result
   in the study** and EXP01's metrics could not see it.
3. **Inverted transfer** — `wustl_iiot` (AUC **0.314**, FPR 0.94) and
   `iot_23` (AUC **0.195**, FPR 0.92): reliably *worse* than chance. A
   systematic polarity flip, not absence of signal — inverting the decision
   would give 0.686 and 0.805. Two independent datasets make it a pattern.

## Artifacts written

```
demo/results/cross_dataset_auc_matrix.csv / .png     <- primary
demo/results/cross_dataset_matrix.csv / .png         <- F1, continuity
demo/results/cross_dataset_lift_matrix.csv           <- F1 minus trivial
demo/results/cross_dataset_metrics_long.csv          <- every metric, tidy
demo/results/cross_dataset_heldout.csv
demo/results/transfer_comparison.csv
demo/results/threshold_transfer.csv                  <- EXP03
```

## Follow-ups

- ~~Threshold transfer study on the `cicddos2019` regime~~ — **done**:
  [[EXP03 - Threshold transfer]]. Ten labelled flows recover 89% of the gap
  where AUC is high, and nothing where it is not.
- Trace the `wustl_iiot` / `iot_23` polarity inversion to a feature.
- Real domain adaptation (CORAL, per-domain quantile normalisation at test
  time, adversarial domain-invariant encoder) — see [[Future Work]].

## Related

[[F01 - SFAF feature mappings are semantically wrong]] ·
[[F02 - Cross-domain F1 includes degenerate classifiers]] ·
[[EXP01 - Cross-dataset study baseline]] · [[Dataset Notes]]
