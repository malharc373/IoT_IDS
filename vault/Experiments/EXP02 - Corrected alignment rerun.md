---
title: EXP02 — Corrected alignment rerun
tags: [experiment, sfaf, results]
date: 2026-08-19
supersedes: EXP01
---

# EXP02 — Cross-dataset study, corrected alignment + honest metrics

Rerun of the full SFAF cross-dataset study after [[F01 - SFAF feature mappings are semantically wrong]]
(correct feature semantics + units) and [[F02 - Cross-domain F1 includes degenerate classifiers]]
(threshold-free, prevalence-robust metrics).

Supersedes [[EXP01 - Cross-dataset study baseline]].

## Setup

```bash
python code/cross_dataset_eval.py --cap 50000     # 10 datasets, 90 ordered pairs
python code/transfer_experiment.py --cap 25000    # LODO over 7 transforms
```

- 10 datasets present (IoT-23 archive not extracted, so 10 of the 11 loaders ran)
- 50k rows/dataset, class-balanced where the source allows
- XGBoost, 120 trees, depth 6, lr 0.3 — identical to the pre-fix run
- NaN preserved end-to-end for structurally absent features

## Headline

| metric | in-domain | cross-domain | gap | chance |
|---|---|---|---|---|
| **ROC-AUC** | 0.996 | **0.514** | 0.482 | 0.500 |
| AP | 0.996 | 0.627 | 0.369 | = prevalence |
| **MCC** | 0.959 | **−0.002** | 0.961 | 0.000 |
| balanced acc | 0.979 | 0.496 | 0.483 | 0.500 |
| F1 | 0.979 | 0.440 | 0.539 | 2p/(1+p) |

```
off-diagonal cells at or below the trivial all-attack baseline: 72/90 (80%)
off-diagonal cells with ROC-AUC <= 0.55 (no usable signal)     : 50/90 (56%)
```

## What changed versus EXP01, and what didn't

**Didn't change: the gap is real.** Fixing the alignment did *not* rescue
transfer. This is the important outcome — the finding survives its own audit.
Before the fix you could not tell whether the gap was domain shift or a bug in
`multidataset.py`. Now you can: it is domain shift.

**Changed: the gap is deeper than reported.** EXP01 said cross-domain F1 = 0.45
versus 0.98 in-domain. That framing implies a degraded-but-present detector.
The corrected measurement says cross-domain **MCC = −0.002, AUC = 0.514** —
there is no detector at all between arbitrary dataset pairs. Much of the old
0.45 was composed of degenerate all-attack predictors banking F1 ≈ 0.667.

**Changed: "IoTID20 generalizes best" was an artifact.** That row was an
all-attack predictor (see [[F02 - Cross-domain F1 includes degenerate classifiers]]).
In the corrected AUC matrix it carries no special status.

**Changed: the transform result reverses.** EXP01 concluded that
`log_quantile` gave "a real, free improvement" lifting LODO F1 from 0.544 to
0.589. Measured threshold-free, that improvement is noise:

| transform | AUC | AP | MCC | bal acc | F1 | F1 lift |
|---|---|---|---|---|---|---|
| log_quantile | 0.569 | 0.650 | 0.042 | 0.533 | 0.375 | −0.325 |
| log_standard | 0.567 | 0.648 | 0.014 | 0.519 | 0.369 | −0.331 |
| raw_standard (baseline) | 0.558 | 0.666 | 0.082 | 0.532 | 0.415 | −0.285 |
| quantile | 0.542 | 0.656 | 0.064 | 0.518 | 0.393 | −0.307 |
| ratios_log | 0.533 | 0.676 | 0.065 | 0.521 | 0.444 | −0.256 |
| log_robust | 0.529 | 0.638 | 0.061 | 0.535 | 0.387 | −0.313 |
| ratios_standard | 0.456 | 0.598 | −0.012 | 0.479 | 0.330 | −0.370 |

Every transform sits between AUC 0.46 and 0.57 against a chance baseline of
0.50, with MCC ≤ 0.08. **Every single one is below the trivial F1 baseline.**
The best-versus-baseline AUC lift is +0.011 — well inside run-to-run noise.

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

Three distinct regimes, indistinguishable under F1 alone:

1. **Genuine transfer** — `mqtt_iot_ids2020`, AUC 0.99 / F1 0.94 / FPR 0.00.
2. **Ranking transfers, threshold does not** — `cicddos2019`, AUC **0.927** but
   F1 only 0.561 at a 0.5 cut (recall 0.41). The model orders that domain's
   attacks nearly perfectly; only the operating point is wrong. A few hundred
   labelled target flows would recover most of this. **This is the single most
   actionable result in the study** and EXP01's metrics could not see it.
3. **Inverted transfer** — `wustl_iiot`, AUC **0.314**, i.e. reliably *worse*
   than chance, with FPR 0.94. A systematic polarity flip, not absence of
   signal. Worth tracing to a specific feature.

## Artifacts written

```
demo/results/cross_dataset_auc_matrix.csv / .png     <- primary
demo/results/cross_dataset_matrix.csv / .png         <- F1, continuity
demo/results/cross_dataset_lift_matrix.csv           <- F1 minus trivial
demo/results/cross_dataset_metrics_long.csv          <- every metric, tidy
demo/results/cross_dataset_heldout.csv
demo/results/transfer_comparison.csv
```

## Follow-ups

- Threshold transfer study on the `cicddos2019` regime — how many labelled
  target flows are needed to recover AUC-implied performance?
- Trace the `wustl_iiot` polarity inversion to a feature.
- Real domain adaptation (CORAL, per-domain quantile normalisation at test
  time, adversarial domain-invariant encoder) — see [[Future Work]].

## Related

[[F01 - SFAF feature mappings are semantically wrong]] ·
[[F02 - Cross-domain F1 includes degenerate classifiers]] ·
[[EXP01 - Cross-dataset study baseline]] · [[Dataset Notes]]
