---
title: F02 — Cross-domain F1 includes degenerate classifiers
tags: [finding, critical, sfaf, research-validity, metrics]
severity: critical
status: fixed
files: ["code/cross_dataset_eval.py", "code/transfer_experiment.py"]
date: 2026-08-19
---

# F02 — Cross-domain F1 includes degenerate classifiers

> [!danger] Second, independent reason the headline number was not safe.
> Separate from [[F01 - SFAF feature mappings are semantically wrong]]: even with
> a perfect feature space, F1-at-threshold-0.5 could not support the claim being
> made from it.

## The problem

### 1. F1 has a high degenerate floor on balanced data

The evaluation caps each dataset to a ~50/50 attack/benign split. On such a set,
a classifier that predicts **"attack" for every single row** scores:

```
precision = prevalence = 0.5,  recall = 1.0
F1 = 2·0.5 / (1 + 0.5) = 0.667
```

That number is *higher* than most of the cross-domain cells being celebrated.
Checking the pre-fix matrix:

```
off-diagonal mean 0.450, median 0.511
21% of off-diagonal cells fall in [0.63, 0.70]
```

And the `iotid20` row — which `CROSS_DATASET_FINDINGS.md` singled out as
*"generalizes best (0.64–0.69 across most datasets)"* — reads:

```
0.670  0.664  0.654  0.999  0.641  —  0.666  0.667  0.688
```

Those are not transfer scores. That is a model saying "attack" to nearly
everything, scored against a balanced test set. The one cell that isn't ≈0.667
is `bot_iot` at 0.999 — which is 99.9% attack, so an all-attack predictor scores
~1.0 there. The write-up correctly flagged the Bot-IoT column as an artifact
without noticing the same artifact was producing the "best generalizer" row.

### 2. F1 at a fixed threshold cannot separate two very different failures

Under domain shift, a model's probability *ranking* often survives while its
*calibration* drifts. A fixed 0.5 cut then reports near-zero F1 for a model
whose ranking is still highly informative.

"No signal transfers" and "the signal transfers but the operating point moved"
demand completely different responses — the first needs new features or domain
adaptation, the second needs a handful of labelled target samples to re-fit a
threshold. F1-at-0.5 reports both as the same number.

## The fix

Every cell now reports a metric set chosen so that neither failure can hide:

| Metric | Chance value | What it answers |
|---|---|---|
| `roc_auc` | 0.500 | **Primary.** Does the ranking carry any signal at all? Threshold-free. |
| `ap` | = prevalence | Ranking quality weighted toward the positive class. |
| `mcc` | 0.000 | Thresholded agreement, robust to prevalence. |
| `bal_acc` | 0.500 | Thresholded, robust to prevalence. |
| `f1` | 2p/(1+p) | Kept for continuity with earlier results. |
| `f1_trivial` | — | What all-attack scores on *this* test set. |
| `f1_lift` | 0.000 | `f1 - f1_trivial`. **Negative = worse than trivial.** |

`trivial_f1()` is now a first-class function in `code/cross_dataset_eval.py`,
printed next to every dataset at load time, and the summary explicitly counts
degenerate cells. New outputs: `cross_dataset_auc_matrix.{csv,png}` (primary),
`cross_dataset_lift_matrix.csv`, and `cross_dataset_metrics_long.csv` (tidy long
format, every metric for every train×test pair).

`code/transfer_experiment.py` got the same treatment and is now ranked by AUC
rather than F1.

## Result after the fix

Full numbers in [[EXP02 - Corrected alignment rerun]]. The short version:

```
metric         in-domain  cross-domain       gap    chance
roc_auc            0.996         0.514     0.482     0.500
mcc                0.959        -0.002     0.961     0.000
bal_acc            0.979         0.496     0.483     0.500
f1                 0.979         0.440     0.539    2p/(1+p)

off-diagonal cells at or below the trivial all-attack baseline: 72/90 (80%)
off-diagonal cells with ROC-AUC <= 0.55 (no usable signal)     : 50/90 (56%)
```

**Cross-domain MCC is −0.002 and AUC is 0.514.** Not "degraded" — *chance*.

This is a stronger and more defensible claim than the original "F1 drops to
0.45", because 0.45 was partly composed of degenerate all-attack predictors
scoring 0.667. The corrected statement is:

> Between arbitrary pairs of public flow-IDS datasets aligned to a shared
> 12-feature space, a supervised binary detector transfers at chance level
> (mean ROC-AUC 0.514, MCC −0.002 over 90 ordered pairs), and 80% of pairs are
> no better than predicting "attack" for everything.

## The interesting exception

The pooled-training experiment shows the calibration-vs-signal split cleanly:

| Held out | AUC | F1 | Reading |
|---|---|---|---|
| `mqtt_iot_ids2020` | 0.987 | 0.943 | genuine transfer |
| `cicddos2019` | **0.927** | **0.561** | **ranking transfers, threshold does not** |
| `x_iiotid` | 0.706 | 0.667 | weak signal, degenerate threshold |
| `iotid20` | 0.606 | 0.135 | barely above chance |
| `wustl_iiot` | **0.314** | 0.337 | **systematically inverted** (AUC < 0.5) |

`cicddos2019` is the case the old metric could not see: AUC 0.927 means the
model ranks that dataset's attacks almost perfectly, and the only thing failing
is where the 0.5 cut lands. A small labelled calibration sample from the target
domain would recover most of that performance — a concrete, cheap research
direction that the F1-only view hid entirely.

`wustl_iiot` at AUC 0.314 is *below* chance, i.e. the model is reliably wrong
there. That is also information: inverted transfer implies a systematic feature
polarity flip, not an absence of signal.

## Related

[[F01 - SFAF feature mappings are semantically wrong]] ·
[[F12 - Missing features are zero-filled]] ·
[[EXP01 - Cross-dataset study baseline]] · [[EXP02 - Corrected alignment rerun]] ·
[[Future Work]]
