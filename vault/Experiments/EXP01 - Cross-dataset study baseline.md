---
title: EXP01 — Cross-dataset study baseline (pre-fix, for the record)
tags: [experiment, sfaf, archive]
date: 2026-08-04
superseded-by: EXP02
status: invalid
---

# EXP01 — Cross-dataset study, as it stood before the 2026-08-19 review

> [!warning] These numbers are not valid and are kept only for provenance.
> They were produced with a broken feature alignment
> ([[F01 - SFAF feature mappings are semantically wrong]]) and a metric that
> cannot distinguish transfer from a degenerate all-attack predictor
> ([[F02 - Cross-domain F1 includes degenerate classifiers]]).
> Superseded by [[EXP02 - Corrected alignment rerun]].

## What was reported

Nine datasets (the README said ten — see [[F17 - Documentation inconsistencies]]),
50k balanced rows each, XGBoost 120×6, binary F1 at threshold 0.5.

| | Binary F1 |
|---|---|
| In-domain (diagonal) | 0.978 |
| Cross-domain (off-diagonal mean) | 0.450 |
| Generalization gap | 0.528 |

LODO transform comparison, ranked by F1:

| transform | LODO F1 | recall | benign FPR |
|---|---|---|---|
| log_standard | 0.594 | 0.594 | 0.461 |
| log_quantile | 0.589 | 0.566 | 0.391 |
| log_robust | 0.568 | 0.551 | 0.429 |
| quantile | 0.566 | 0.521 | 0.382 |
| raw_standard (baseline) | 0.544 | 0.504 | 0.346 |
| ratios_log | 0.543 | 0.552 | 0.451 |
| ratios_standard | 0.526 | 0.507 | 0.329 |

## Claims made from it, and their status

| Claim | Status after EXP02 |
|---|---|
| "In-domain F1 0.98 vs cross-domain 0.45, a 0.53 gap" | **Understated.** True gap is AUC 0.996 → 0.514, MCC 0.959 → −0.002. |
| "CICIDS2017 overfits catastrophically (0.00–0.08 elsewhere)" | Directionally survives; the specific cells moved. |
| "IoTID20 generalizes best (0.64–0.69)" | **False.** That row was an all-attack predictor scoring the trivial 0.667. |
| "The Bot-IoT column is an artifact of 99.9% attack prevalence" | **Correct** — and the same artifact was operating on the IoTID20 row unnoticed. |
| "log+quantile lifts transfer 0.54 → 0.59, a real free improvement" | **False.** Threshold-free, the lift is +0.011 AUC on a 0.500 chance baseline — noise. |
| "A fixed transform does not close the gap; domain adaptation is the open problem" | **Correct, and now actually demonstrated** rather than asserted alongside contradicting evidence. |

## Why it is kept

The delta between EXP01 and EXP02 is itself a useful methodological result for
the thesis: it shows concretely how a plausible-looking alignment layer and a
familiar metric can jointly manufacture a "finding". That is worth a paragraph
in the write-up.

## Related

[[EXP02 - Corrected alignment rerun]] · [[Review 2026-08-19]]
