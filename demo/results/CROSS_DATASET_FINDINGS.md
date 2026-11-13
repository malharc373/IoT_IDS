# Cross-dataset study — corrected rerun (10 November 2026)

The corrected 11-dataset SFAF study was rerun on 2026-11-10 against the full
dataset mount and passed the reproduction gate below. These are the current
results. They replace nothing in `legacy/`, which stays quarantined as
historical evidence.

Run record: [`SFAF_RERUN_MANIFEST.json`](SFAF_RERUN_MANIFEST.json) (commit,
library versions, arguments, run times, SHA-256 of every output) and
[`SFAF_RERUN_LOG.txt`](SFAF_RERUN_LOG.txt) (full console output, including the
per-dataset loader quality reports). Command:

```bash
export IOTIDS_DATASETS_ROOT=/path/to/Datasets
python code/run_sfaf_rerun.py --check-only && python code/run_sfaf_rerun.py
# = cross_dataset_eval --cap 50000; transfer_experiment --cap 25000;
#   threshold_transfer --cap 50000 --repeats 20; seed 42
```

Host: Apple Silicon Mac, macOS 26.6, Python 3.10.14, numpy 2.2.6, pandas
2.3.3, scikit-learn 1.7.2, xgboost 3.2.0. Total run time about 14.5 minutes.

## Headline: no transfer on average

Each dataset contributes 50,000 rows (balanced 50/50 except Bot-IoT; see
below). Each model trains on a fixed stratified 80% split (40,000 rows). The
diagonal is measured on the untouched 20% (10,000 rows); every off-diagonal
cell tests on the whole of a different dataset.

| | In-domain (diagonal, n = 11) | Cross-domain (off-diagonal, n = 110) | Gap |
|---|---|---|---|
| ROC-AUC, mean | **0.979** (median 0.998, min 0.856 IoT-23) | **0.497** (median 0.515) | **0.482** |
| MCC, mean | 0.898 | −0.014 | 0.911 |
| Balanced accuracy, mean | 0.944 | 0.496 | 0.448 |
| Average precision, mean | 0.976 | 0.601 | — |

Excluding Bot-IoT from both axes (see the caveat below) changes little:
diagonal ROC-AUC 0.977 / MCC 0.897, off-diagonal ROC-AUC 0.489 / MCC −0.008
over 90 cells.

- 94 of 110 off-diagonal cells (85%) are at or below the trivial all-attack
  F1 baseline.
- 65 of 110 (59%) have ROC-AUC ≤ 0.55, so no usable ranking signal.
- 50 of 110 (45%) are **below** 0.5, down to 0.016. This is systematic
  inversion, not noise: the same feature direction means "attack" in one lab
  and "benign" in another. WUSTL-IIoT as a target averages 0.267.
- Only 4 cells reach ROC-AUC ≥ 0.9 (UNSW-NB15, CICDDoS2019 and CICIDS2017 →
  MQTT-IoT-IDS2020; CICDDoS2019 → WUSTL-IIoT), and their MCC is at most 0.46,
  so even those need a recalibrated threshold.

This confirms the qualitative finding of the withdrawn run (off-diagonal mean
0.509) with a protocol-correct diagonal: within-dataset accuracy near ceiling
says nothing about accuracy on another network.

Matrices: [`cross_dataset_auc_matrix.csv`](cross_dataset_auc_matrix.csv) /
[`.png`](cross_dataset_auc_matrix.png), F1 and F1-lift matrices beside it, and
every cell with `evaluation`, `n_train`, `n_eval` in
[`cross_dataset_metrics_long.csv`](cross_dataset_metrics_long.csv).

### Bot-IoT caveat

Bot-IoT is 99.9% attack even after balanced sampling: its 50,000 rows hold
only about 55 benign flows. Its held-out 20% therefore contains **11 benign
flows**, so its diagonal ROC-AUC of 1.000 rests on 11 negatives, and each
"→ Bot-IoT" cell rests on 55. Its row (Bot-IoT as the training set) is
measured on balanced targets but comes from a model that saw about 44 benign
flows. Treat every Bot-IoT number as indicative only. The headline above is
reported both with and without it.

## Pooled training does not fix it

Training one model on five datasets (CICIDS2017, UNSW-NB15, TON_IoT, Bot-IoT,
CIC-IoT-2023; 250,000 rows) and testing on the six never-seen datasets
([`cross_dataset_heldout.csv`](cross_dataset_heldout.csv)):

| Held-out | ROC-AUC | MCC | FPR at 0.5 |
|---|---|---|---|
| MQTT-IoT-IDS2020 | 0.979 | 0.788 | 0.008 |
| CICDDoS2019 | 0.930 | 0.300 | 0.051 |
| X-IIoTID | 0.594 | 0.099 | 0.716 |
| IoTID20 | 0.513 | −0.049 | 0.022 |
| IoT-23 | 0.442 | −0.010 | 0.965 |
| WUSTL-IIoT | 0.347 | −0.052 | 0.995 |

Two targets transfer; four do not, and two of those flag almost every benign
flow.

## Feature transforms help a little

Leave-one-dataset-out over all 11 datasets
([`transfer_comparison.csv`](transfer_comparison.csv)), mean over held-out
datasets:

| Transform | ROC-AUC | MCC | Benign FPR | Deployable on MCU |
|---|---|---|---|---|
| raw_standard (baseline) | 0.612 | 0.094 | 0.379 | yes |
| log_standard | 0.571 | 0.076 | 0.425 | yes |
| log_robust | 0.572 | 0.068 | 0.425 | yes |
| ratios_standard | 0.513 | 0.019 | 0.406 | yes |
| ratios_log | 0.537 | 0.061 | 0.404 | yes |
| quantile | 0.609 | 0.075 | 0.462 | no (lookup tables) |
| log_quantile | 0.646 | 0.188 | 0.342 | no (lookup tables) |

No deployable transform beats the plain StandardScaler baseline. The best
result (log_quantile, 0.646) is a research upper bound only.

## Threshold calibration helps only where ranking already transfers

[`threshold_transfer.csv`](threshold_transfer.csv): pooled model, operating
threshold re-fit on *k* labelled target flows, 20 repeats per budget, all
42 calibrated rows report 20/20 repeats with a calibration success rate of 1.0.

- CICDDoS2019 (AUC 0.930): MCC 0.300 at the default threshold → 0.764 with
  10 labels (F1 0.410 → 0.885, oracle F1 0.920; trivial F1 0.667).
- MQTT-IoT-IDS2020 (AUC 0.979): MCC 0.788 → 0.909 with 10 labels (F1 0.871 →
  0.953, oracle F1 0.962).
- Values are means over 20 repeats; the CSV records no spread.
- IoTID20, X-IIoTID, WUSTL-IIoT, IoT-23: the best achievable threshold is the
  all-attack classifier (oracle lift ≤ 0.011), so there is nothing to
  calibrate toward.

Calibration is a real fix on 2 of 6 held-out datasets. Elsewhere the ranking
itself fails, which needs domain adaptation or target-domain training data,
not labels for a threshold.

## Reproduction gate (status: passed)

1. All eleven datasets present: **yes** (11/11). Both classes are present in
   every cell, but see the Bot-IoT caveat for how thin the benign class is.
2. Every diagonal row says `evaluation=held_out_20pct`: **yes** (11/11);
   every off-diagonal row says `independent_dataset` (110/110).
3. `n_train` and `n_eval` nonzero and disjoint by construction: **yes**
   (40,000 / 10,000 on the diagonal).
4. Loader quality reports retained with the run manifest: **yes**, in
   `SFAF_RERUN_LOG.txt`.
5. Threshold rows report all requested repeats and their calibration success
   rates: **yes** (checked by hand; the runner only checks that the column
   exists).
6. README, report, data card, benchmark and paper updated from the generated
   artifacts in the same commit: **yes**, except the presentation
   (`output/presentation/IOT_IDS_Corrected_Project_Review.pptx`), which still
   needs a manual update.

The manifest records `"dirty": true` because the freshly written result files
were untracked when it was generated. No tracked file differed from commit
`458c02a` during the run.

## Why the prior result was withdrawn (history)

The prior off-diagonal measurement was a genuine cross-dataset test and found
mean ROC-AUC 0.509 over 110 ordered pairs. However, its diagonal trained and
evaluated on the same rows. Its reported 0.995 ROC-AUC was resubstitution, so
the claimed gap of 0.487 was invalid. The 2026-08-22 audit also found that
rows with a parse failure were silently deleted, that Bot-IoT was fully loaded
before sampling, that IoT-23 caches ignored loader changes, and that
small-budget threshold means skipped single-class samples. All four were fixed
before this rerun. The affected outputs remain under
[`legacy/resubstitution-results/`](../../legacy/resubstitution-results/).

## Protocol

`code/cross_dataset_eval.py` creates one deterministic stratified 80/20 split
per dataset. Every model is fit only on its 80% partition. Its diagonal cell is
evaluated on the untouched 20%; off-diagonal cells are evaluations on
independent target datasets. The alignment layer keeps parse failures as NaN
and reports row quality. Bot-IoT uses bounded-memory reservoir sampling. IoT-23
cache names carry a digest of the loader source. Threshold calibration
evaluates every repeat, falling back to the deployed 0.5 threshold on
single-class samples, and reports the calibration success rate.
