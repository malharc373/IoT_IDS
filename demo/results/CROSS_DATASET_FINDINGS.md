# Cross-dataset generalization study

**Question:** does a flow-based IDS trained on one dataset actually detect
attacks in *other* datasets (different labs, devices, tools, attack
implementations)? This is the honest test of the overfitting concern — and the
whole motivation for a unified feature space.

**Method.** Nine public datasets were SFAF-aligned to one 12-feature flow space
(`code/multidataset.py`) with labels normalized to a shared taxonomy. We then
trained a binary XGBoost on each dataset and tested it on every dataset — no
random split of a merged corpus (which would leak near-duplicates), so every
off-diagonal cell is genuine cross-domain transfer. 50k balanced rows/dataset.

## Headline numbers

| | Binary F1 |
|---|---|
| In-domain (train and test = same dataset, diagonal) | **0.978** |
| Cross-domain (train ≠ test, off-diagonal mean) | **0.450** |
| **Generalization gap** | **0.528** |

A model that scores 0.98 on its own test data averages **0.45** on datasets it
has never seen. This is exactly the failure the project set out to address, now
measured across nine datasets instead of asserted.

## What the matrix shows (`cross_dataset_matrix.png`)

- **CICIDS2017 overfits catastrophically.** Trained on CICIDS2017 it scores 0.98
  in-domain but **0.00–0.08** on every other dataset — it has memorized
  CICFlowMeter/CICIDS-specific artifacts, not attack behavior. (This reproduces
  the classic 98%→36% CICIDS→UNSW collapse the thesis is built on.)
- **IoTID20 generalizes best** (0.64–0.69 across most datasets) — a broad IoT
  capture transfers more than a single-environment enterprise one.
- **The `bot_iot` test column is green for everyone (0.95–1.00) — an artifact,
  not a success.** Bot-IoT is 99.9% attack, so any attack-biased model scores
  high F1 there. It is not evidence of transfer and should be read with caution.
- **Pooled multi-dataset training still struggles on held-out data** (Experiment
  2): training on CICIDS2017+UNSW+TON+Bot-IoT+CIC-IoT-2023 and testing on the
  rest gives F1 = 0.10 (IoTID20), 0.34 (CICDDoS2019), 0.56 (X-IIoTID, but with a
  50% benign false-positive rate). Merely pooling datasets does not buy
  generalization.

## Honest conclusion

Naive 12-feature semantic alignment is **not** sufficient for cross-dataset
generalization. In-domain accuracy (including the live edge model's ~99.9% on
synthetic traffic) is easy; cross-domain transfer is the hard, unsolved part and
is where the real research contribution lies (better feature alignment, domain
adaptation, per-family calibration). Reporting this gap honestly is more
valuable than a single optimistic merged-split accuracy.

## Caveats
- Two datasets are lossy under alignment (MQTT-IoT-IDS2020 has no flow duration/
  rate; CIC-IoT-2023 is packet-stat based with no fwd/bwd direction) — their
  zero-filled features depress transfer somewhat.
- Bot-IoT class imbalance inflates its column (see above).
- Edge-IIoTSet and N-BaIoT are excluded — different feature paradigms that do not
  map to the flow space (documented in `code/multidataset.py`).

Reproduce: `python code/cross_dataset_eval.py --cap 50000`
