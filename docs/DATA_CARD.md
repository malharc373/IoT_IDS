# Data card

## Data domains

This repository uses two distinct data domains. They must not be merged in
claims, training code, or runtime metadata.

### Live 22-feature synthetic corpus

- **Producer:** `attacks/build_corpus.py` and `attacks/traffic_gen.py`
- **Contents:** generated packet captures and derived bidirectional flow rows
- **Labels:** benign plus nine generated attack types
- **Split:** scenario-grouped; one scenario cannot span train and test
- **Purpose:** executable integration testing, model/export development, and
  demonstrations
- **Known limitation:** the classes are synthetically generated and trivially
  separable. Accuracy does not estimate performance on real networks.
- **Storage:** generated pcaps and processed rows are ignored by Git; only the
  scripts, contract, aggregate results, and deployable artifacts are tracked.

The generator can create hostile traffic. Run it only inside an authorized,
isolated environment. Generated addresses and payloads are not a substitute
for a privacy review of real captures.

### SFAF 12-feature public-dataset study

- **Loader:** `code/multidataset.py`
- **Datasets:** the eleven sources listed in `README.md`
- **Purpose:** binary cross-dataset transfer research, not live enforcement
- **Acquisition:** external publishers/Kaggle; licenses and terms remain those
  of each source and the raw data is not redistributed here
- **Normalization:** semantic mappings, units, missingness, derivations, row
  retention, and coverage are reported by the loader
- **Evaluation:** source diagonals use an untouched 20% split; off-diagonals use
  independent datasets
- **Current status:** exact results are withheld pending a full corrected rerun
  from the unavailable external dataset mount

The supported IoT-23 input is Zeek flow logs rather than raw packet captures.
It therefore cannot validate the packet-derived 22-feature live model.

## Privacy and leakage controls

Real packet or flow data may contain addresses, payloads, credentials, device
identifiers, and behavioral traces. Do not commit it. Before analysis, document
authorization, retention, redaction, label provenance, host/time grouping, and
whether any device, scenario, or capture can leak across a split.

## Acceptance data still required

R21 needs a legally usable, labelled packet capture whose ground truth can be
joined to rows emitted by `src/flow_features.py`. Report per-family recall,
benign false-positive rate, unknown-family behavior, and grouping strategy.
The executable evidence requirements and split/join safeguards are specified in
[`REAL_TRAFFIC_ACCEPTANCE.md`](REAL_TRAFFIC_ACCEPTANCE.md).
