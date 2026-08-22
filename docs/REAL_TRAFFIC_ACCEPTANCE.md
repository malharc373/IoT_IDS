# Real labelled-traffic acceptance protocol

This protocol closes R21. It is intentionally data-source-neutral because the
currently supported IoT-23 input is Zeek flow logs and cannot reproduce the
live model's packet-derived 22-feature contract.

## Required input

Obtain an authorized packet capture with ground truth that can be joined to
flows by capture, time interval, protocol, and endpoints. Record:

- source, version, license/authorization, collection topology and capture point;
- label provenance and taxonomy mapping to benign plus the nine known classes;
- device, capture session, attacker, victim and time boundaries;
- truncation/snaplen, packet loss, link type, encryption and redactions; and
- a stable SHA-256 for every immutable input.

Mixed captures need interval/flow labels; assigning one label to every flow in
an attack capture is invalid when background traffic is present.

## Grouping before measurement

Define groups before inspecting scores. At minimum, no device, capture session,
attacker identity, or generated derivative may cross train/calibration/test.
Keep one or more wholly independent test groups. If the existing synthetic
model is evaluated without retraining, report that explicitly and do not call
the test set a validation split.

## Exact live path

Run each pcap through `src/flow_features.py`/`FlowTable`, not a public dataset's
precomputed columns. Join ground truth to the emitted flow metadata, fail on
ambiguous or unmatched labels, and preserve counts for dropped packets, parsed
packets, extracted flows, labelled flows, ambiguous flows, and excluded flows.

Evaluate the committed `models/live_ids.onnx` through `ids_daemon.Detector`.
Do not fit a scaler or substitute the 12-feature SFAF artifact.

## Minimum report

- confusion matrix and per-class support/precision/recall/F1;
- attack recall and benign false-positive rate with confidence intervals;
- known-family versus unknown-family behavior;
- per-device and per-capture results, not only a pooled mean;
- confidence reliability and the effect of abstention thresholds;
- failure examples and label-join exclusions; and
- commit, model hash, metadata contract version, data hashes and exact command.

Only after this gate may the README state a real-traffic detection result. IPS
threshold selection requires a separate calibration group and cost analysis.
