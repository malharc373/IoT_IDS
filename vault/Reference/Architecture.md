---
title: Architecture
tags: [reference, architecture]
date: 2026-08-19
---

# Architecture

Two separate prototypes testing one idea: *whether a small, flow-based model
can detect and help stop network attacks on constrained edge hardware.* The
architecture is implemented; real-traffic and target-hardware claims remain
gated on R21 and R22.

## Half 1 — the live edge IDS/IPS

```
packets ─▶ parse_raw ─▶ FlowTable ─▶ 22-feature vector ─▶ ONNX ─▶ verdict
   │           │            │                                       │
 3 sources   IPv4/IPv6   locked,                              aggregate into
 pcap /      VLAN/QinQ   windowed,                            per-(src,kind)
 replay /    snaplen-    teardown-aware,                      incidents
 live iface  aware       dirty-tracked                              │
                                                                    ▼
                                                        IPS ladder + dashboard
```

### Module map

| file | role |
|---|---|
| `src/flow_features.py` | **single source of truth** for packets → features. Imported by the trainer, the daemon, the corpus builder and the C exporter, so there is no train/serve *code* skew. |
| `src/train_live_model.py` | trains XGBoost on raw features; scenario-level split; dst_port ablation; exports ONNX and aborts if it does not match |
| `src/ids_daemon.py` | the sensor: `--pcap` / `--replay` / `--iface`; `VerdictCache`; `AlertLog` with rotation |
| `src/ips_response.py` | response ladder monitor → throttle → block; nft/iptables; v4+v6; host vs network scope |
| `src/dashboard.py` | stdlib web UI; loopback by default, token auth; incremental log reads |
| `src/export_c.py` | the ensemble as a dependency-free C header for MCUs |
| `attacks/traffic_gen.py` | scapy generators: benign family + 9 attack types |
| `attacks/build_corpus.py` | scenarios → labelled flows, attacks mixed with benign background |

### Three feeding modes, one detection core

`--pcap` (batch), `--replay` (pcap through the live windowed logic) and
`--iface` (real sniffing) all converge on the same `FlowTable` → `Detector`
path. That is what makes the root-free Mac demo meaningful as a rehearsal for
the Pi.

### Invariants worth preserving

- **Feature order is a contract.** `Detector` validates
  `meta["features"] == FEATURE_NAMES` at load.
- **Exports are verified or not shipped.** `train_live_model.py` deletes the
  ONNX and exits non-zero below 99.9% agreement; `export_c.py --verify`
  compares against the booster; `02_train_sfaf.py` does the same. This exists
  because two separate export bugs shipped silently —
  [[F03 - xgb_edge.onnx exported with the wrong scaler]] and
  [[F18 - Pipeline ONNX export silently ships a broken model]].
- **`FlowTable` is thread-safe** and its lock is public and re-entrant —
  [[F05 - Data race between sniffer thread and flush loop]].
- **Nothing deployable is a pickle.** ONNX or C header only.

## Half 2 — the SFAF cross-dataset study

```
11 public datasets ─▶ multidataset.load() ─▶ 12-feature space ─▶ XGBoost
                            │                                       │
                    alignment contract                   NxN transfer matrix
                    (units, derived, NaN)                LODO transform sweep
                            │                                       │
                     coverage() per dataset              AUC / MCC / F1-vs-trivial
```

| file | role |
|---|---|
| `code/multidataset.py` | **single source of truth** for dataset alignment |
| `code/cross_dataset_eval.py` | NxN train-on-one/test-on-others + pooled held-out |
| `code/transfer_experiment.py` | leave-one-dataset-out sweep over fixed transforms |
| `code/02_train_sfaf.py` | headless thesis pipeline; one scaler, verified export |

The notebooks in `code/*.ipynb` are a historical record of the thesis work; the
`.py` scripts are the runnable, maintained versions.

## Where the two halves meet

Nowhere, by design — and that is the honest position. The live model is a
10-class model trained on synthetic traffic in the 22-feature space; the SFAF
artifact is a binary research model in a different 12-feature space. It cannot
be passed to the daemon or described as the deployed sensor model. Metadata now
encodes this distinction and the daemon validates purpose, compatibility,
feature names and semantic contract before loading ONNX.

The live model's ~100% is a statement about the generators
([[F13 - Live model in-domain metrics are leaky]]). Exact SFAF results are
pending a protocol-correct rerun; the previous off-domain run is historical and
its resubstitution diagonal is withdrawn ([[Remediation 2026-08-22]]).

Closing that gap — running real labelled captures through
`src/flow_features.py` end to end — is the top item in [[Future Work]].

## Related

[[Feature Spaces]] · [[Dataset Notes]] · [[Remediation Log]]
