---
title: Feature Spaces
tags: [reference, features]
date: 2026-08-19
---

# Feature Spaces

The project has **two** feature spaces that must not be confused. They serve
different halves of the work and share no code.

## 1. The live space — 22 features, packets → flows

Defined in `src/flow_features.py:FEATURE_NAMES`. Order is load-bearing: the
trainer, the ONNX model, the C header and the daemon all index it positionally,
and `Detector.__init__` asserts `meta["features"] == FEATURE_NAMES`.

| # | feature | notes |
|---|---|---|
| 0 | `proto` | 6 TCP / 17 UDP / 1 ICMP (ICMPv6 normalised to 1) |
| 1 | `duration` | seconds |
| 2–3 | `tot_pkts`, `tot_bytes` | both directions |
| 4–5 | `pkts_per_sec`, `bytes_per_sec` | |
| 6–9 | `mean/std/min/max_pkt_len` | on-the-wire length (honours snaplen) |
| 10–11 | `mean_iat`, `std_iat` | inter-arrival |
| 12–15 | `syn/fin/rst/ack_ratio` | flag counts ÷ packets |
| 16 | `fwd_bwd_pkt_ratio` | asymmetry |
| 17 | `down_up_bytes_ratio` | |
| 18–20 | `host_dst_ports`, `host_dst_ips`, `host_flow_count` | **host context** |
| 21 | `dst_port` | service port the initiator targeted |

### Host context is the interesting part — and the fragile part

Features 18–20 are computed across *all* of a source's flows in the rolling
window, not from the flow itself. They are what separates a port scan (many
ports, one host) from a Mirai spread (one port, many hosts) from a flood (one
host+port, huge volume) — distinctions invisible at the single-flow level.

Two consequences fall out of that, both of which caused bugs:

- A flow's feature vector **changes when its peers change**. Any caching of
  verdicts must invalidate on the host-context triple, not just on the flow's
  own packet count — see [[F04 - Live mode reclassifies the entire flow table]].
- Their values depend on what else is on the wire, so training scenarios must
  include realistic background traffic — see
  [[F14 - Host-context features shift between train and deploy]].

### No scaler

The live model consumes **raw** features. A gradient-boosted tree splits on
`x < threshold`, so it is invariant to any strictly monotone per-feature
transform; a StandardScaler cannot change a single split. Removing it also
removed a fragile ONNX export path and 22 divisions per MCU inference —
[[F18 - Pipeline ONNX export silently ships a broken model]].

## 2. The SFAF space — 12 features, cross-dataset

Defined in `code/multidataset.py:UNIFIED_FEATURES`, with units in
`FEATURE_UNITS`. This is the space the ten public datasets are aligned into.

| feature | unit |
|---|---|
| Flow Duration | **seconds** |
| Total Fwd/Backward Packets | packets |
| Total Length of Fwd/Bwd Packets | bytes |
| Flow / Fwd / Bwd Packets/s | **packets per second** |
| Min / Max / Mean / Std Packet Length | bytes |

### The alignment contract

Three rules, stated at the top of the module and asserted in
`tests/smoke_test.py::t_alignment_contract`:

1. **Convert units.** CICFlowMeter datasets report duration in microseconds;
   Zeek/Argus ones in seconds. Everything is normalised to seconds.
2. **Derive, don't substitute.** Compute a missing rate from packets ÷
   duration; never drop a semantically different column into the slot.
3. **Absent means NaN, never 0.** A zero-filled column is a constant that
   identifies the source dataset.

All three were violated before 2026-08-19 —
[[F01 - SFAF feature mappings are semantically wrong]] and
[[F12 - Missing features are zero-filled]].

### Coverage is declared, not inferred

`multidataset.coverage(name)` records which of the 12 each dataset can
structurally supply. See [[Dataset Notes]] for the per-dataset table.

## Related

[[Architecture]] · [[Dataset Notes]] · [[Home]]
