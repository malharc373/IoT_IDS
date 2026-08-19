---
title: F16 — Moderate issues roundup
tags: [finding, moderate, roundup]
severity: moderate
status: fixed
date: 2026-08-19
---

# F16 — Moderate issues roundup

Smaller defects, fixed together. Each is individually minor; several would have
produced silently wrong output rather than an error.

## 1. IPv4 only — an IPv6 attacker was invisible

`parse_raw` bailed on anything that was not EtherType `0x0800`:

```python
if eth_type != 0x0800:  # IPv4 only
    return None
```

And the IPS blocklist was `type ipv4_addr`, with an iptables-only command path.
So on a dual-stack segment — normal for modern IoT, which leans on mDNS and
link-local IPv6 — an attacker was **not seen**, and even if seen could **not be
blocked**.

Fixed: IPv6 parsing including extension-header walking (hop-by-hop, routing,
fragment, AH, dstopts, mobility), `ipv6_addr` nft sets (`blocked6`,
`throttled6`), and an `ip6tables` path selected per address family by
`_sets_for()`. ICMPv6 is normalised onto the ICMP protocol value so the model
sees one "control message" identity regardless of IP version.

Also added a second stacked VLAN tag (802.1ad QinQ) — the old code handled
exactly one 802.1Q tag.

## 2. Snaplen truncation shrank every length feature

`read_pcap` returned only the stored bytes and `parse_raw` used `len(raw)` as
the packet length. On a capture taken with a snaplen (`tcpdump -s 96`, very
common) the stored frame is truncated, so `tot_bytes`, `mean_pkt_len`,
`max_pkt_len` and `bytes_per_sec` all silently shrank toward the snaplen.

The pcap record header carries both lengths, and the project's own older
`code/feature_extractor.py` had actually read `orig_len` correctly — the
capability was lost in the rewrite. `read_pcap` now returns
`(ts, raw, orig_len)` and `parse_raw(raw, orig_len)` uses the on-the-wire
length.

## 3. Flows never closed

Flows were keyed on the 5-tuple forever, with only a 120 s idle eviction. A TCP
connection that closed and whose ephemeral port was reused merged into a single
record spanning both connections — wrong duration, wrong packet counts, wrong
flag ratios.

`Flow` now tracks teardown: closed by RST, or by FIN observed in **both**
directions. `FlowTable` keeps a `(5-tuple, generation)` key so a later packet
on a closed tuple starts a distinct record. A one-sided FIN does not close the
flow, and a long-lived flow with no teardown remains one record.

## 4. Five dead scripts

`code/live_inference.py`, `code/04_live_inference.py`,
`code/03_edge_deployment.py`, `code/feature_extractor.py` and
`code/test_pcap_reader.py` all loaded `models/xgb_edge.onnx` and
`models/scaler_unified_4dataset.pkl` — artifacts not present in the repo, and
in the first case with mojibake comments (`â”€`) from an encoding accident. The
one artifact they targeted was also the one broken by
[[F03 - xgb_edge.onnx exported with the wrong scaler]]. Deleted.

## 5. ~110 MB of pickles in git

`random_forest_v1.pkl` (54 MB), `rf_multiclass.pkl` (31 MB),
`rf_baseline.pkl` (13 MB), `rf_reduced_top20.pkl` (7 MB), plus the XGBoost
pickles, encoders and scalers were tracked. `.gitignore` had `*.pkl` but they
predated it.

Beyond bloat: **a pickle executes arbitrary code on load** — an odd artifact to
ship from a security project, and nothing in the codebase loads any of them.
Untracked, with a comment in `.gitignore` recording why no `.pkl` appears in
the committed-artifacts exception list. Everything deployable is ONNX or a C
header.

## 6. Non-hermetic tests

`t_sfaf_trainer_guard` branched on whether the external `Datasets/` drive
happened to be mounted and read real CSVs off `/Volumes/GOAT`. That is what
produced the intermittent failure observed during the review — the suite
reported `23 passed, 1 failed` while the same script passed standing alone,
consistent with the volume stalling mid-run.

It now points `IOTIDS_DATASETS_ROOT` at an empty temp directory and asserts a
clean, actionable failure. `t_multidataset_taxonomy` no longer loads a real
dataset either.

## 7. The MCU path cannot honour a confidence gate

`ids_predict()` in the generated C header returns an arg-max class id and no
probability, so the `--min-conf` / `--ips-min-conf` gates that the Python path
applies have no equivalent on a microcontroller. Not fixed — it is a real
design constraint worth stating rather than a bug. Recorded in
[[Future Work]]; relevant to [[F09 - IPS gate uses uncalibrated confidence]].

## Verification

New tests `parse ipv6 / vlan / snaplen` and `tcp teardown splits flows` cover
items 1–3. Suite: 36 passed, 0 failed.

## Related

[[F17 - Documentation inconsistencies]] · [[F03 - xgb_edge.onnx exported with the wrong scaler]] ·
[[F09 - IPS gate uses uncalibrated confidence]] · [[Future Work]]
