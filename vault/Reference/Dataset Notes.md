---
title: Dataset Notes
tags: [reference, datasets, sfaf]
date: 2026-08-19
---

# Dataset Notes

Per-dataset schema, units and gotchas for the eleven loaders in
`code/multidataset.py`. Ten are present locally; IoT-23 needs its archive
extracted.

Root: `Datasets/` (symlink to the external drive) or `IOTIDS_DATASETS_ROOT`.

## Coverage against the 12-feature SFAF space

| dataset | tool | duration unit | cannot supply |
|---|---|---|---|
| cicids2017 | CICFlowMeter | **µs** | — |
| cicddos2019 | CICFlowMeter v3 | **µs** | — |
| iotid20 | CICFlowMeter v3 | **µs** | — |
| bot_iot | Argus | s | — |
| mqtt_iot_ids2020 | biflow | *no duration column* | — (duration approximated) |
| unsw_nb15 | Argus/Bro | s | pkt-len min / max / std |
| ton_iot | Zeek conn | s | pkt-len min / max / std |
| iot_23 | Zeek conn | s | pkt-len min / max / std |
| x_iiotid | Zeek-derived | s | pkt-len min / max / std |
| wustl_iiot | Argus | s | pkt-len std |
| cic_iot_2023 | packet-stat windows | *no duration column* | **all backward-direction features** |

The four Zeek-derived datasets sharing exactly the same three gaps is the
signal that it is Zeek's limitation, not the alignment's. Those slots are
**NaN**, never zero — [[F12 - Missing features are zero-filled]].

## Per-dataset gotchas

### CICIDS2017
- `MachineLearningCVE/*.csv`, 2.83M rows, 19.7% attack.
- Column names carry leading spaces — every loader does `.str.strip()`.
- Duration in **microseconds**. Overfits hardest of any dataset in the study.

### CICDDoS2019
- Parquet. Uses `Fwd Packets Length Total` (not `Total Length of Fwd Packets`)
  and **has** true flow-level `Packet Length Min/Max` — the old map wrongly
  used the forward-only columns.

### IoTID20
- 625k rows, **93.6% attack**. Same forward-only trap as CICDDoS2019
  (`Pkt_Len_Min/Max` are the right columns).
- Previously reported as the "best generalizer"; that was an all-attack
  predictor hitting F1 ≈ 0.667 — [[F02 - Cross-domain F1 includes degenerate classifiers]].

### UNSW-NB15
- Parquet, train + test concatenated, 254k rows.
- **Traps:** `sload`/`dload` are **bits per second**, not packets/s;
  `sjit`/`djit` are jitter in ms, not packet lengths; `smean`/`dmean` are
  per-direction *mean* sizes, so a flow-level mean is derivable but min/max/std
  are not.

### TON-IoT
- Zeek conn log, 151k rows, 75.3% attack.
- **Trap:** `src_port`/`dst_port` were previously mapped into the packet-length
  slots. Packet-length mean is derivable from `src_ip_bytes + dst_ip_bytes` ÷
  packet counts.

### Bot-IoT
- **99.99% attack.** Its trivial all-attack F1 baseline is ~1.000, so its test
  *column* in the transfer matrix looks green for every model. Not evidence of
  transfer.
- CSVs are ordered by attack, so a `nrows=` head-read returns a biased slice —
  sampled uniformly now, [[F11 - Bot-IoT is loaded non-randomly]].

### MQTT-IoT-IDS2020
- Per-file labels (`biflow_<type>.csv`) plus an `is_attack` column.
- No duration; approximated as the longer directional span,
  `(n-1) · mean_iat`. Packet-length stats exist **per direction** and are
  combined (min of mins, max of maxes, count-weighted mean, pooled std) rather
  than taking forward only.
- Transfers unusually well from pooled training (AUC 0.987) — the one clear
  success in [[EXP02 - Corrected alignment rerun]].

### CIC-IoT-2023
- Packet-stat **windows**, not biflows: no forward/backward split at all.
  `Number` is the packet count, `Rate` its packet rate, so a window span is
  derivable. Structurally lossy; flagged in `LOSSY`.

### X-IIoTID
- Single large CSV with host-telemetry columns mixed in (CPU, memory, OSSEC
  alerts) — only the network columns are used. `paket_rate` (sic) is the total
  packet rate.

### WUSTL-IIoT-2021
- Argus industrial-control flows, 1.19M rows, only **7.3% attack**.
- **Transfers inverted**: pooled-trained AUC 0.314, i.e. reliably worse than
  chance, with FPR 0.94. Worth tracing to a specific feature — see
  [[Future Work]].

### IoT-23
- Zeek `conn.log.labeled` files from `iot_23_datasets_small.tar.gz`, extracted
  under `Datasets/IoT23`. **Extracted 2026-08-20**; the study now runs on all
  eleven.
- **Two traps, both fixed in [[F19 - IoT-23 labels parsed as all-benign]]:**
  - The final `tunnel_parents / label / detailed-label` triple is separated by
    **spaces**, not tabs — in the `#fields` header *and* the data rows. Split on
    tabs alone and there is no `label` column, so the loader's benign default
    labelled the entire malware corpus as clean.
  - ~27 GB extracted, one file **10 GB alone**. `pd.read_csv` on that
    materialises tens of GB. Now byte-scan row count → chunked parse → sample
    at the corresponding fraction, with only the needed columns parsed.
- Sampled frames are cached at
  `Datasets/IoT23/_sampled_<n>_<loader-digest>.parquet` and reused while newer
  than every source file. The digest covers the loader source, so parsing,
  taxonomy or alignment changes cannot silently reuse an incompatible cache.
- Detailed-label vocabulary (by frequency): `PartOfAHorizontalPortScan`,
  `-` (benign), `DDoS`, `C&C`, `Attack`, `C&C-Torii`. Plain `C&C` had no
  taxonomy entry and fell through to `other_attack`; `c&c` is now a botnet key.
- The only source of *real IoT malware* captures. The "small" archive ships
  only Zeek logs, **not pcaps** — so it feeds the 12-feature SFAF study but
  cannot yet validate the live 22-feature model. Getting the pcap variant is
  what [[Future Work]] item 1 needs.

## Deliberately excluded

`edge_iiotset` (raw Wireshark protocol fields, no flow aggregation),
`n_baiot` (Kitsune per-packet damped-window statistics), `nsl_kdd` (legacy KDD
connection features). Different feature paradigms; mapping them would corrupt
the space rather than extend it.

## Related

[[Feature Spaces]] · [[EXP02 - Corrected alignment rerun]] ·
[[F01 - SFAF feature mappings are semantically wrong]]
