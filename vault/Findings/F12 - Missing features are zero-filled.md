---
title: F12 — Missing features are zero-filled
tags: [finding, significant, sfaf, leakage]
severity: significant
status: fixed
files: ["code/multidataset.py", "code/transfer_experiment.py"]
date: 2026-08-19
---

# F12 — Missing features are zero-filled, fingerprinting the source dataset

## The problem

When a dataset could not supply a canonical feature, the aligner substituted a
constant zero:

```python
for canon, src in feat_src.items():
    if src is None or src not in df.columns:
        out[canon] = 0.0          # <-- "absent" becomes a value
```

`code/dataset_maps.py` did the same, and `code/transfer_experiment.py` then
re-imposed it on anything that survived:

```python
X = np.nan_to_num(Xraw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
```

## Why zero is the worst possible choice

MQTT-IoT-IDS2020 has no flow duration and no rate columns, so it arrived with
`Flow Duration = 0`, `Flow Packets/s = 0`, `Fwd Packets/s = 0`,
`Bwd Packets/s = 0` — for **every row**. CIC-IoT-2023 arrived with all
backward-direction features at 0.

A constant column is a **perfect dataset fingerprint**. In any pooled or
leave-one-dataset-out experiment, a tree can learn the rule
`if Flow Duration == 0 then this row is from MQTT` and route on domain identity
instead of on attack behaviour. That is precisely the leakage the cross-dataset
protocol exists to prevent, reintroduced by the alignment layer.

Zero is also a *plausible* value for these features — a genuinely instantaneous
flow has duration 0 — so the model cannot distinguish "absent" from "measured as
zero". The two are collapsed into one number that means neither.

The study's own caveats section acknowledged half of this ("their zero-filled
features depress transfer somewhat") without recognising that zero-filling can
just as easily *inflate* results by handing the model a domain label.

## The fix

**Absent means NaN, never 0** — rule 3 of the alignment contract in
`code/multidataset.py`. `_finish()` now distinguishes three source kinds:

```python
src is None       -> emit an all-NaN column; do NOT drop rows because of it
isinstance Series -> a derived value, already in canonical units
str               -> a source column, coerced to numeric
```

Parse failures in features a dataset claims to supply are also retained as NaN
instead of deleting the row. Deleting them can select on traffic type or label.
The frame carries an `alignment_report` with per-feature missing counts:

```python
out.attrs["alignment_report"] = {
    "input_rows": len(df), "output_rows": len(out),
    "dropped_rows": 0, "rows_missing_supplied": rows_missing,
}
```

XGBoost handles NaN natively by learning a default branch direction per split,
so "absent" is represented as genuinely unknown rather than as a value.
`StandardScaler`, `RobustScaler` and `QuantileTransformer` all disregard NaN
when fitting and preserve it when transforming, so it survives the whole
pipeline — `code/transfer_experiment.py` now only converts `inf` (a real
division artifact) and leaves NaN alone.

Coverage is declared explicitly per dataset via `_COVERAGE` / `coverage(name)`,
so what each source can and cannot supply is machine-readable rather than
implicit in whether a column happened to be present.

## Side effect: many more usable rows

Because rows are no longer discarded for a column the dataset was never able to
fill, several datasets grew substantially — WUSTL-IIoT to 1.19M rows,
MQTT-IoT-IDS2020 to 206k.

## Verification

Asserted directly in the `SFAF alignment contract` test:

```python
out = mds._finish(src, fmap, [0, 1, 0], [...], "synthetic")
assert len(out) == 3, "rows dropped for a structurally absent feature"
assert out["Packet Length Std"].isna().all(), "absent feature was zero-filled"
```

Plus a related guard on `_rate()`: a rate over zero duration is **NaN**
(unknown), not `inf`. A single-packet flow of unknown length has an unknown
rate, not an infinite one — and `inf` would previously have been swept into `0.0`
by the same `nan_to_num` call.

## Related

[[F01 - SFAF feature mappings are semantically wrong]] ·
[[F02 - Cross-domain F1 includes degenerate classifiers]] ·
[[EXP02 - Corrected alignment rerun]]
