---
title: F19 — IoT-23 labels parsed as all-benign
tags: [finding, critical, sfaf, datasets, found-during-remediation]
severity: critical
status: fixed
files: ["code/multidataset.py"]
date: 2026-08-20
---

# F19 — IoT-23 labels parsed as all-benign

> [!note] Found by extracting the archive, not by reading the code.
> `load_iot23` had never been run against real data — the archive was never
> unpacked — so the loader shipped with a silent, total labelling failure.

## Discovery

After extracting `iot_23_datasets_small.tar.gz` and re-running the study with
all eleven datasets:

```
loaded iot_23   n= 50,000 attack%=  0.0 trivial_F1=0.000
```

**0.0% attack.** IoT-23 is the CTU malware-capture dataset — it is
overwhelmingly *malicious*. A dataset that is 0% attack is not a plausible
reading of it; it is a parse failure that happens to produce valid-looking
output.

## Root cause

IoT-23's `conn.log.labeled` files deviate from standard Zeek TSV: the final
three fields are separated by **spaces**, not tabs, in both the header and the
data rows.

```
#fields  ts  uid  ...  resp_ip_bytes  tunnel_parents   label   detailed-label
1525879831.0  CUmrqr4  ...  0          (empty)   Malicious   PartOfAHorizontalPortScan
```

`_read_zeek` split `#fields` on tabs, so it produced one column literally named

```
"tunnel_parents   label   detailed-label"
```

and **no `label` column at all**. The loader's defensive default then did the
damage:

```python
lab = df.get("label", pd.Series(["Benign"] * len(df))).astype(str).str.strip()
y = (lab.str.lower() != "benign").astype(int)
```

`df.get(...)` found nothing, fell back to "Benign" for every row, and the
dataset entered the study as 50,000 perfectly benign flows.

### The defensive default is the real lesson

Every ingredient here was individually reasonable. `df.get(col, default)` is
idiomatic. A benign default is the "safe" choice. Together they converted a
**structural parse failure into a plausible dataset** — no exception, no
warning, no missing-column error. The same pattern of silent-plausible-wrong
runs through
[[F03 - xgb_edge.onnx exported with the wrong scaler]] and
[[F18 - Pipeline ONNX export silently ships a broken model]].

A missing label column should be fatal. It is not a value to be defaulted.

## The fix

`_split_composite_columns()` expands any column whose **name** contains
whitespace into its constituent columns, splitting the corresponding values the
same way. Applied by both `_read_zeek` and the new chunked reader, so
`label` and `detailed-label` exist as real columns.

## Second problem, found at the same time: memory

The extracted corpus is ~27 GB across ten files, one of them **10 GB on its
own**:

```
10G   CTU-IoT-Malware-Capture-39-1/bro/conn.log.labeled
7.6G  CTU-IoT-Malware-Capture-17-1/bro/conn.log.labeled
7.3G  CTU-IoT-Malware-Capture-33-1/bro/conn.log.labeled
```

The loader called `pd.read_csv(path)` on each and *then* sampled — materialising
tens of GB per file before discarding 99% of it. In practice it thrashed rather
than finished.

`_read_zeek_sampled()` now:

1. counts data lines with a **byte scan** (`chunk.count(b"\n")` over 4 MB
   buffers — I/O only, no parsing),
2. computes the sampling fraction from that count,
3. parses in `chunksize=250_000` blocks and samples each at that fraction.

The result is a uniform sample of the whole file with peak memory bounded by one
chunk. Sampling stays random rather than head-truncated, for the same reason as
[[F11 - Bot-IoT is loaded non-randomly]] — these files are ordered by capture
session.

## Impact on prior results

None on [[EXP02 - Corrected alignment rerun]], which ran on the ten datasets
that were actually present. IoT-23 was listed as an eleventh loader but had no
data behind it, so the published numbers were never contaminated.

Any figure quoting `iot_23` from the first eleven-dataset run is invalid —
that run reported it as 0% attack and produced `nan` metrics for it.

## Related

[[F11 - Bot-IoT is loaded non-randomly]] ·
[[F18 - Pipeline ONNX export silently ships a broken model]] ·
[[Dataset Notes]] · [[EXP02 - Corrected alignment rerun]]
