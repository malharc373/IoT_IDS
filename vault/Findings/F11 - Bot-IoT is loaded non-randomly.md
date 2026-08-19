---
title: F11 — Bot-IoT is loaded non-randomly
tags: [finding, significant, sfaf, sampling]
severity: significant
status: fixed
files: ["code/multidataset.py"]
date: 2026-08-19
---

# F11 — Bot-IoT is loaded non-randomly

## The problem

`load_botiot()` took the **first N rows** of each CSV:

```python
per = max(max_rows // max(len(files), 1), 1) if max_rows else None
for f in files:
    t = pd.read_csv(f, low_memory=False, nrows=per)     # head-read
```

Bot-IoT's CSVs are ordered by attack type and capture session, so `nrows=` does
not return a sample of the dataset — it returns the first slice of whichever
attack happens to lead each file. Everything downstream (the class balance, the
transfer matrix row and column for `bot_iot`, the LODO averages) inherited that
bias.

The contrast within the same module is telling: `load_iot23()` already did the
right thing with `t.sample(max_rows_per_file, random_state=RS)`. The two loaders
were written to different standards.

## Why it mattered here specifically

The study's own write-up flagged the `bot_iot` **column** as an artifact of
class imbalance ("Bot-IoT is 99.9% attack, so any attack-biased model scores
high F1 there"). That was correct. What went unexamined was that the `bot_iot`
**row** — the model trained *on* Bot-IoT and evaluated against everyone else —
was trained on a head-slice rather than a sample.

## The fix

Read each file fully, then sample uniformly with a fixed seed:

```python
t = pd.read_csv(f, low_memory=False)
t.columns = t.columns.str.strip()
if per and len(t) > per:
    t = t.sample(per, random_state=RS)
```

`random_state=RS` keeps runs reproducible. The cost is reading the full CSVs
rather than a head slice, which is acceptable for a study script and is what
every other loader in the module already does.

## Verification

```
bot_iot   rows=394,642  attack%=100.0  NaN-features=0
```

Bot-IoT is genuinely ~99.99% attack — with uniform sampling the benign rows are
so rare they round away at this cap. That is now a measured property of the
dataset rather than an accident of file ordering, and
[[F02 - Cross-domain F1 includes degenerate classifiers]] makes the consequence
explicit by printing Bot-IoT's trivial-baseline F1 as **1.000** next to it at
load time.

## Related

[[F01 - SFAF feature mappings are semantically wrong]] ·
[[F02 - Cross-domain F1 includes degenerate classifiers]] · [[Dataset Notes]]
