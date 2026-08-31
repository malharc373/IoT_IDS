---
title: F01 — SFAF feature mappings are semantically wrong
tags: [finding, critical, sfaf, research-validity]
severity: critical
status: fixed
files: ["code/multidataset.py", "code/02_train_sfaf.py", "code/dataset_maps.py"]
date: 2026-08-19
---

# F01 — SFAF feature mappings are semantically wrong

> [!danger] This finding invalidates the headline research number as measured.
> The README's *"in-domain F1 0.98 vs cross-domain 0.45"* was, in part, measuring
> the alignment layer's own bugs rather than a property of network traffic.

## The problem

`code/multidataset.py` aligned eleven datasets into a shared 12-feature space by
**column name substitution**, with no check that the substituted column meant
the same thing or was in the same unit. Three distinct classes of error:

### 1. Semantically unrelated columns dropped into slots

TON-IoT, IoT-23 and X-IIoTID mapped **port numbers into packet-length features**:

```python
# BEFORE — code/multidataset.py load_ton()
"Flow Packets/s":     "src_ip_bytes",           # a byte COUNT → a RATE slot
"Fwd Packets/s":      "missed_bytes",           # a byte COUNT → a RATE slot
"Min Packet Length":  "src_port",               # a TCP PORT → a LENGTH slot
"Max Packet Length":  "dst_port",               # a TCP PORT → a LENGTH slot
"Packet Length Mean": "http_request_body_len",  # an HTTP body size
"Packet Length Std":  "http_response_body_len",
```

UNSW-NB15 was similar: `sjit`/`djit` (jitter, ms) → "Packet Length Mean/Std";
`smean` (a *mean*) → "Min Packet Length"; `sload`/`dload` (**bits** per second)
→ "Packets/s".

A port number is uniformly distributed over 0–65535. A packet length is
bimodal around 60 and 1500. Feeding one into the other's slot means the model
trained on CICIDS sees a feature distribution that has nothing to do with the
one it sees on TON-IoT. Of course it does not transfer.

### 2. Unit mismatch inside the same column

CICFlowMeter datasets (CICIDS2017, CICDDoS2019, IoTID20) report `Flow Duration`
in **microseconds**. Zeek/Argus datasets (UNSW, TON-IoT, Bot-IoT, WUSTL, IoT-23,
X-IIoTID) report it in **seconds**. No conversion was applied — a 10⁶ scale
difference in feature 0 across the row axis of the transfer matrix.

### 3. Forward-only stand-ins for flow-level statistics

CICDDoS2019 and IoTID20 both ship true flow-level `Packet Length Min/Max`
columns, but the map used the forward-direction-only `Fwd Packet Length Min/Max`
/ `Fwd_Pkt_Len_Min/Max`, silently changing what the feature means relative to
CICIDS2017 (which used the flow-level one).

## Why it matters

The whole thesis contribution is the measurement *"flow behaviour does not
transfer across datasets."* That measurement is only meaningful if the feature
space is actually shared. It wasn't. The gap being reported is
**alignment error + genuine domain shift**, with no way to tell the two apart.

See [[F02 - Cross-domain F1 includes degenerate classifiers]] for the second,
independent problem with the same number.

## The fix

Rewrote the alignment layer around an explicit **alignment contract** — three
rules, stated at the top of `code/multidataset.py`:

1. **Convert units.** All durations normalised to seconds (`US_PER_S` divisor
   on the three CICFlowMeter datasets).
2. **Derive, don't substitute.** Where a dataset lacks a directional packet
   rate but has packets and duration, the rate is *computed* (`_rate()`).
   Packet-length means are derived from byte totals ÷ packet counts. MQTT's
   per-direction length stats are properly combined (min of mins, max of maxes,
   count-weighted mean, `_pooled_std()`). A semantically different column is
   never dropped into a slot.
3. **Absent means NaN, never 0** — see [[F12 - Missing features are zero-filled]].

Added `FEATURE_UNITS` (the canonical unit of every slot) and `coverage(name)` /
`_COVERAGE` (which features each dataset structurally supplies), so the honest
limits of each dataset are machine-readable instead of buried in a comment.

`_finish()` now accepts three source kinds per feature: a column name, a derived
`pd.Series`, or `None` for structurally absent — and only drops rows for the
features the dataset actually claims to supply.

## Verification

```
$ python code/multidataset.py

  cicids2017         rows=2,827,876  attack%= 19.7  NaN-features=0
  unsw_nb15          rows=  254,066  attack%= 64.8  NaN-features=3
                     absent: ['Min Packet Length', 'Max Packet Length', 'Packet Length Std']
  ton_iot            rows=  151,030  attack%= 75.3  NaN-features=3
  bot_iot            rows=  394,642  attack%=100.0  NaN-features=0
  cicddos2019        rows=  431,371  attack%= 77.3  NaN-features=0
  iotid20            rows=  625,415  attack%= 93.6  NaN-features=0
  x_iiotid           rows=  596,962  attack%= 43.5  NaN-features=3
  mqtt_iot_ids2020   rows=  206,633  attack%= 13.9  NaN-features=0 (lossy)
  cic_iot_2023       rows=1,799,932  attack%= 83.3  NaN-features=4 (lossy)
  wustl_iiot         rows=1,194,464  attack%=  7.3  NaN-features=1
```

Row counts rose substantially for several datasets (WUSTL 1.19M, MQTT 206k)
because rows are no longer discarded on account of a column the dataset was
never able to fill.

Which datasets can supply what, after the fix:

| Dataset | Missing from the 12-feature space |
|---|---|
| cicids2017, cicddos2019, iotid20, bot_iot, mqtt | — (full coverage) |
| unsw_nb15, ton_iot, iot_23, x_iiotid | packet-length min / max / std |
| wustl_iiot | packet-length std |
| cic_iot_2023 | all backward-direction features (no direction in schema) |

The Zeek-derived datasets sharing exactly the same three gaps is itself the
signal: those are Zeek's limitations, not ours, and now they are recorded as
NaN rather than fabricated.

## Consequences to re-check

- [[EXP02 - Corrected alignment rerun]] — the transfer matrix must be regenerated
- `demo/results/CROSS_DATASET_FINDINGS.md` and the README's headline table are
  stale until that rerun lands
- [[F03 - xgb_edge.onnx exported with the wrong scaler]] uses the same maps via
  `code/02_train_sfaf.py`

## Related

[[F02 - Cross-domain F1 includes degenerate classifiers]] ·
[[F12 - Missing features are zero-filled]] ·
[[F11 - Bot-IoT is loaded non-randomly]] ·
[[Dataset Notes]] · [[Feature Spaces]]
