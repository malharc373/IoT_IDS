---
title: F18 — Pipeline ONNX export silently ships a broken model
tags: [finding, critical, export, train-serve-skew, found-during-remediation]
severity: critical
status: fixed
files: ["src/train_live_model.py", "src/export_c.py"]
date: 2026-08-19
---

# F18 — The pipeline ONNX export silently shipped a broken model

> [!note] Not in the original review — found while fixing
> [[F13 - Live model in-domain metrics are leaky]], because the retrain
> surfaced it. Same family as
> [[F03 - xgb_edge.onnx exported with the wrong scaler]]: an export that
> produces a plausible artifact which does not match the model it came from.

## Discovery

After retraining on the scenario-level split, the export check printed:

```
ONNX vs sklearn agreement (2000 samples): 29.05%
```

The exported model was **internally consistent** — its `label` output matched
its own `probabilities` argmax on 100% of rows — and completely wrong: 16.6%
accuracy standing in for a 99.99% model, predicting `icmpflood` for benign
traffic. Nothing errored. Had the check not printed a number, a broken
`models/live_ids.onnx` would have shipped.

## Diagnosis

Two contributing causes, isolated by bisecting the pipeline.

**1. Vector `base_score` (partial cause).** XGBoost ≥ 2.0 fits a *per-class*
base_score vector rather than a scalar:

```
base_score: [5.2251625E-1, 6.2634754E-1, 5.396688E-1, -1.5872514E0, ...]
num_class: 10
```

With a highly separable corpus, 743 of 1200 trees collapsed to a single leaf,
so these per-class offsets dominated the score. Pinning `base_score=0.5` lifted
agreement from 29% to 87%.

This also silently broke `src/export_c.py`, whose comment reasoned that
base_score "shifts every class equally so the arg-max is unaffected" — true for
a scalar, **false for a vector**. Pinning the scalar restores that reasoning.

**2. The Pipeline converter (root cause).** 87% is still broken. Bisecting:

| what was converted | agreement with sklearn |
|---|---|
| `StandardScaler` alone | exact to 1.9e-6 |
| `XGBClassifier` alone (`onnxmltools.convert_xgboost`) | **100.00%** |
| `Pipeline([scaler, clf])` via `convert_sklearn` + `update_registered_converter` | **83%** |

Both halves convert perfectly; only the composition fails.
(skl2onnx 1.20.0 / onnxmltools 1.16.0 / xgboost 3.2.0 / onnxruntime 1.23.2.)

Ruled out: precision. No feature is near-constant, and the worst
scaling-amplified float32 error across all 22 features is ~5e-6 — nowhere near
enough to flip 17% of predictions.

## The fix: delete the scaler

Rather than work around a third-party converter bug, the scaler was removed —
justified independently of the bug:

**A gradient-boosted tree splits on `x < threshold`, so it is invariant to any
strictly monotone per-feature transform.** Standardising the inputs cannot
change a single split decision. Measured directly: identical 99.98% test
accuracy trained on scaled and on raw features.

So the scaler was a no-op that bought nothing and cost:

- the fragile `convert_sklearn` composition path (this bug)
- `IDS_MEAN` / `IDS_SCALE` tables in the C header
- 22 divisions per inference on the MCU
- the whole "scaler baked in" invariant that
  [[F03 - xgb_edge.onnx exported with the wrong scaler]] violated

The model now trains on raw features and exports via
`onnxmltools.convert_xgboost` directly.

## The guard that matters more than the fix

The parity check existed before and *printed* a number nobody had to act on.
It now aborts and deletes the artifact:

```python
if agree < 0.999:
    os.remove(onnx_path)
    sys.exit(f"[ERROR] exported ONNX disagrees with the trained pipeline "
             f"({agree*100:.2f}% agreement) — removed {onnx_path} rather than "
             f"shipping it.")
```

A verification step that cannot fail the build is documentation, not a
verification step. The same pattern is now in
`code/02_train_sfaf.py` and `src/export_c.py --verify`.

## Verification

```
ONNX vs sklearn agreement (5,000 samples): 100.00%
[verify] C vs XGBoost booster on 525 flows: 100.00%
```

Both deployment artifacts match the trained model exactly. Full suite: 34
passed. The `demo_mixed` replay detects the same 9 attack types.

## Related

[[F03 - xgb_edge.onnx exported with the wrong scaler]] ·
[[F13 - Live model in-domain metrics are leaky]] · [[Architecture]]
