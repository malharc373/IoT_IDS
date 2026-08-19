---
title: F03 — xgb_edge.onnx exported with the wrong scaler
tags: [finding, critical, sfaf, train-serve-skew, export]
severity: critical
status: fixed
files: ["code/02_train_sfaf.py", "code/dataset_maps.py"]
date: 2026-08-19
---

# F03 — `xgb_edge.onnx` exported with the wrong scaler

> [!danger] Silent corruption of a shipped artifact.
> Every prediction `models/xgb_edge.onnx` ever made was on input normalised by a
> scaler the model inside it had never been trained against. Nothing errored.

## The problem

`code/02_train_sfaf.py` fit a **separate** `StandardScaler` per dataset inside
`preprocess()`, which returned already-scaled feature blocks:

```python
def preprocess(df, rs=42):
    ...
    Xr, Xe, yr, ye = train_test_split(X, y, test_size=0.2, ...)
    sc = StandardScaler()
    return sc.fit_transform(Xr), sc.transform(Xe), yr, ye, sc   # per-dataset sc
```

Those per-dataset-scaled blocks were then stacked and used for training:

```python
Xtr_c, Xte_c, ytr_c, yte_c, sc_c = preprocess(df_cic)   # scaler A
Xtr_u, Xte_u, ytr_u, yte_u, sc_u = preprocess(df_unsw)  # scaler B
Xtr_t, Xte_t, ytr_t, yte_t, sc_t = preprocess(df_ton)   # scaler C
...
Xtr = np.vstack([Xtr_c, Xtr_u, Xtr_t, ...])             # A|B|C-scaled stack
edge.fit(Xtr, ytr)                                       # trained on that
```

…and then a **fourth, different** scaler was fit on the raw pooled rows and
baked into the export:

```python
sc_unified = StandardScaler().fit(df_all[UNIFIED_FEATURES]...)   # scaler D
export_edge_onnx(sc_unified, edge, onnx_path)                    # D + model(A|B|C)
```

The exported ONNX applies **D** and hands the result to a model trained on
**A|B|C**. A classic train/serve skew, made invisible by the fact that both
paths produce plausible-looking floats.

`models/xgb_edge.onnx` is exactly the artifact loaded by `code/live_inference.py`,
`code/04_live_inference.py` and `code/03_edge_deployment.py` (all of which were
also broken for a different reason — see [[F16 - Moderate issues roundup]]).

## The second, subtler problem

Per-dataset scaling is not just inconsistent with the export — it is an
**oracle that does not exist at deployment time**. Fitting a scaler per source
domain hands the model perfect per-domain normalisation, which is precisely the
information a cross-domain model is supposed to do without. Any "the unified
SFAF model closes the generalization gap" claim measured this way is inflated by
a signal the deployed system can never have.

## The fix

**One scaler, fit once, on pooled raw training rows** — used for training, for
evaluation, and for the export. The single-dataset baseline gets its own
single-dataset scaler, because that is genuinely what a single-dataset
practitioner would have.

The script was also rewired onto `code/multidataset.py` as the single source of
truth for alignment, and `code/dataset_maps.py` — which held a **second,
divergent copy** of the same broken maps from
[[F01 - SFAF feature mappings are semantically wrong]], plus its own inventions
(`Header_Length` → "Total Backward Packets", `Tot size` → "Total Length of Bwd
Packets") — was deleted. Three loader stacks became one.

Finally, the export is now **verified in-script** and the run fails loudly if it
ever drifts again:

```python
ref = edge.predict(scaler.transform(probe))
got = rt.InferenceSession(onnx_path).run(None, {"input": probe})[0].ravel()
if (got == ref).mean() < 0.999:
    sys.exit("[ERROR] exported ONNX does not match the trained pipeline")
```

## Verification

```
[Scaler] one StandardScaler fit on 1,279,789 pooled raw training rows
         — used for training, evaluation and export
...
[Artifact] models/xgb_edge.onnx written (19.1 KB)
[Verify]  ONNX vs sklearn pipeline on 5,000 rows: 100.00%
```

## Bonus: the SFAF result is now actually demonstrated

With one honest scaler and corrected alignment, the baseline-vs-unified
comparison finally shows what the thesis claims, on identical held-out splits:

| dataset | CICIDS-only baseline AUC | unified AUC | delta |
|---|---|---|---|
| cicids2017 | 0.9984 | 0.9982 | −0.0002 |
| **unsw_nb15** | **0.2191** | **0.9830** | **+0.7639** |
| ton_iot | 0.7383 | 0.9996 | +0.2613 |
| bot_iot | 0.6465 | 1.0000 | +0.3535 |
| cic_iot_2023 | 0.5892 | 0.9532 | +0.3641 |

The CICIDS-only model scores **AUC 0.219 on UNSW-NB15** — meaningfully *below*
chance, i.e. reliably inverted, which is a sharper illustration of the
overfitting problem than the old accuracy-based framing.

> [!note] What this does and does not show
> These are in-distribution test splits of datasets the unified model trained
> on. It shows pooling helps *on the domains you pooled*. It is **not**
> cross-dataset generalization — that is [[EXP02 - Corrected alignment rerun]],
> where transfer to genuinely unseen datasets remains at chance. The script's
> docstring and `edge_meta.json` now both carry this caveat.

## Related

[[F01 - SFAF feature mappings are semantically wrong]] ·
[[F13 - Live model in-domain metrics are leaky]] ·
[[F16 - Moderate issues roundup]] · [[EXP02 - Corrected alignment rerun]]
