#!/usr/bin/env python3
"""
02_train_sfaf.py — headless reproduction of the SFAF unified pipeline.

This is a runnable, non-notebook version of 02_SFAF_Unified_Model.ipynb. It
regenerates the deployment artifact that was gitignored out of the repo:

    models/xgb_edge.onnx                 (research binary model, raw 12 features)

…plus models/xgb_unified.json, models/edge_meta.json and the thesis metrics.


NO SCALER  (see vault/Findings/F03 and F18)
-------------------------------------------
The pre-2026-08 version fit a **separate** StandardScaler per dataset, stacked
the per-dataset-scaled blocks, trained the edge model on that stack, and then
baked a **different**, pooled scaler into the exported ONNX. Every prediction
the exported model made was therefore on input normalised by a scaler the model
had never been trained against (F03).

Per-dataset scaling is also an oracle that does not exist at deployment time: it
hands the model perfect per-domain normalisation, which is precisely the
information a cross-domain model is supposed to have to do without, inflating
any "the unified model closes the gap" claim.

The scaler is now gone entirely rather than merely unified. A gradient-boosted
tree splits on `x < threshold`, so it is invariant to any strictly monotone
per-feature transform — standardising the inputs cannot change a single split
decision. Removing it also removes the `convert_sklearn` Pipeline export path,
which was observed to silently emit an 83%-accurate stand-in for a 99.98% model
in the live pipeline (F18). This script's export is verified against the trained
model on every run and aborts rather than shipping a mismatch.


Datasets are loaded through code/multidataset.py, which is the single source of
truth for feature alignment (correct units, derived rather than substituted
values, NaN for structurally absent features). The old code/dataset_maps.py held
a second, divergent copy of those maps and was deleted.

Run:
    python code/download_datasets.py     # fetch/prepare datasets
    python code/02_train_sfaf.py         # reproduce everything
"""
from __future__ import annotations

import os
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(BASE, "models")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multidataset as md  # noqa: E402
from cross_dataset_eval import trivial_f1  # noqa: E402

UNIFIED_FEATURES = md.UNIFIED_FEATURES

# The three datasets the thesis pipeline is built on, plus any extras present.
CORE = ["cicids2017", "unsw_nb15", "ton_iot"]
EXTRA = ["bot_iot", "cic_iot_2023"]
RS = 42
CAP = 400_000          # rows per dataset, keeps a full run tractable


def load_datasets():
    """Load every core dataset (hard requirement) plus present extras.

    Raises SystemExit with a pointer to the downloader if a core dataset is
    missing, so the smoke test can assert clean failure without the data.
    """
    present = md.available()
    missing = [n for n in CORE if n not in present]
    if missing:
        sys.exit(f"[ERROR] missing core dataset(s): {', '.join(missing)}\n"
                 f"        Run: python code/download_datasets.py")
    frames = {}
    for name in CORE + [e for e in EXTRA if e in present]:
        df = md.load(name)
        if len(df) > CAP:
            df = df.sample(CAP, random_state=RS).reset_index(drop=True)
        frames[name] = df
        cov = md.coverage(name)
        absent = [f for f, ok in cov.items() if not ok]
        note = f"  ({len(absent)} feature(s) NaN by construction)" if absent else ""
        print(f"  {name:<16} {len(df):>9,} rows  attack%={df.y.mean()*100:5.1f}{note}")
    return frames


def split(frames):
    """Stratified 80/20 split per dataset. Returns (train, test) dicts of frames.

    CAVEAT: this is a random row split within each dataset. CICFlowMeter corpora
    contain near-duplicate flows from the same capture session, so the in-domain
    numbers below are optimistic. They are reported as an in-domain *ceiling*,
    not as a deployment estimate — the honest transfer number is the
    cross-dataset study in code/cross_dataset_eval.py.
    """
    tr, te = {}, {}
    for name, df in frames.items():
        a, b = train_test_split(df, test_size=0.2, random_state=RS,
                                stratify=df.y.values)
        tr[name], te[name] = a.reset_index(drop=True), b.reset_index(drop=True)
    return tr, te


def _X(df):
    return df[UNIFIED_FEATURES].values.astype(np.float32)


def fit_model(train_df, n_estimators=100, max_depth=6):
    """Train on RAW features — trees are scale-invariant, see the module docstring.

    base_score is pinned to a scalar: XGBoost >= 2.0 otherwise fits a per-class
    vector, which the ONNX converter mishandles (F18).
    """
    clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                        tree_method="hist", eval_metric="logloss",
                        base_score=0.5, random_state=RS, n_jobs=-1)
    clf.fit(_X(train_df), train_df.y.values)
    return clf


def evaluate(clf, test_df):
    X = _X(test_df)
    y = test_df.y.values
    p = clf.predict(X)
    both = len(np.unique(y)) > 1
    prob = clf.predict_proba(X)[:, 1]
    f1 = f1_score(y, p, zero_division=0)
    return {
        "acc": accuracy_score(y, p),
        "f1": f1,
        "f1_trivial": trivial_f1(y),
        "f1_lift": f1 - trivial_f1(y),
        "auc": roc_auc_score(y, prob) if both else float("nan"),
        "mcc": matthews_corrcoef(y, p) if both else float("nan"),
        "bal_acc": balanced_accuracy_score(y, p) if both else float("nan"),
    }


def _row(name, m):
    return (f"  {name:<16} auc={m['auc']:.4f} mcc={m['mcc']:+.4f} "
            f"bal_acc={m['bal_acc']:.4f} f1={m['f1']:.4f} "
            f"(trivial {m['f1_trivial']:.3f}, lift {m['f1_lift']:+.3f})")


def export_edge_onnx(edge, path, n_features=None):
    """Convert the XGBoost classifier straight to ONNX.

    Deliberately NOT convert_sklearn on a Pipeline — that path silently produced
    an 83%-accurate stand-in for a 99.98% model in the live pipeline (F18).
    """
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    nf = n_features or len(UNIFIED_FEATURES)
    onx = onnxmltools.convert_xgboost(
        edge, initial_types=[("input", FloatTensorType([None, nf]))],
        target_opset=15,
    )
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return os.path.getsize(path)


def main():
    os.makedirs(MODELS, exist_ok=True)
    print("Loading SFAF-aligned datasets...")
    frames = load_datasets()
    train, test = split(frames)

    pooled_train = pd.concat(list(train.values()), ignore_index=True)
    print(f"\n[Features] raw, unscaled — trees are scale-invariant (see F18); "
          f"{len(pooled_train):,} pooled training rows")

    # ── Stage 3: single-dataset baseline (CICIDS-only) ───────────────────────
    baseline = fit_model(train["cicids2017"])
    print("\n[Baseline] trained on CICIDS2017 only")
    base_results = {}
    for name in test:
        m = evaluate(baseline, test[name])
        base_results[name] = m
        tag = "in-domain " if name == "cicids2017" else "transfer  "
        print(f"  {tag}" + _row(name, m).strip())

    # ── Stage 4: unified SFAF model ──────────────────────────────────────────
    t0 = time.time()
    unified = fit_model(pooled_train)
    print(f"\n[Unified] trained on {len(pooled_train):,} pooled rows "
          f"in {time.time()-t0:.1f}s")
    unified.save_model(os.path.join(MODELS, "xgb_unified.json"))

    uni_results = {}
    for name in test:
        m = evaluate(unified, test[name])
        uni_results[name] = m
        print(_row(name, m))

    # ── Baseline vs unified, on identical held-out splits ────────────────────
    print("\n[Comparison] same held-out test split for both models")
    print(f"  {'dataset':<16}{'baseline AUC':>14}{'unified AUC':>14}{'delta':>9}")
    for name in test:
        b, u = base_results[name]["auc"], uni_results[name]["auc"]
        print(f"  {name:<16}{b:>14.4f}{u:>14.4f}{u-b:>+9.4f}")

    # ── Stage 5: edge model + ONNX ───────────────────────────────────────────
    edge = fit_model(pooled_train, n_estimators=20, max_depth=4)
    onnx_path = os.path.join(MODELS, "xgb_edge.onnx")
    size = export_edge_onnx(edge, onnx_path)
    print(f"[Artifact] models/xgb_edge.onnx written ({size/1024:.1f} KB)")

    # ── Verify the export: ONNX(raw) must equal edge(raw), or refuse to ship ──
    import onnxruntime as rt
    probe = _X(pd.concat(list(test.values()), ignore_index=True)
               .sample(min(5000, sum(len(t) for t in test.values())),
                       random_state=RS))
    ref = edge.predict(probe)
    got = rt.InferenceSession(onnx_path).run(None, {"input": probe})[0].ravel()
    agree = float((got == ref).mean())
    print(f"[Verify]  ONNX vs sklearn pipeline on {len(probe):,} rows: "
          f"{agree*100:.2f}%")
    if agree < 0.999:
        sys.exit("[ERROR] exported ONNX does not match the trained pipeline")

    meta = {
        "purpose": "sfaf_cross_dataset_research",
        "runtime_compatible": False,
        "evidence_status": "requires protocol-correct rerun",
        "unified_features": UNIFIED_FEATURES,
        "feature_units": md.FEATURE_UNITS,
        "datasets": {n: int(len(frames[n])) for n in frames},
        "scaler": None,   # trees are scale-invariant; see F18
        "baseline_cicids_only": {n: {k: round(float(v), 4) for k, v in m.items()}
                                 for n, m in base_results.items()},
        "unified": {n: {k: round(float(v), 4) for k, v in m.items()}
                    for n, m in uni_results.items()},
        "onnx_kb": round(size / 1024, 1),
        "onnx_parity": round(agree, 4),
        "caveat": ("in-domain splits are random row splits and are optimistic "
                   "for CICFlowMeter corpora; see cross_dataset_eval.py for the "
                   "honest transfer number"),
    }
    with open(os.path.join(MODELS, "edge_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[Artifact] models/edge_meta.json written")
    print("\nDone.")


if __name__ == "__main__":
    main()
