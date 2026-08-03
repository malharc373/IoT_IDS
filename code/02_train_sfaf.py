#!/usr/bin/env python3
"""
02_train_sfaf.py — headless reproduction of the SFAF unified pipeline.

This is a runnable, non-notebook version of 02_SFAF_Unified_Model.ipynb.  It
regenerates the two deployment artifacts that were gitignored out of the repo:

    models/scaler_unified_4dataset.pkl   (12-feature StandardScaler)
    models/xgb_edge.onnx                 (edge model, scaler baked in)

…plus the full-model pkl, the comparison tables, and the thesis metrics.

Requires the three datasets under Datasets/ (see code/download_datasets.py):
    Datasets/MachineLearningCVE/*.csv                 CICIDS2017
    Datasets/UNSWNB15/UNSW_NB15_{training,testing}-set.parquet
    Datasets/TONIoT/train_test_network.csv

Run:
    python code/download_datasets.py     # fetch/prepare datasets
    python code/02_train_sfaf.py         # reproduce everything
"""
from __future__ import annotations

import os
import sys
import time
import json
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CICIDS = os.path.join(BASE, "Datasets", "MachineLearningCVE")
UNSW = os.path.join(BASE, "Datasets", "UNSWNB15")
TON = os.path.join(BASE, "Datasets", "TONIoT")
MODELS = os.path.join(BASE, "models")

UNIFIED_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std",
]
UNSW_MAP = {
    "dur": "Flow Duration", "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets", "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets", "rate": "Flow Packets/s",
    "sload": "Fwd Packets/s", "dload": "Bwd Packets/s",
    "smean": "Min Packet Length", "dmean": "Max Packet Length",
    "sjit": "Packet Length Mean", "djit": "Packet Length Std",
}
TON_MAP = {
    "duration": "Flow Duration", "src_pkts": "Total Fwd Packets",
    "dst_pkts": "Total Backward Packets", "src_bytes": "Total Length of Fwd Packets",
    "dst_bytes": "Total Length of Bwd Packets", "src_ip_bytes": "Flow Packets/s",
    "dst_ip_bytes": "Bwd Packets/s", "missed_bytes": "Fwd Packets/s",
    "src_port": "Min Packet Length", "dst_port": "Max Packet Length",
    "http_request_body_len": "Packet Length Mean",
    "http_response_body_len": "Packet Length Std",
}


def _need(path, what):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] missing {what}: {path}\n"
                 f"        Run: python code/download_datasets.py")


def load_datasets():
    _need(CICIDS, "CICIDS2017 dir")
    files = sorted(glob.glob(os.path.join(CICIDS, "*.csv")))
    if not files:
        sys.exit(f"[ERROR] no CSVs in {CICIDS}")
    dfs = []
    for f in files:
        t = pd.read_csv(f, low_memory=False)
        t.columns = t.columns.str.strip()
        dfs.append(t)
    df_cic = pd.concat(dfs, ignore_index=True)
    df_cic["label"] = (df_cic["Label"].str.strip().str.upper() != "BENIGN").astype(int)

    df_unsw = pd.concat([
        pd.read_parquet(os.path.join(UNSW, "UNSW_NB15_training-set.parquet")),
        pd.read_parquet(os.path.join(UNSW, "UNSW_NB15_testing-set.parquet")),
    ], ignore_index=True)
    df_unsw.columns = df_unsw.columns.str.strip()
    df_unsw["label"] = df_unsw["label"].astype(int)
    df_unsw = df_unsw.rename(columns=UNSW_MAP)

    df_ton = pd.read_csv(os.path.join(TON, "train_test_network.csv"), low_memory=False)
    df_ton.columns = df_ton.columns.str.strip()
    df_ton["label"] = df_ton["label"].astype(int)
    df_ton = df_ton.rename(columns=TON_MAP)

    print(f"CICIDS2017: {len(df_cic):,}  UNSW-NB15: {len(df_unsw):,}  "
          f"TON-IoT: {len(df_ton):,}")
    return df_cic, df_unsw, df_ton


def preprocess(df, rs=42):
    df = df.copy().replace([np.inf, -np.inf], np.nan).dropna(
        subset=UNIFIED_FEATURES + ["label"])
    X = df[UNIFIED_FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(int)
    Xr, Xe, yr, ye = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)
    sc = StandardScaler()
    return sc.fit_transform(Xr), sc.transform(Xe), yr, ye, sc


def export_edge_onnx(scaler, edge, path):
    """Pipeline(scaler, edge XGB) -> ONNX with scaling baked in."""
    from sklearn.pipeline import Pipeline
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import (
        calculate_linear_classifier_output_shapes,
    )
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
        convert_xgboost,
    )
    update_registered_converter(
        XGBClassifier, "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    pipe = Pipeline([("sc", scaler), ("clf", edge)])
    onx = convert_sklearn(
        pipe, "xgb_edge",
        initial_types=[("input", FloatTensorType([None, 12]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
        options={id(edge): {"zipmap": False}},
    )
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return os.path.getsize(path)


def main():
    os.makedirs(MODELS, exist_ok=True)
    df_cic, df_unsw, df_ton = load_datasets()

    Xtr_c, Xte_c, ytr_c, yte_c, sc_c = preprocess(df_cic)
    Xtr_u, Xte_u, ytr_u, yte_u, sc_u = preprocess(df_unsw)
    Xtr_t, Xte_t, ytr_t, yte_t, sc_t = preprocess(df_ton)

    # ── Stage 3: generalization baseline (CICIDS-only) ────────────────────────
    base = XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist",
                         eval_metric="logloss", random_state=42)
    base.fit(Xtr_c, ytr_c)
    acc_c = accuracy_score(yte_c, base.predict(Xte_c))
    du = df_unsw.replace([np.inf, -np.inf], np.nan).dropna(subset=UNIFIED_FEATURES)
    Xu = du[UNIFIED_FEATURES].values.astype(np.float32)
    yu = du["label"].values
    acc_u = accuracy_score(yu, base.predict(sc_c.transform(Xu)))
    print(f"\n[Baseline] CICIDS in-dist={acc_c:.4f}  UNSW unseen={acc_u:.4f}  "
          f"gap={(acc_c-acc_u)*100:.1f}pp")

    # ── Stage 4: unified SFAF model ───────────────────────────────────────────
    Xtr = np.vstack([Xtr_c, Xtr_u, Xtr_t])
    ytr = np.hstack([ytr_c, ytr_u, ytr_t])
    unified = XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist",
                            eval_metric="logloss", random_state=42)
    t0 = time.time(); unified.fit(Xtr, ytr)
    print(f"[Unified] trained on {len(Xtr):,} rows in {time.time()-t0:.1f}s")
    unified.save_model(os.path.join(MODELS, "xgb_unified.json"))
    import joblib
    joblib.dump(unified, os.path.join(MODELS, "xgb_unified_4dataset.pkl"))

    results = {}
    for name, Xe, ye in [("CICIDS2017", Xte_c, yte_c),
                         ("UNSW-NB15", Xte_u, yte_u),
                         ("TON-IoT", Xte_t, yte_t)]:
        p = unified.predict(Xe); pr = unified.predict_proba(Xe)[:, 1]
        results[name] = dict(acc=accuracy_score(ye, p), f1=f1_score(ye, p),
                             auc=roc_auc_score(ye, pr))
        print(f"  {name:<12} acc={results[name]['acc']:.4f} "
              f"f1={results[name]['f1']:.4f} auc={results[name]['auc']:.4f}")

    # ── Unified scaler (fit on ALL raw rows) → the missing artifact ───────────
    df_all = pd.concat([
        df_cic.replace([np.inf, -np.inf], np.nan).dropna(subset=UNIFIED_FEATURES),
        df_unsw.replace([np.inf, -np.inf], np.nan).dropna(subset=UNIFIED_FEATURES),
        df_ton.replace([np.inf, -np.inf], np.nan).dropna(subset=UNIFIED_FEATURES),
    ], ignore_index=True)
    sc_unified = StandardScaler().fit(df_all[UNIFIED_FEATURES].values.astype(np.float32))
    joblib.dump(sc_unified, os.path.join(MODELS, "scaler_unified_4dataset.pkl"))
    print("[Artifact] models/scaler_unified_4dataset.pkl written")

    # ── Stage 5: edge model + ONNX (scaler baked in) ──────────────────────────
    edge = XGBClassifier(n_estimators=20, max_depth=4, tree_method="hist",
                         eval_metric="logloss", random_state=42)
    edge.fit(Xtr, ytr)
    joblib.dump(edge, os.path.join(MODELS, "xgb_edge.pkl"))
    onnx_path = os.path.join(MODELS, "xgb_edge.onnx")
    size = export_edge_onnx(sc_unified, edge, onnx_path)
    print(f"[Artifact] models/xgb_edge.onnx written ({size/1024:.1f} KB)")

    # ── Save reproduced metrics ───────────────────────────────────────────────
    meta = {
        "unified_features": UNIFIED_FEATURES,
        "baseline": {"cicids_acc": round(float(acc_c), 4),
                     "unsw_unseen_acc": round(float(acc_u), 4),
                     "gain_pp": round(float((results['UNSW-NB15']['acc']-acc_u)*100), 2)},
        "unified": {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                    for k, v in results.items()},
        "onnx_kb": round(size / 1024, 1),
    }
    with open(os.path.join(MODELS, "edge_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[Artifact] models/edge_meta.json written")
    print("\nDone. Regenerated scaler_unified_4dataset.pkl + xgb_edge.onnx.")


if __name__ == "__main__":
    main()
