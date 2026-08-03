"""
train_live_model.py — train the self-consistent live edge model.

Trains a multiclass XGBoost on the SAME flow-feature space that the live
sniffer produces (see flow_features.py), then exports a single self-contained
ONNX file with the StandardScaler baked in.  The Raspberry Pi therefore needs
only onnxruntime + numpy at inference time — no sklearn, no xgboost, no joblib.

Outputs:
    models/live_ids.onnx     scaler + classifier, one file
    models/live_meta.json     feature order, label map, metrics
    demo/results/live_confusion_matrix.png
    demo/results/live_classification_report.csv
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from flow_features import FEATURE_NAMES, N_FEATURES  # noqa: E402

DATA = os.path.join(ROOT, "data", "processed", "flows.parquet")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "demo", "results")

CAP_PER_CLASS = 8000   # balance: undersample the huge flood classes
RS = 42


def balanced_sample(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for kind, g in df.groupby("kind"):
        if len(g) > CAP_PER_CLASS:
            g = g.sample(CAP_PER_CLASS, random_state=RS)
        parts.append(g)
    out = pd.concat(parts).sample(frac=1.0, random_state=RS).reset_index(drop=True)
    return out


def export_onnx(pipe, path):
    """Convert Pipeline(StandardScaler, XGBClassifier) -> ONNX, scaler baked in."""
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
    clf = pipe.steps[-1][1]
    onx = convert_sklearn(
        pipe, "live_ids",
        initial_types=[("input", FloatTensorType([None, N_FEATURES]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
        options={id(clf): {"zipmap": False}},
    )
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return os.path.getsize(path)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} flows")
    df = balanced_sample(df)
    print("Balanced class distribution:")
    print(df["kind"].value_counts().to_string())

    # contiguous label ids in a fixed order
    kinds = ["benign", "portscan", "synflood", "icmpflood",
             "udpflood", "ssh_bruteforce", "slowloris"]
    kind_to_id = {k: i for i, k in enumerate(kinds)}
    df = df[df["kind"].isin(kinds)]
    y = df["kind"].map(kind_to_id).values
    X = df[FEATURE_NAMES].values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )
    print(f"\nTrain: {len(X_tr):,}  Test: {len(X_te):,}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=120, max_depth=6, learning_rate=0.3,
            tree_method="hist", eval_metric="mlogloss",
            objective="multi:softprob", num_class=len(kinds),
            random_state=RS, n_jobs=-1,
        )),
    ])
    pipe.fit(X_tr, y_tr)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    # binary view: benign(0) vs attack(!=0)
    bin_true = (y_te != 0).astype(int)
    bin_pred = (y_pred != 0).astype(int)
    bin_acc = accuracy_score(bin_true, bin_pred)
    bin_f1 = f1_score(bin_true, bin_pred)

    print(f"\nMulticlass accuracy : {acc:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")
    print(f"Binary accuracy    : {bin_acc:.4f}")
    print(f"Binary F1 (attack) : {bin_f1:.4f}\n")
    rep = classification_report(y_te, y_pred, target_names=kinds, digits=4)
    print(rep)

    rep_dict = classification_report(
        y_te, y_pred, target_names=kinds, output_dict=True
    )
    pd.DataFrame(rep_dict).T.to_csv(
        os.path.join(RESULTS, "live_classification_report.csv")
    )

    # confusion matrix plot
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(kinds))); ax.set_yticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=45, ha="right"); ax.set_yticklabels(kinds)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Live Edge IDS — Confusion Matrix")
    for i in range(len(kinds)):
        for j in range(len(kinds)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS, "live_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    # ── Export ────────────────────────────────────────────────────────────────
    os.makedirs(MODELS, exist_ok=True)
    onnx_path = os.path.join(MODELS, "live_ids.onnx")
    size = export_onnx(pipe, onnx_path)
    print(f"Exported {onnx_path}  ({size/1024:.1f} KB)")

    meta = {
        "model": "live_ids",
        "type": "xgboost-multiclass",
        "features": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "labels": {str(i): k for i, k in enumerate(kinds)},
        "attack_labels": [k for k in kinds if k != "benign"],
        "onnx_input": "input",
        "metrics": {
            "multiclass_accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "binary_accuracy": round(float(bin_acc), 4),
            "binary_f1_attack": round(float(bin_f1), 4),
        },
        "cap_per_class": CAP_PER_CLASS,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    with open(os.path.join(MODELS, "live_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {os.path.join(MODELS, 'live_meta.json')}")

    # ── Verify the ONNX matches sklearn predictions ───────────────────────────
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    out = sess.run(None, {"input": X_te[:2000].astype(np.float32)})
    onnx_labels = out[0].ravel()
    agree = (onnx_labels == y_pred[:2000]).mean()
    print(f"ONNX vs sklearn agreement (2000 samples): {agree*100:.2f}%")


if __name__ == "__main__":
    main()
