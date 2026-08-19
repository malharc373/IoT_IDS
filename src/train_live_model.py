"""
train_live_model.py — train the self-consistent live edge model.

Trains a multiclass XGBoost on the SAME flow-feature space that the live
sniffer produces (see flow_features.py), then exports a single self-contained
ONNX file with the StandardScaler baked in.  The Raspberry Pi therefore needs
only onnxruntime + numpy at inference time — no sklearn, no xgboost, no joblib.


SPLIT BY SCENARIO, NOT BY FLOW  (vault/Findings/F13)
-----------------------------------------------------
The corpus is generated as randomized *scenarios*. Every flow inside one
scenario shares an attacker, a victim, and — bit for bit — the same
host-context feature values, because those three features are computed across
the scenario's own flows. A random row split therefore puts near-identical
siblings on both sides of the boundary, which is why the pre-2026-08 model
reported multiclass accuracy, macro-F1, binary accuracy and binary F1 of
exactly 1.0000. That is a leakage signature, not a result.

Splitting on `scenario` with GroupShuffleSplit means the test set contains
attackers the model has never seen.


DST_PORT ABLATION  (vault/Findings/F15)
----------------------------------------
Several classes are near-separable from the target port alone (mqtt_flood is
always 1883, ssh_bruteforce always 22, slowloris always 80). A model can score
well by memorising port constants rather than learning behaviour, so every run
also trains an identical model with `dst_port` removed and reports both. The
gap between them is the honest measure of how much work the port is doing.

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

from sklearn.model_selection import GroupShuffleSplit
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


def group_split(df, features, y, test_size=0.2):
    """Split so that no scenario appears on both sides (see F13)."""
    if "scenario" not in df.columns:
        sys.exit("[ERROR] flows.parquet has no 'scenario' column — regenerate it:\n"
                 "        python attacks/build_corpus.py --scenarios 30")
    groups = df["scenario"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RS)
    tr_idx, te_idx = next(gss.split(features, y, groups=groups))
    return tr_idx, te_idx, groups


def fit_pipeline(X_tr, y_tr, n_class):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=120, max_depth=6, learning_rate=0.3,
            tree_method="hist", eval_metric="mlogloss",
            objective="multi:softprob", num_class=n_class,
            random_state=RS, n_jobs=-1,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


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

    # contiguous label ids in a fixed order (must match traffic_gen.ATTACK_KINDS)
    sys.path.insert(0, os.path.join(ROOT, "attacks"))
    import traffic_gen as tg
    kinds = list(tg.ATTACK_KINDS)
    kind_to_id = {k: i for i, k in enumerate(kinds)}
    df = df[df["kind"].isin(kinds)]
    y = df["kind"].map(kind_to_id).values
    X = df[FEATURE_NAMES].values.astype(np.float32)

    tr_idx, te_idx, groups = group_split(df, X, y)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    n_tr_scen = len(set(groups[tr_idx]))
    n_te_scen = len(set(groups[te_idx]))
    overlap = set(groups[tr_idx]) & set(groups[te_idx])
    assert not overlap, f"scenario leaked across the split: {list(overlap)[:3]}"
    print(f"\nTrain: {len(X_tr):,} flows / {n_tr_scen} scenarios"
          f"   Test: {len(X_te):,} flows / {n_te_scen} scenarios")
    print("Split is by SCENARIO — the test set contains unseen attackers.")

    pipe = fit_pipeline(X_tr, y_tr, len(kinds))

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

    # ── Ablation: how much work is dst_port doing? (F15) ─────────────────────
    port_idx = FEATURE_NAMES.index("dst_port")
    keep = [i for i in range(N_FEATURES) if i != port_idx]
    abl = fit_pipeline(X_tr[:, keep], y_tr, len(kinds))
    y_abl = abl.predict(X_te[:, keep])
    abl_acc = accuracy_score(y_te, y_abl)
    abl_f1 = f1_score(y_te, y_abl, average="macro")
    print(f"\n── dst_port ablation ────────────────────────────────────────")
    print(f"  with dst_port   : acc={acc:.4f}  macro-F1={macro_f1:.4f}")
    print(f"  without dst_port: acc={abl_acc:.4f}  macro-F1={abl_f1:.4f}")
    print(f"  delta           : acc={acc-abl_acc:+.4f}  macro-F1={macro_f1-abl_f1:+.4f}")
    per_class_full = f1_score(y_te, y_pred, average=None, labels=range(len(kinds)))
    per_class_abl = f1_score(y_te, y_abl, average=None, labels=range(len(kinds)))
    hits = [(kinds[i], per_class_full[i] - per_class_abl[i])
            for i in range(len(kinds))]
    hits.sort(key=lambda x: -x[1])
    worst = [f"{k} {d:+.3f}" for k, d in hits[:3] if d > 0.02]
    if worst:
        print(f"  most port-dependent classes: {', '.join(worst)}")
    else:
        print(f"  no class loses more than 0.02 F1 without the port")

    # ── Export ────────────────────────────────────────────────────────────────
    os.makedirs(MODELS, exist_ok=True)
    onnx_path = os.path.join(MODELS, "live_ids.onnx")
    size = export_onnx(pipe, onnx_path)
    print(f"Exported {onnx_path}  ({size/1024:.1f} KB)")

    # raw booster + scaler params — enables the dependency-free C export (MCUs)
    pipe.named_steps["clf"].get_booster().save_model(
        os.path.join(MODELS, "live_ids_booster.json"))
    sc = pipe.named_steps["scaler"]

    meta = {
        "model": "live_ids",
        "type": "xgboost-multiclass",
        "features": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "labels": {str(i): k for i, k in enumerate(kinds)},
        "categories": {k: tg.CATEGORY.get(k, "attack") for k in kinds},
        "attack_labels": [k for k in kinds if k != "benign"],
        "num_class": len(kinds),
        "scaler_mean": [round(float(v), 6) for v in sc.mean_],
        "scaler_scale": [round(float(v), 6) for v in sc.scale_],
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
        "split": "GroupShuffleSplit on scenario (no scenario spans the split)",
        "n_train_scenarios": int(n_tr_scen),
        "n_test_scenarios": int(n_te_scen),
        "ablation_no_dst_port": {
            "multiclass_accuracy": round(float(abl_acc), 4),
            "macro_f1": round(float(abl_f1), 4),
            "accuracy_delta": round(float(acc - abl_acc), 4),
            "macro_f1_delta": round(float(macro_f1 - abl_f1), 4),
        },
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
