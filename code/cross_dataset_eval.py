"""
cross_dataset_eval.py — measure whether learned flow behaviour transfers across
independent datasets (labs, devices, tools, attack implementations).

This is the honest answer to the overfitting question. Instead of randomly
splitting one merged corpus (which leaks near-duplicates across the split), we:

  1. train on one or more datasets,
  2. test on OTHER, untouched datasets,
  3. report binary attack-vs-benign metrics per held-out dataset.

Two experiments:
  * NxN matrix    — train on each dataset, test on every dataset. The diagonal
                    is in-domain performance; off-diagonal is cross-dataset
                    generalisation.
  * pooled held-out — train on a chosen set, evaluate each held-out dataset.

Outputs (demo/results/):
  cross_dataset_matrix.csv / .png     F1 heatmap, train (rows) x test (cols)
  cross_dataset_heldout.csv           pooled-train per-held-out-dataset metrics
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import multidataset as md  # noqa: E402

RESULTS = os.path.join(ROOT, "demo", "results")
RS = 42


def balanced(df, cap):
    """Cap rows per dataset, keeping class balance where possible."""
    if len(df) <= cap:
        return df
    pos = df[df.y == 1]; neg = df[df.y == 0]
    half = cap // 2
    take_neg = min(len(neg), half)
    take_pos = min(len(pos), cap - take_neg)
    take_neg = min(len(neg), cap - take_pos)   # refill if one side short
    parts = []
    if take_pos: parts.append(pos.sample(take_pos, random_state=RS))
    if take_neg: parts.append(neg.sample(take_neg, random_state=RS))
    return pd.concat(parts).sample(frac=1, random_state=RS).reset_index(drop=True)


def load_all(names, cap):
    data = {}
    for n in names:
        try:
            df = balanced(md.load(n), cap)
            if df.y.nunique() >= 1 and len(df) > 100:
                data[n] = df
                print(f"  loaded {n:18} n={len(df):>7,} attack%={df.y.mean()*100:5.1f}")
        except Exception as e:
            print(f"  SKIP {n}: {e}")
    return data


def _fit(train_df):
    sc = StandardScaler().fit(train_df[md.UNIFIED_FEATURES].values.astype(np.float32))
    Xtr = sc.transform(train_df[md.UNIFIED_FEATURES].values.astype(np.float32))
    clf = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.3,
                        tree_method="hist", eval_metric="logloss",
                        random_state=RS, n_jobs=-1)
    clf.fit(Xtr, train_df.y.values)
    return sc, clf


def _eval(sc, clf, test_df):
    X = sc.transform(test_df[md.UNIFIED_FEATURES].values.astype(np.float32))
    p = clf.predict(X)
    y = test_df.y.values
    out = {"f1": f1_score(y, p, zero_division=0),
           "acc": accuracy_score(y, p),
           "recall": recall_score(y, p, zero_division=0)}
    neg = y == 0
    out["fpr"] = float((p[neg] == 1).mean()) if neg.any() else float("nan")
    return out


def nxn_matrix(data, out_png, out_csv):
    names = list(data)
    fitted = {n: _fit(data[n]) for n in names}
    print("  trained per-dataset models")
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for tr in names:
        sc, clf = fitted[tr]
        for te in names:
            M.loc[tr, te] = round(_eval(sc, clf, data[te])["f1"], 3)
    M.to_csv(out_csv)

    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.0 * len(names) + 1.5))
    im = ax.imshow(M.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("TEST dataset"); ax.set_ylabel("TRAIN dataset")
    ax.set_title("Cross-dataset binary F1 (diagonal = in-domain)")
    for i in range(len(names)):
        for j in range(len(names)):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_png, dpi=150)
    print(f"  wrote {out_csv} and {out_png}")

    # summary: mean off-diagonal (cross-domain) vs diagonal (in-domain)
    vals = M.values.astype(float)
    diag = np.diag(vals).mean()
    off = (vals.sum() - np.diag(vals).sum()) / (vals.size - len(names))
    print(f"\n  in-domain mean F1     : {diag:.3f}")
    print(f"  cross-domain mean F1  : {off:.3f}")
    print(f"  generalisation gap    : {(diag-off):.3f}")
    return M


def pooled_heldout(data, train_names, out_csv):
    train_names = [n for n in train_names if n in data]
    held = [n for n in data if n not in train_names]
    if not train_names or not held:
        print("  (skip pooled: need >=1 train and >=1 held-out present)")
        return
    train_df = pd.concat([data[n] for n in train_names], ignore_index=True)
    sc, clf = _fit(train_df)
    print(f"\n  pooled train on {train_names} (n={len(train_df):,})")
    rows = []
    for n in held:
        m = _eval(sc, clf, data[n])
        rows.append({"held_out": n, **{k: round(v, 4) for k, v in m.items()}})
        print(f"    {n:18} F1={m['f1']:.3f} acc={m['acc']:.3f} "
              f"recall={m['recall']:.3f} benign_FPR={m['fpr']:.3f}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=60_000, help="max rows per dataset")
    ap.add_argument("--train", default="cicids2017,unsw_nb15,ton_iot,bot_iot,cic_iot_2023",
                    help="pooled-train datasets (rest are held out)")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    names = md.available()
    print(f"Loading {len(names)} datasets (cap {args.cap:,}/dataset)...")
    data = load_all(names, args.cap)
    print(f"\n=== Experiment 1: NxN cross-dataset F1 matrix ===")
    nxn_matrix(data, os.path.join(RESULTS, "cross_dataset_matrix.png"),
               os.path.join(RESULTS, "cross_dataset_matrix.csv"))
    print(f"\n=== Experiment 2: pooled-train, held-out test ===")
    pooled_heldout(data, args.train.split(","),
                   os.path.join(RESULTS, "cross_dataset_heldout.csv"))


if __name__ == "__main__":
    main()
