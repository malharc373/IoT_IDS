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


WHY F1 ALONE IS NOT ENOUGH  (see vault/Findings/F02)
----------------------------------------------------
On a class-balanced test set, a classifier that predicts "attack" for every
single row scores **F1 = 0.667**. That is not a hypothetical: in the pre-2026-08
matrix, 21% of off-diagonal cells sat in [0.63, 0.70] and the row the write-up
called "generalizes best" was 0.667 almost uniformly — a degenerate all-attack
predictor being read as successful transfer.

F1 is also threshold-dependent. Under domain shift a model's probabilities
routinely stay well-ordered while their *calibration* drifts, so a fixed 0.5
cut can report F1 ≈ 0 for a model whose ranking is still informative. "No signal
transfers" and "the signal transfers but the threshold moved" are completely
different research findings and F1-at-0.5 cannot distinguish them.

So every cell now reports:

  roc_auc   threshold-free ranking quality.   0.5 = no signal.  PRIMARY METRIC.
  ap        average precision (PR-AUC), threshold-free, prevalence-aware.
  mcc       Matthews correlation.             0.0 = no better than chance.
  bal_acc   balanced accuracy.                0.5 = chance.
  f1        kept for continuity with earlier results.
  f1_trivial  what always-predict-attack scores on THIS test set: 2p/(1+p).
  f1_lift     f1 - f1_trivial. Negative means worse than the trivial baseline.

Outputs (demo/results/):
  cross_dataset_auc_matrix.csv / .png   ROC-AUC heatmap  (primary)
  cross_dataset_matrix.csv / .png       F1 heatmap       (continuity)
  cross_dataset_lift_matrix.csv         F1 minus trivial-baseline F1
  cross_dataset_metrics_long.csv        every metric, tidy long format
  cross_dataset_heldout.csv             pooled-train per-held-out-dataset metrics
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
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import multidataset as md  # noqa: E402

RESULTS = os.path.join(ROOT, "demo", "results")
RS = 42


def trivial_f1(y):
    """F1 of the all-positive ('everything is an attack') classifier.

    precision = prevalence p, recall = 1  =>  F1 = 2p / (1 + p).
    Any reported F1 at or below this number is evidence of a degenerate
    classifier, not of transfer.
    """
    p = float(np.mean(y))
    return 2 * p / (1 + p) if p > 0 else 0.0


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
                print(f"  loaded {n:18} n={len(df):>7,} attack%={df.y.mean()*100:5.1f} "
                      f"trivial_F1={trivial_f1(df.y.values):.3f}")
        except Exception as e:
            print(f"  SKIP {n}: {e}")
    return data


def _fit(train_df):
    """StandardScaler + XGBoost.

    Both handle NaN natively (the scaler ignores it in fit and preserves it in
    transform; XGBoost learns a default split direction for it), which is what
    lets structurally-absent features stay NaN instead of being zero-filled.
    """
    sc = StandardScaler().fit(train_df[md.UNIFIED_FEATURES].values.astype(np.float32))
    Xtr = sc.transform(train_df[md.UNIFIED_FEATURES].values.astype(np.float32))
    clf = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.3,
                        tree_method="hist", eval_metric="logloss",
                        random_state=RS, n_jobs=-1)
    clf.fit(Xtr, train_df.y.values)
    return sc, clf


def _eval(sc, clf, test_df):
    """Full metric set for one (model, test set) pair."""
    X = sc.transform(test_df[md.UNIFIED_FEATURES].values.astype(np.float32))
    p = clf.predict(X)
    y = test_df.y.values
    both = len(np.unique(y)) > 1          # AUC/AP undefined on a single class

    if both:
        try:
            prob = clf.predict_proba(X)[:, 1]
        except Exception:
            prob = p.astype(float)
        auc = roc_auc_score(y, prob)
        ap = average_precision_score(y, prob)
    else:
        auc = ap = float("nan")

    tf1 = trivial_f1(y)
    f1 = f1_score(y, p, zero_division=0)
    out = {
        "roc_auc": auc,
        "ap": ap,
        "mcc": matthews_corrcoef(y, p) if both else float("nan"),
        "bal_acc": balanced_accuracy_score(y, p) if both else float("nan"),
        "f1": f1,
        "f1_trivial": tf1,
        "f1_lift": f1 - tf1,
        "acc": accuracy_score(y, p),
        "recall": recall_score(y, p, zero_division=0),
        "prevalence": float(np.mean(y)),
    }
    neg = y == 0
    out["fpr"] = float((p[neg] == 1).mean()) if neg.any() else float("nan")
    return out


def _heatmap(M, title, path, cmap="RdYlGn", vmin=0.0, vmax=1.0, midline=None):
    names = list(M.index)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.0 * len(names) + 1.5))
    im = ax.imshow(M.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("TEST dataset"); ax.set_ylabel("TRAIN dataset")
    ax.set_title(title, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black")
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    if midline is not None:
        cb.ax.axhline(midline, color="black", lw=1.4)
        cb.ax.text(1.6, midline, " chance", fontsize=7, va="center",
                   transform=cb.ax.get_yaxis_transform())
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def nxn_matrix(data, results_dir):
    names = list(data)
    fitted = {n: _fit(data[n]) for n in names}
    print("  trained per-dataset models")

    long_rows = []
    for tr in names:
        sc, clf = fitted[tr]
        for te in names:
            m = _eval(sc, clf, data[te])
            long_rows.append({"train": tr, "test": te, "in_domain": tr == te, **m})
    long = pd.DataFrame(long_rows)
    long_path = os.path.join(results_dir, "cross_dataset_metrics_long.csv")
    long.round(4).to_csv(long_path, index=False)

    mats = {}
    for metric, fname, title, cmap, vmin, vmax, mid in [
        ("roc_auc", "cross_dataset_auc_matrix",
         "Cross-dataset ROC-AUC (0.5 = no signal; diagonal = in-domain)",
         "RdYlGn", 0.0, 1.0, 0.5),
        ("f1", "cross_dataset_matrix",
         "Cross-dataset binary F1 (diagonal = in-domain)",
         "RdYlGn", 0.0, 1.0, None),
        ("f1_lift", "cross_dataset_lift_matrix",
         "F1 minus trivial all-attack baseline (<= 0 means degenerate)",
         "RdYlGn", -0.7, 0.7, None),
    ]:
        M = long.pivot(index="train", columns="test", values=metric).loc[names, names]
        M.round(3).to_csv(os.path.join(results_dir, fname + ".csv"))
        _heatmap(M.round(3), title, os.path.join(results_dir, fname + ".png"),
                 cmap=cmap, vmin=vmin, vmax=vmax, midline=mid)
        mats[metric] = M
    print(f"  wrote AUC / F1 / lift matrices and {os.path.basename(long_path)}")

    # ── summary ───────────────────────────────────────────────────────────────
    off = long[~long.in_domain]
    dia = long[long.in_domain]
    print("\n  " + "-" * 62)
    print(f"  {'metric':<12}{'in-domain':>12}{'cross-domain':>14}{'gap':>10}{'chance':>10}")
    print("  " + "-" * 62)
    for metric, chance in [("roc_auc", "0.500"), ("ap", "= prev"), ("mcc", "0.000"),
                           ("bal_acc", "0.500"), ("f1", "= 2p/(1+p)")]:
        d, o = dia[metric].mean(), off[metric].mean()
        print(f"  {metric:<12}{d:>12.3f}{o:>14.3f}{d-o:>10.3f}{chance:>10}")
    print("  " + "-" * 62)

    n_degen = int((off.f1_lift <= 0).sum())
    print(f"\n  off-diagonal cells at or below the trivial all-attack baseline: "
          f"{n_degen}/{len(off)} ({n_degen/len(off)*100:.0f}%)")
    n_chance = int((off.roc_auc <= 0.55).sum())
    print(f"  off-diagonal cells with ROC-AUC <= 0.55 (no usable signal)      : "
          f"{n_chance}/{len(off)} ({n_chance/len(off)*100:.0f}%)")
    print("\n  Read the AUC matrix first: it separates 'no signal transfers' from")
    print("  'signal transfers but the decision threshold moved'.")
    return long


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
    print(f"    {'held out':<18}{'AUC':>7}{'AP':>7}{'MCC':>7}{'balAcc':>8}"
          f"{'F1':>7}{'triv':>7}{'lift':>7}{'FPR':>7}")
    for n in held:
        m = _eval(sc, clf, data[n])
        rows.append({"held_out": n, **{k: round(v, 4) for k, v in m.items()}})
        print(f"    {n:<18}{m['roc_auc']:>7.3f}{m['ap']:>7.3f}{m['mcc']:>7.3f}"
              f"{m['bal_acc']:>8.3f}{m['f1']:>7.3f}{m['f1_trivial']:>7.3f}"
              f"{m['f1_lift']:>+7.3f}{m['fpr']:>7.3f}")
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
    print("\n=== Experiment 1: NxN cross-dataset transfer matrix ===")
    nxn_matrix(data, RESULTS)
    print("\n=== Experiment 2: pooled-train, held-out test ===")
    pooled_heldout(data, args.train.split(","),
                   os.path.join(RESULTS, "cross_dataset_heldout.csv"))


if __name__ == "__main__":
    main()
