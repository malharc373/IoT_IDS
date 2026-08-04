"""
transfer_experiment.py — can a *deployable* feature transform close the
cross-dataset generalization gap?

Baseline (StandardScaler on the raw 12 features) gave leave-one-dataset-out
(LODO) transfer far below in-domain. Network-flow features are heavy-tailed and
reported in different units across datasets, so a scaler fit on one domain does
not match another. We test fixed transforms that are all exportable to ONNX/C
(fit on the training pool, applied unchanged at inference):

  raw_standard      StandardScaler on raw 12                (baseline)
  log_standard      log1p(|x|)·sign then StandardScaler
  log_robust        log1p then RobustScaler (median/IQR)
  ratios_standard   7 dimensionless ratio features only
  ratios_log        ratios + log1p(raw 12), StandardScaler   (combined)

Protocol: leave-one-dataset-out. Train on all-but-one, test on the held-out
dataset; average binary F1 / recall / benign-FPR across held-outs. No random
split of a merged corpus.

Output: demo/results/transfer_comparison.csv
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import multidataset as md  # noqa: E402
from cross_dataset_eval import balanced  # noqa: E402

RESULTS = os.path.join(ROOT, "demo", "results")
RS = 42
F = md.UNIFIED_FEATURES


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def ratio_features(X):
    """7 dimensionless (unit-independent) features from the 12 base features.
    Column order in F:
      0 Flow Duration 1 TotFwdPkts 2 TotBwdPkts 3 TotLenFwd 4 TotLenBwd
      5 FlowPkts/s 6 FwdPkts/s 7 BwdPkts/s 8 MinLen 9 MaxLen 10 MeanLen 11 StdLen
    """
    d = X.astype(np.float64)
    eps = 1.0
    fwd_p, bwd_p = d[:, 1], d[:, 2]
    fwd_b, bwd_b = d[:, 3], d[:, 4]
    fpps, bpps = d[:, 6], d[:, 7]
    mn, mx, mean, std = d[:, 8], d[:, 9], d[:, 10], d[:, 11]
    out = np.column_stack([
        fwd_p / (bwd_p + eps),               # fwd/bwd packet ratio
        fwd_b / (bwd_b + eps),               # fwd/bwd byte ratio
        fwd_b / (fwd_p + eps),               # bytes per fwd packet
        fwd_p / (fwd_p + bwd_p + eps),       # fwd packet fraction
        fpps / (bpps + eps),                 # fwd/bwd rate ratio
        (mx - mn) / (mean + eps),            # packet-length spread
        std / (mean + eps),                  # packet-length coeff. of variation
    ])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


TRANSFORMS = ["raw_standard", "log_standard", "log_robust",
              "ratios_standard", "ratios_log", "quantile", "log_quantile"]


def make_features(name, Xraw, fitted=None):
    """Return (features, fitted_state). fitted_state is reused for test."""
    X = np.nan_to_num(Xraw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if name == "raw_standard":
        base = X
    elif name == "log_standard":
        base = signed_log(X)
    elif name == "log_robust":
        base = signed_log(X)
    elif name == "ratios_standard":
        base = ratio_features(X)
    elif name == "ratios_log":
        base = np.column_stack([signed_log(X), ratio_features(X)])
    elif name == "quantile":
        base = X
    elif name == "log_quantile":
        base = signed_log(X)
    else:
        raise ValueError(name)
    if name in ("quantile", "log_quantile"):
        if fitted is None:
            sc = QuantileTransformer(output_distribution="normal",
                                     n_quantiles=1000, random_state=RS).fit(base)
            return sc.transform(base).astype(np.float32), sc
        return fitted.transform(base).astype(np.float32), fitted
    scaler_cls = RobustScaler if name == "log_robust" else StandardScaler
    if fitted is None:
        sc = scaler_cls().fit(base)
        return sc.transform(base).astype(np.float32), sc
    return fitted.transform(base).astype(np.float32), fitted


def lodo(data, transform):
    names = list(data)
    f1s, recs, fprs = [], [], []
    for held in names:
        train = pd.concat([data[n] for n in names if n != held], ignore_index=True)
        Xtr, st = make_features(transform, train[F].values)
        clf = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.3,
                            tree_method="hist", eval_metric="logloss",
                            random_state=RS, n_jobs=-1)
        clf.fit(Xtr, train.y.values)
        Xte, _ = make_features(transform, data[held][F].values, fitted=st)
        p = clf.predict(Xte); y = data[held].y.values
        f1s.append(f1_score(y, p, zero_division=0))
        recs.append(recall_score(y, p, zero_division=0))
        neg = y == 0
        fprs.append(float((p[neg] == 1).mean()) if neg.any() else np.nan)
    return np.mean(f1s), np.mean(recs), np.nanmean(fprs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=25_000)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    names = md.available()
    print(f"Loading {len(names)} datasets (cap {args.cap:,})...")
    data = {}
    for n in names:
        df = balanced(md.load(n), args.cap)
        if len(df) > 100 and df.y.nunique() >= 1:
            data[n] = df
    print(f"  {len(data)} datasets, {sum(len(d) for d in data.values()):,} rows\n")

    rows = []
    print(f"{'transform':<18}{'LODO F1':>9}{'recall':>9}{'benign FPR':>12}")
    for t in TRANSFORMS:
        f1, rec, fpr = lodo(data, t)
        rows.append({"transform": t, "lodo_f1": round(f1, 4),
                     "recall": round(rec, 4), "benign_fpr": round(fpr, 4)})
        print(f"{t:<18}{f1:>9.3f}{rec:>9.3f}{fpr:>12.3f}")
    dfres = pd.DataFrame(rows).sort_values("lodo_f1", ascending=False)
    dfres.to_csv(os.path.join(RESULTS, "transfer_comparison.csv"), index=False)
    best = dfres.iloc[0]
    base = dfres[dfres["transform"] == "raw_standard"].iloc[0]
    print(f"\nbest: {best['transform']} (F1={best['lodo_f1']}) vs "
          f"baseline raw_standard (F1={base['lodo_f1']})  "
          f"lift={best['lodo_f1']-base['lodo_f1']:+.3f}")
    print(f"wrote {os.path.join(RESULTS, 'transfer_comparison.csv')}")


if __name__ == "__main__":
    main()
