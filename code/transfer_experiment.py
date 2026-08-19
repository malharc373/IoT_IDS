"""
transfer_experiment.py — can a *deployable* feature transform close the
cross-dataset generalization gap?

Baseline (StandardScaler on the raw 12 features) gave leave-one-dataset-out
(LODO) transfer far below in-domain. Network-flow features are heavy-tailed, so
a scaler fit on one domain does not match another. We test fixed transforms
(fit on the training pool, applied unchanged at inference):

  raw_standard      StandardScaler on raw 12                (baseline)
  log_standard      signed log1p then StandardScaler
  log_robust        signed log1p then RobustScaler (median/IQR)
  ratios_standard   7 dimensionless ratio features only
  ratios_log        ratios + signed log1p(raw 12), StandardScaler  (combined)
  quantile          QuantileTransformer -> normal
  log_quantile      signed log1p then QuantileTransformer -> normal

Exportability: the scaler-based transforms (raw/log/ratio + Standard/Robust)
are affine and export cleanly to ONNX and to the C header. The two
QuantileTransformer variants are ONNX-exportable but carry a 1000-knot lookup
per feature, which is not realistic for the MCU path — they are included as a
research upper bound, not as a deployable option.

Protocol: leave-one-dataset-out. Train on all-but-one, test on the held-out
dataset; average every metric across held-outs. No random split of a merged
corpus. Metrics follow vault/Findings/F02: threshold-free AUC/AP are primary,
and any F1 is reported against the trivial all-attack baseline.

NaN handling: features a dataset structurally cannot supply arrive as NaN and
are kept as NaN all the way to XGBoost, which learns a default split direction
for them. Zero-filling would turn "absent" into a constant that identifies the
source dataset (vault/Findings/F12).

Output: demo/results/transfer_comparison.csv
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
import collections
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import multidataset as md  # noqa: E402
from cross_dataset_eval import balanced, trivial_f1  # noqa: E402

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
    # inf is a division artifact and is not meaningful; NaN means "absent"
    # and is preserved for XGBoost to handle natively (see F12).
    return np.where(np.isinf(out), np.nan, out)


TRANSFORMS = ["raw_standard", "log_standard", "log_robust",
              "ratios_standard", "ratios_log", "quantile", "log_quantile"]


def make_features(name, Xraw, fitted=None):
    """Return (features, fitted_state). fitted_state is reused for test.

    NaN is preserved throughout: signed_log, StandardScaler, RobustScaler and
    QuantileTransformer all disregard NaN when fitting and maintain it when
    transforming, and XGBoost consumes it natively.
    """
    X = Xraw.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
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
    """Leave-one-dataset-out. Returns the mean of every metric across held-outs.

    Reported threshold-free (AUC, AP) as well as thresholded (F1, MCC,
    balanced accuracy) — under domain shift a transform can improve the
    *ranking* without moving F1, or move F1 purely by shifting the operating
    point. See vault/Findings/F02.
    """
    names = list(data)
    acc = collections.defaultdict(list)
    for held in names:
        train = pd.concat([data[n] for n in names if n != held], ignore_index=True)
        Xtr, st = make_features(transform, train[F].values)
        clf = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.3,
                            tree_method="hist", eval_metric="logloss",
                            random_state=RS, n_jobs=-1)
        clf.fit(Xtr, train.y.values)
        Xte, _ = make_features(transform, data[held][F].values, fitted=st)
        p = clf.predict(Xte)
        y = data[held].y.values
        both = len(np.unique(y)) > 1
        prob = clf.predict_proba(Xte)[:, 1]

        acc["f1"].append(f1_score(y, p, zero_division=0))
        acc["f1_trivial"].append(trivial_f1(y))
        acc["recall"].append(recall_score(y, p, zero_division=0))
        acc["roc_auc"].append(roc_auc_score(y, prob) if both else np.nan)
        acc["ap"].append(average_precision_score(y, prob) if both else np.nan)
        acc["mcc"].append(matthews_corrcoef(y, p) if both else np.nan)
        acc["bal_acc"].append(balanced_accuracy_score(y, p) if both else np.nan)
        neg = y == 0
        acc["benign_fpr"].append(float((p[neg] == 1).mean()) if neg.any() else np.nan)
    out = {k: float(np.nanmean(v)) for k, v in acc.items()}
    out["f1_lift"] = out["f1"] - out["f1_trivial"]
    # LODO averages a handful of folds whose difficulty varies enormously, so a
    # bare mean hides whether a "lift" is a real effect or one lucky held-out
    # dataset. Report the spread and the worst fold alongside it.
    out["roc_auc_std"] = float(np.nanstd(acc["roc_auc"]))
    out["roc_auc_min"] = float(np.nanmin(acc["roc_auc"]))
    out["n_folds_above_chance"] = int(sum(1 for v in acc["roc_auc"] if v > 0.55))
    out["n_folds"] = len(acc["roc_auc"])
    return out


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
    print(f"{'transform':<18}{'AUC':>8}{'±sd':>7}{'worst':>7}{'>chance':>9}"
          f"{'MCC':>8}{'F1':>8}{'lift':>8}")
    print("-" * 74)
    for t in TRANSFORMS:
        m = lodo(data, t)
        rows.append({"transform": t, **{k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in m.items()}})
        print(f"{t:<18}{m['roc_auc']:>8.3f}{m['roc_auc_std']:>7.3f}"
              f"{m['roc_auc_min']:>7.3f}"
              f"{m['n_folds_above_chance']:>6}/{m['n_folds']:<2}"
              f"{m['mcc']:>8.3f}{m['f1']:>8.3f}{m['f1_lift']:>+8.3f}")
    dfres = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    dfres.to_csv(os.path.join(RESULTS, "transfer_comparison.csv"), index=False)

    best = dfres.iloc[0]
    base = dfres[dfres["transform"] == "raw_standard"].iloc[0]
    print("-" * 74)
    print(f"\nRanked by ROC-AUC (threshold-free). Chance = 0.500, MCC chance = 0.000.")
    lift = best["roc_auc"] - base["roc_auc"]
    print(f"best: {best['transform']} AUC={best['roc_auc']:.3f} "
          f"(sd {best['roc_auc_std']:.3f} across {int(best['n_folds'])} folds, "
          f"worst fold {best['roc_auc_min']:.3f}) "
          f"vs baseline raw_standard AUC={base['roc_auc']:.3f} (lift {lift:+.3f})")
    if lift < best["roc_auc_std"]:
        print("NOTE: the lift is smaller than the fold-to-fold spread — it is "
              "not distinguishable from which datasets happen to be held out.")
    if best["n_folds_above_chance"] <= best["n_folds"] // 2:
        print(f"NOTE: only {int(best['n_folds_above_chance'])} of "
              f"{int(best['n_folds'])} held-out folds clear AUC 0.55, so the "
              f"mean is carried by a minority of easy targets.")
    if best["f1_lift"] <= 0:
        print("WARNING: the best transform's F1 is at or below the trivial "
              "all-attack baseline — treat any F1-based claim as degenerate.")
    print(f"wrote {os.path.join(RESULTS, 'transfer_comparison.csv')}")


if __name__ == "__main__":
    main()
