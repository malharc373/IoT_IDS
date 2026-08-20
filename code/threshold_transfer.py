"""
threshold_transfer.py — how many labelled target flows buy back the gap?

MOTIVATION (see vault/Experiments/EXP02)
-----------------------------------------
The cross-dataset study found that transfer between arbitrary dataset pairs is
at chance. But the pooled-training experiment showed something the F1-only view
had hidden: on some held-out datasets the model's *ranking* survives while only
its *decision threshold* fails.

    held out              AUC     F1
    cicddos2019         0.927  0.561      <- ranking transfers, threshold does not
    mqtt_iot_ids2020    0.987  0.943      <- everything transfers
    iotid20             0.606  0.135      <- nothing transfers
    wustl_iiot          0.314  0.337      <- transfers inverted (worse than chance)

Those are three different problems wearing one number. Only the first is cheap
to fix, and this script measures exactly how cheap: if a handful of labelled
flows from the target domain recover most of the AUC-implied performance, then
"calibrate on a small labelled sample" is a deployable answer for that regime —
far cheaper than domain adaptation.

METHOD
------
For each held-out dataset:

  1. pool-train a detector on the training datasets (never sees the target),
  2. score the target domain,
  3. draw a labelled calibration sample of n flows, fit ONLY a scalar threshold
     on it, and evaluate on the *remaining* target flows,
  4. repeat over seeds and report the mean.

Three reference lines bound the result:

  default    threshold 0.5, i.e. what the deployed model does today
  oracle     the best threshold fitted on the ENTIRE target set — unreachable
             in practice, and the ceiling that ranking quality alone permits
  trivial    what "everything is an attack" scores, 2p/(1+p)

A single scalar is the only thing fitted, so nothing here can smuggle in target
information beyond the operating point. That is the point: it isolates how much
of the gap is calibration rather than representation.

THE TRAP THIS EXPERIMENT CAN FALL INTO
---------------------------------------
On a domain where the ranking carries no signal, fitting a threshold to
maximise F1 simply *rediscovers the trivial all-attack classifier* — the
optimum is a threshold below every score. The run then reports "recovered 99%
of the calibration gap" while having learned nothing, which is exactly the
degeneracy vault/Findings/F02 was written about.

So the result is only meaningful where `oracle_lift = oracle_f1 - trivial_f1`
is clearly positive. Every row carries that column, and the summary refuses to
claim a win without it.

Output: demo/results/threshold_transfer.csv
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import multidataset as md  # noqa: E402
from cross_dataset_eval import load_all, _fit, trivial_f1  # noqa: E402

RESULTS = os.path.join(ROOT, "demo", "results")
RS = 42
BUDGETS = [10, 25, 50, 100, 250, 500, 1000]


def best_threshold(y, prob, grid=201):
    """Threshold maximising F1 on (y, prob). Returns (threshold, f1)."""
    if len(np.unique(y)) < 2:
        return 0.5, 0.0
    # candidate thresholds spread over the observed score range
    lo, hi = float(np.min(prob)), float(np.max(prob))
    if hi <= lo:
        return 0.5, f1_score(y, (prob >= 0.5).astype(int), zero_division=0)
    cands = np.linspace(lo, hi, grid)
    scores = [f1_score(y, (prob >= t).astype(int), zero_division=0) for t in cands]
    i = int(np.argmax(scores))
    return float(cands[i]), float(scores[i])


def _metrics(y, prob, thr):
    p = (prob >= thr).astype(int)
    return {
        "f1": f1_score(y, p, zero_division=0),
        "mcc": matthews_corrcoef(y, p) if len(np.unique(y)) > 1 else float("nan"),
    }


def study(data, train_names, budgets, repeats, out_csv):
    train_names = [n for n in train_names if n in data]
    held = [n for n in data if n not in train_names]
    if not train_names or not held:
        print("  (need >=1 train and >=1 held-out dataset present)")
        return

    train_df = pd.concat([data[n] for n in train_names], ignore_index=True)
    sc, clf = _fit(train_df)
    print(f"\n  pooled train on {train_names} (n={len(train_df):,})\n")

    rows = []
    for name in held:
        df = data[name]
        X = sc.transform(df[md.UNIFIED_FEATURES].values.astype(np.float32))
        prob = clf.predict_proba(X)[:, 1]
        y = df.y.values
        both = len(np.unique(y)) > 1
        auc = roc_auc_score(y, prob) if both else float("nan")

        default = _metrics(y, prob, 0.5)
        _, oracle_f1 = best_threshold(y, prob)
        triv = trivial_f1(y)
        oracle_lift = oracle_f1 - triv
        degenerate = oracle_lift <= 0.02

        flag = "  <-- DEGENERATE: best threshold == all-attack" if degenerate else ""
        print(f"  {name}   AUC={auc:.3f}  default F1={default['f1']:.3f}  "
              f"oracle F1={oracle_f1:.3f}  (trivial {triv:.3f}, "
              f"oracle lift {oracle_lift:+.3f}){flag}")

        rows.append({"held_out": name, "n_labels": 0, "auc": round(auc, 4),
                     "f1": round(default["f1"], 4), "mcc": round(default["mcc"], 4),
                     "oracle_f1": round(oracle_f1, 4), "trivial_f1": round(triv, 4),
                     "oracle_lift": round(oracle_lift, 4),
                     "degenerate": bool(degenerate),
                     "recovered": 0.0, "note": "default threshold 0.5"})

        rng = np.random.RandomState(RS)
        for n in budgets:
            if n >= len(y):
                continue
            f1s, mccs = [], []
            for _ in range(repeats):
                idx = rng.choice(len(y), size=n, replace=False)
                mask = np.zeros(len(y), dtype=bool); mask[idx] = True
                if len(np.unique(y[mask])) < 2:
                    continue          # calibration sample saw one class only
                thr, _ = best_threshold(y[mask], prob[mask])
                m = _metrics(y[~mask], prob[~mask], thr)
                f1s.append(m["f1"]); mccs.append(m["mcc"])
            if not f1s:
                continue
            f1m = float(np.mean(f1s))
            # fraction of the default->oracle gap that this budget recovers
            span = oracle_f1 - default["f1"]
            rec = (f1m - default["f1"]) / span if span > 1e-9 else float("nan")
            rows.append({"held_out": name, "n_labels": n, "auc": round(auc, 4),
                         "f1": round(f1m, 4), "mcc": round(float(np.nanmean(mccs)), 4),
                         "oracle_f1": round(oracle_f1, 4),
                         "trivial_f1": round(triv, 4),
                         "oracle_lift": round(oracle_lift, 4),
                         "degenerate": bool(degenerate),
                         "recovered": round(rec, 4), "note": ""})
            note = " (to a degenerate optimum)" if degenerate else ""
            print(f"      {n:>5} labels -> F1 {f1m:.3f}  "
                  f"({rec*100:5.1f}% of the calibration gap){note}")
        print()

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    # headline: the smallest budget recovering >=80% of the gap, per dataset —
    # but only where the target of that recovery beats the trivial classifier
    print("\n  labels needed to recover 80% of the calibration gap:")
    for name in held:
        sub = out[(out.held_out == name) & (out.n_labels > 0)]
        if not len(sub):
            continue
        auc = sub.auc.iloc[0]
        if bool(sub.degenerate.iloc[0]):
            print(f"    {name:<18} AUC={auc:.3f}  ->  n/a: the best achievable "
                  f"threshold IS the all-attack classifier "
                  f"(oracle lift {sub.oracle_lift.iloc[0]:+.3f}) — "
                  f"nothing to calibrate toward")
            continue
        hit = sub[sub.recovered >= 0.8]
        if len(hit):
            print(f"    {name:<18} AUC={auc:.3f}  ->  {int(hit.n_labels.iloc[0])} "
                  f"labels  (F1 {sub.f1.iloc[0]:.3f} at 10 -> oracle "
                  f"{sub.oracle_f1.iloc[0]:.3f}, from default "
                  f"{out[(out.held_out == name) & (out.n_labels == 0)].f1.iloc[0]:.3f})")
        else:
            print(f"    {name:<18} AUC={auc:.3f}  ->  not reached within "
                  f"{max(budgets)} labels")

    real = out[(out.n_labels > 0) & (~out.degenerate)].held_out.unique()
    print(f"\n  Calibration is a real fix on {len(real)}/{len(held)} held-out "
          f"datasets: {', '.join(real) if len(real) else 'none'}.")
    print("  Everywhere else the ranking carries too little signal for an")
    print("  operating point to rescue — that needs domain adaptation, not labels.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=50_000)
    ap.add_argument("--train", default="cicids2017,unsw_nb15,ton_iot,bot_iot,cic_iot_2023")
    ap.add_argument("--repeats", type=int, default=20,
                    help="calibration samples drawn per budget (default 20)")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    names = md.available()
    print(f"Loading {len(names)} datasets (cap {args.cap:,}/dataset)...")
    data = load_all(names, args.cap)
    study(data, args.train.split(","), BUDGETS, args.repeats,
          os.path.join(RESULTS, "threshold_transfer.csv"))


if __name__ == "__main__":
    main()
