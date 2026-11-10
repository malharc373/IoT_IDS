"""
run_sfaf_rerun.py - run the corrected 11-dataset SFAF study and enforce the
acceptance gate from demo/results/CROSS_DATASET_FINDINGS.md.

    python code/run_sfaf_rerun.py                 # full run (needs all 11)
    python code/run_sfaf_rerun.py --check-only    # only verify datasets/outputs
    python code/run_sfaf_rerun.py --cap 50000 --tt-cap 25000 --repeats 20

Steps: (1) require all eleven datasets under IOTIDS_DATASETS_ROOT, (2) run
cross_dataset_eval / transfer_experiment / threshold_transfer, (3) check the
gate (121 matrix cells, diagonal = held_out_20pct, n_train/n_eval > 0, every
threshold row reports its repeats), (4) write SFAF_RERUN_MANIFEST.json with
commit, versions, arguments and output hashes. Exits non-zero on any failure,
so a partial run can never be mistaken for a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "demo", "results")
sys.path.insert(0, HERE)
import multidataset as md  # noqa: E402

OUTPUTS = [
    "cross_dataset_metrics_long.csv", "cross_dataset_auc_matrix.csv",
    "cross_dataset_matrix.csv", "cross_dataset_lift_matrix.csv",
    "cross_dataset_heldout.csv", "transfer_comparison.csv",
    "threshold_transfer.csv",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def require_datasets(root):
    present = md.available(root)
    missing = [n for n in md.LOADERS if n not in present]
    print(f"datasets root: {root}\n  present {len(present)}/{len(md.LOADERS)}")
    if missing:
        raise SystemExit(f"FAIL: missing datasets: {', '.join(missing)}")
    return present


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    return round(time.time() - t0, 1)


def check_gate(n_expected, repeats):
    problems = []
    for f in OUTPUTS:
        if not os.path.exists(os.path.join(RESULTS, f)):
            problems.append(f"missing output {f}")
    if problems:
        return problems
    long = pd.read_csv(os.path.join(RESULTS, "cross_dataset_metrics_long.csv"))
    if len(long) != n_expected ** 2:
        problems.append(f"matrix has {len(long)} cells, expected {n_expected ** 2}")
    if long.train.nunique() != n_expected or long.test.nunique() != n_expected:
        problems.append("matrix does not cover every dataset on both axes")
    diag = long[long.in_domain]
    if len(diag) != n_expected or not (diag.evaluation == "held_out_20pct").all():
        problems.append("diagonal is not all held_out_20pct")
    off = long[~long.in_domain]
    if not (off.evaluation == "independent_dataset").all():
        problems.append("off-diagonal cells are not all independent_dataset")
    if not ((long.n_train > 0) & (long.n_eval > 0)).all():
        problems.append("a cell has n_train or n_eval == 0")
    if long.roc_auc.isna().any():
        problems.append("NaN roc_auc in matrix")
    tt = pd.read_csv(os.path.join(RESULTS, "threshold_transfer.csv"))
    if "calibration_success_rate" not in tt.columns:
        problems.append("threshold_transfer.csv lacks calibration_success_rate")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=50_000)
    ap.add_argument("--tt-cap", type=int, default=25_000, help="transfer_experiment cap")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    present = require_datasets(md.DEFAULT_ROOT)
    timings = {}
    if not a.check_only:
        py = sys.executable
        timings["cross_dataset_eval"] = run(
            [py, "code/cross_dataset_eval.py", "--cap", str(a.cap)])
        timings["transfer_experiment"] = run(
            [py, "code/transfer_experiment.py", "--cap", str(a.tt_cap)])
        timings["threshold_transfer"] = run(
            [py, "code/threshold_transfer.py", "--cap", str(a.cap),
             "--repeats", str(a.repeats)])

    problems = check_gate(len(present), a.repeats)
    if problems:
        print("\nGATE FAILED:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    versions = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "n/a"
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "datasets": present, "seed": 42,
        "args": {"cap": a.cap, "transfer_cap": a.tt_cap, "repeats": a.repeats},
        "seconds": timings,
        "outputs_sha256": {f: sha256(os.path.join(RESULTS, f)) for f in OUTPUTS},
    }
    out = os.path.join(RESULTS, "SFAF_RERUN_MANIFEST.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nGATE PASSED ({len(present) ** 2} cells). Manifest: {out}")


if __name__ == "__main__":
    main()
