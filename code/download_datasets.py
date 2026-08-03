#!/usr/bin/env python3
"""
download_datasets.py — fetch the three IDS datasets for SFAF reproduction.

The datasets are large and gated behind registration, so full automation needs
a Kaggle API token.  This script uses Kaggle if configured, and otherwise
prints exact manual instructions and the expected directory layout.

Target layout (relative to repo root):

    Datasets/
      MachineLearningCVE/*.csv                             CICIDS2017
      UNSWNB15/UNSW_NB15_training-set.parquet
      UNSWNB15/UNSW_NB15_testing-set.parquet
      TONIoT/train_test_network.csv

Kaggle setup (one-time):
    pip install kaggle
    # create a token at https://www.kaggle.com/settings -> "Create New Token"
    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    python code/download_datasets.py
"""
from __future__ import annotations

import os
import sys
import glob
import subprocess

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DS = os.path.join(BASE, "Datasets")

# (kaggle dataset slug, subdir, note)
KAGGLE_SOURCES = [
    ("cicdataset/cicids2017", "MachineLearningCVE",
     "CICIDS2017 — use the MachineLearningCVE CSV folder"),
    ("mrwellsdavid/unsw-nb15", "UNSWNB15",
     "UNSW-NB15 — training/testing set (convert CSV->parquet if needed)"),
    ("dhoogla/toniotnetwork", "TONIoT",
     "TON-IoT — train_test_network.csv"),
]

MANUAL = """
Manual download (no Kaggle token):

  CICIDS2017  https://www.unb.ca/cic/datasets/ids-2017.html
              -> download "MachineLearningCVE" CSVs
              -> Datasets/MachineLearningCVE/*.csv

  UNSW-NB15   https://research.unsw.edu.au/projects/unsw-nb15-dataset
              -> UNSW_NB15_training-set.csv / testing-set.csv
              -> save as parquet:  Datasets/UNSWNB15/UNSW_NB15_{training,testing}-set.parquet

  TON-IoT     https://research.unsw.edu.au/projects/toniot-datasets
              -> Processed/Network -> train_test_network.csv
              -> Datasets/TONIoT/train_test_network.csv

Then run:  python code/02_train_sfaf.py
"""


def have_kaggle():
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        return False
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:
        return subprocess.call(["which", "kaggle"],
                               stdout=subprocess.DEVNULL) == 0


def check_layout():
    ok = True
    checks = [
        (glob.glob(os.path.join(DS, "MachineLearningCVE", "*.csv")), "CICIDS2017 CSVs"),
        (glob.glob(os.path.join(DS, "UNSWNB15", "*.parquet")), "UNSW-NB15 parquet"),
        (glob.glob(os.path.join(DS, "TONIoT", "train_test_network.csv")), "TON-IoT csv"),
    ]
    for found, name in checks:
        mark = "✓" if found else "✗"
        print(f"  {mark} {name}: {len(found)} file(s)")
        ok = ok and bool(found)
    return ok


def kaggle_fetch():
    for slug, subdir, note in KAGGLE_SOURCES:
        dest = os.path.join(DS, subdir)
        os.makedirs(dest, exist_ok=True)
        print(f"\n[kaggle] {slug} -> {dest}\n         {note}")
        rc = subprocess.call([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", slug, "-p", dest, "--unzip",
        ])
        if rc != 0:
            print(f"[warn] kaggle download failed for {slug} (rc={rc})")


def main():
    os.makedirs(DS, exist_ok=True)
    print("Checking dataset layout under Datasets/ ...")
    if check_layout():
        print("\nAll datasets present. Run: python code/02_train_sfaf.py")
        return
    if have_kaggle():
        print("\nKaggle token found — attempting download ...")
        kaggle_fetch()
        print("\nRe-checking layout:")
        if check_layout():
            print("\nReady. Run: python code/02_train_sfaf.py")
            return
        print("\nSome datasets still missing — see manual steps below.")
    else:
        print("\nNo Kaggle token configured.")
    print(MANUAL)


if __name__ == "__main__":
    main()
