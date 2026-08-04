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

# (kaggle dataset slug, subdir, note) — VERIFIED working refs (small CSV/parquet
# mirrors preferred: Kaggle stalls on very large single-zip downloads >~150MB,
# so use per-file or a cleaned mirror for the big ones).
KAGGLE_SOURCES = [
    # -- core SFAF datasets --
    ("cicdataset/cicids2017", "MachineLearningCVE",
     "CICIDS2017 — MachineLearningCVE CSVs"),
    ("mrwellsdavid/unsw-nb15", "UNSWNB15",
     "UNSW-NB15 — training/testing set"),
    ("dhoogla/toniotnetwork", "TONIoT", "TON-IoT — train_test_network.csv"),
    # -- IoT-specific additions (flow-level, good fit for the edge model) --
    ("soulchain1/ciciot2023", "CICIoT2023",
     "CIC-IoT-2023 (cleaned CSVs: benign/DDoS/Dos/mirai/Recon/spoofing)"),
    ("ogunyemioluwapelumi/mqtt-iot-ids2020-private", "MQTT_IoT_IDS2020",
     "MQTT-IoT-IDS2020 — bidirectional-flow CSVs (mqtt brute/scan/sparta)"),
    ("sibasispradhan/edge-iiotset-dataset", "EdgeIIoTset",
     "Edge-IIoTSet — use -f ML-EdgeIIoT-dataset.csv (82MB, avoids 1.2GB raw)"),
    # -- external / held-out test datasets --
    ("rohulaminlabid/iotid20-dataset", "IoTID20", "IoTID20 — external test"),
    ("munaalhawawreh/xiiotid-iiot-intrusion-dataset", "X-IIoTID",
     "X-IIoTID — IIoT, hierarchical labels, external test"),
    ("dhoogla/cicddos2019", "CICDDoS2019",
     "CIC-DDoS2019 — cleaned parquet (DNS/LDAP/NTP/SNMP reflection), external test"),
]

# Datasets with an auth-free direct URL (no Kaggle token needed).
# (url, subdir, note, extra_wget_flags)
DIRECT_SOURCES = [
    ("https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz",
     "IoT23", "IoT-23 — real IoT malware Zeek conn logs (~8.8GB, slow host)", ""),
    ("https://www.cse.wustl.edu/~jain/iiot2/ftp/wustl_iiot_2021.zip",
     "WUSTL_IIoT_2021", "WUSTL-IIoT-2021 — industrial control CSV",
     "--no-check-certificate"),
]

# Already-present but different modality (NOT in the 12-feature flow space):
#   NBaIoT  — 115 Kitsune per-packet statistical features (separate model)
#   NSLKDD  — legacy KDD features (optional legacy baseline)

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

  CIC-IoT-23  https://www.unb.ca/cic/datasets/iotdataset-2023.html
              -> merged CSVs -> Datasets/CICIoT2023/*.csv

  Bot-IoT     https://research.unsw.edu.au/projects/bot-iot-dataset
              -> UNSW_2018_IoT_Botnet CSVs -> Datasets/BotIoT/*.csv

Column alignment for all five datasets lives in code/dataset_maps.py.
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
    # Sequential — Kaggle downloads stall when several run in parallel on the
    # same (esp. exFAT) volume, and hang on very large single-zip archives.
    for slug, subdir, note in KAGGLE_SOURCES:
        dest = os.path.join(DS, subdir)
        os.makedirs(dest, exist_ok=True)
        print(f"\n[kaggle] {slug} -> {dest}\n         {note}")
        rc = subprocess.call([
            "kaggle", "datasets", "download",
            "-d", slug, "-p", dest, "--unzip",
        ])
        if rc != 0:
            print(f"[warn] kaggle download failed for {slug} (rc={rc})")


def direct_fetch():
    """Fetch datasets that have an auth-free direct URL (wget, resumable),
    then extract any downloaded archive so the loaders can see the files."""
    for url, subdir, note, flags in DIRECT_SOURCES:
        dest = os.path.join(DS, subdir)
        os.makedirs(dest, exist_ok=True)
        print(f"\n[direct] {url}\n         {note}")
        cmd = ["wget", "-c", "--tries=20", "--timeout=60"]
        if flags:
            cmd += flags.split()
        cmd += ["-P", dest, url]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[warn] direct download failed for {subdir} (rc={rc}); "
                  f"the IoT-23 host is slow — the download is resumable, re-run.")
            continue
        _extract_archives(dest)


def _extract_archives(dest):
    """Extract .tar.gz / .zip found in dest (idempotent-ish)."""
    import tarfile, zipfile
    for arc in glob.glob(os.path.join(dest, "*.tar.gz")) + \
               glob.glob(os.path.join(dest, "*.tgz")):
        try:
            print(f"  extracting {os.path.basename(arc)} ...")
            with tarfile.open(arc) as tf:
                tf.extractall(dest)
        except Exception as e:
            print(f"  [warn] extract failed: {e}")
    for arc in glob.glob(os.path.join(dest, "*.zip")):
        try:
            print(f"  extracting {os.path.basename(arc)} ...")
            with zipfile.ZipFile(arc) as zf:
                zf.extractall(dest)
        except Exception as e:
            print(f"  [warn] extract failed: {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch IDS datasets for SFAF")
    ap.add_argument("--direct", action="store_true",
                    help="also fetch auth-free direct-URL datasets (IoT-23, WUSTL)")
    args = ap.parse_args()

    os.makedirs(DS, exist_ok=True)
    print("Checking dataset layout under Datasets/ ...")
    if check_layout() and not args.direct:
        print("\nCore datasets present. Run: python code/02_train_sfaf.py")
        return
    if have_kaggle():
        print("\nKaggle token found — attempting downloads (sequential) ...")
        kaggle_fetch()
    else:
        print("\nNo Kaggle token configured.")
        print(MANUAL)
    if args.direct:
        direct_fetch()
    print("\nRe-checking core layout:")
    if check_layout():
        print("\nReady. Run: python code/02_train_sfaf.py")
    else:
        print("\nSome core datasets still missing — see manual steps above.")


if __name__ == "__main__":
    main()
