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

Direct-source archives retain TLS verification, print their SHA-256, and are
not extracted without either a trusted ``--sha256 SUBDIR=HEX`` value or an
explicit ``--allow-unverified`` risk acknowledgement. All archives are checked
for path traversal, links, and special files before extraction.
"""
from __future__ import annotations

import os
import sys
import glob
import stat
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from urllib.parse import urlparse

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DS = os.environ.get("IOTIDS_DATASETS_ROOT") or os.path.join(BASE, "Datasets")


def ensure_dataset_root(path=None):
    """Create the dataset root, tolerating the external-drive layout.

    `Datasets/` is normally a symlink to an external volume. When that volume
    is not mounted the symlink dangles, and `os.makedirs(..., exist_ok=True)`
    still raises FileExistsError — `exist_ok` only suppresses the error when
    the target is an existing *directory*, and a broken symlink is not one.
    The entry point in the README then failed with a bare traceback instead of
    saying which drive to plug in (vault/Findings/F21).
    """
    path = path or DS
    if os.path.islink(path) and not os.path.exists(path):
        sys.exit(
            f"[ERROR] {path} is a symlink to {os.readlink(path)}, which is not "
            f"mounted.\n        Mount that volume, or point the pipeline "
            f"elsewhere with IOTIDS_DATASETS_ROOT=/path/to/datasets."
        )
    if os.path.exists(path) and not os.path.isdir(path):
        sys.exit(f"[ERROR] {path} exists and is not a directory — move it aside.")
    os.makedirs(path, exist_ok=True)

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
# (url, subdir, note, publisher_sha256_or_none)
DIRECT_SOURCES = [
    ("https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz",
     "IoT23", "IoT-23 — real IoT malware Zeek conn logs (~8.8GB, slow host)", None),
    ("https://www.cse.wustl.edu/~jain/iiot2/ftp/wustl_iiot_2021.zip",
     "WUSTL_IIoT_2021", "WUSTL-IIoT-2021 — industrial control CSV", None),
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

Column alignment for every dataset lives in code/multidataset.py.
Then run:  python code/02_train_sfaf.py
"""


def have_kaggle():
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        return False
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:
        return shutil.which("kaggle") is not None


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
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        print("[ERROR] Kaggle credentials exist, but the kaggle executable is unavailable.")
        return
    for slug, subdir, note in KAGGLE_SOURCES:
        dest = os.path.join(DS, subdir)
        os.makedirs(dest, exist_ok=True)
        print(f"\n[kaggle] {slug} -> {dest}\n         {note}")
        rc = subprocess.call([
            kaggle_bin, "datasets", "download",
            "-d", slug, "-p", dest,
        ])
        if rc != 0:
            print(f"[warn] kaggle download failed for {slug} (rc={rc})")
        else:
            _extract_archives(dest)


def _sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(dest, member_name):
    """Resolve an archive member and reject absolute/traversal destinations."""
    if not member_name or os.path.isabs(member_name):
        raise ValueError(f"unsafe absolute/empty archive member: {member_name!r}")
    root = os.path.realpath(dest)
    target = os.path.realpath(os.path.join(root, member_name))
    if os.path.commonpath([root, target]) != root:
        raise ValueError(f"archive member escapes destination: {member_name!r}")
    return target


def _safe_extract_tar(path, dest):
    with tarfile.open(path) as tf:
        members = tf.getmembers()
        for member in members:
            _safe_target(dest, member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"archive special file is not allowed: {member.name!r}")
        for member in members:
            target = _safe_target(dest, member.name)
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f"archive file has no payload: {member.name!r}")
            with source:
                _atomic_extract_file(source, target)


def _safe_extract_zip(path, dest):
    with zipfile.ZipFile(path) as zf:
        members = zf.infolist()
        for member in members:
            _safe_target(dest, member.filename)
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type == stat.S_IFLNK:
                raise ValueError(f"archive links are not allowed: {member.filename!r}")
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"archive special file is not allowed: {member.filename!r}")
        for member in members:
            target = _safe_target(dest, member.filename)
            if member.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            with zf.open(member) as source:
                _atomic_extract_file(source, target)


def _atomic_extract_file(source, target):
    """Write an archive member without following a pre-existing file symlink."""
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=parent, prefix=".iotids-extract-", delete=False
        ) as out:
            tmp_path = out.name
            shutil.copyfileobj(source, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def _checksum_allows_extract(path, expected, allow_unverified=False):
    actual = _sha256(path)
    print(f"  SHA-256 {os.path.basename(path)}: {actual}")
    if expected:
        if actual.lower() != expected.lower():
            raise ValueError(
                f"checksum mismatch for {os.path.basename(path)}: expected "
                f"{expected.lower()}, got {actual.lower()}")
        return True
    if allow_unverified:
        print("  [warn] publisher provides no pinned digest; extracting only because "
              "--allow-unverified was explicitly supplied")
        return True
    print("  [warn] archive downloaded but NOT extracted: no trusted checksum. "
          "Review the SHA-256 above, then rerun with --sha256 SUBDIR=HEX (preferred) "
          "or --allow-unverified (explicit risk acceptance).")
    return False


def direct_fetch(checksums=None, allow_unverified=False):
    """Fetch datasets that have an auth-free direct URL (wget, resumable),
    then extract any downloaded archive so the loaders can see the files."""
    checksums = checksums or {}
    wget_bin = shutil.which("wget")
    if not wget_bin:
        print("[ERROR] direct downloads require the wget executable.")
        return
    for url, subdir, note, publisher_sha256 in DIRECT_SOURCES:
        dest = os.path.join(DS, subdir)
        os.makedirs(dest, exist_ok=True)
        print(f"\n[direct] {url}\n         {note}")
        cmd = [wget_bin, "-c", "--tries=20", "--timeout=60"]
        cmd += ["-P", dest, url]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[warn] direct download failed for {subdir} (rc={rc}); "
                  f"the IoT-23 host is slow — the download is resumable, re-run.")
            continue
        archive = os.path.join(dest, os.path.basename(urlparse(url).path))
        expected = checksums.get(subdir, publisher_sha256)
        try:
            if _checksum_allows_extract(archive, expected, allow_unverified):
                _extract_archives(dest)
        except ValueError as e:
            print(f"[ERROR] refusing {subdir}: {e}")


def _extract_archives(dest):
    """Safely extract .tar.gz / .zip found in dest (idempotent-ish)."""
    for arc in glob.glob(os.path.join(dest, "*.tar.gz")) + \
               glob.glob(os.path.join(dest, "*.tgz")):
        try:
            print(f"  extracting {os.path.basename(arc)} ...")
            _safe_extract_tar(arc, dest)
        except Exception as e:
            print(f"  [warn] extract failed: {e}")
    for arc in glob.glob(os.path.join(dest, "*.zip")):
        try:
            print(f"  extracting {os.path.basename(arc)} ...")
            _safe_extract_zip(arc, dest)
        except Exception as e:
            print(f"  [warn] extract failed: {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch IDS datasets for SFAF")
    ap.add_argument("--direct", action="store_true",
                    help="also fetch auth-free direct-URL datasets (IoT-23, WUSTL)")
    ap.add_argument("--sha256", action="append", default=[], metavar="SUBDIR=HEX",
                    help="trusted SHA-256 for a direct archive; repeat per source")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="extract a direct archive without a trusted digest "
                         "(explicit supply-chain risk acceptance)")
    ap.add_argument("--check-only", action="store_true",
                    help="report which datasets are present and exit without "
                         "downloading anything (used by the test suite, and by "
                         "anyone who just wants to know what is missing)")
    args = ap.parse_args()

    checksums = {}
    for item in args.sha256:
        try:
            name, digest = item.split("=", 1)
        except ValueError:
            ap.error("--sha256 must be SUBDIR=HEX")
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            ap.error(f"invalid SHA-256 for {name!r}")
        checksums[name] = digest.lower()

    ensure_dataset_root()
    print("Checking dataset layout under Datasets/ ...")
    if args.check_only:
        ok = check_layout()
        print("\nCore datasets present." if ok else "\nSome core datasets are missing.")
        return
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
        direct_fetch(checksums=checksums, allow_unverified=args.allow_unverified)
    print("\nRe-checking core layout:")
    if check_layout():
        print("\nReady. Run: python code/02_train_sfaf.py")
    else:
        print("\nSome core datasets still missing — see manual steps above.")


if __name__ == "__main__":
    main()
