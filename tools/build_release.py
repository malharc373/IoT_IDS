#!/usr/bin/env python3
"""Build a deterministic capstone release bundle and SHA-256 manifest."""
from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUNDLE = DIST / "iot-ids-capstone.tar.gz"
MANIFEST = DIST / "SHA256SUMS"

RELEASE_FILES = (
    "README.md",
    "SECURITY.md",
    "deploy/PI_ACCEPTANCE.md",
    "deploy/README_MCU.md",
    "deploy/README_PI.md",
    "deploy/iot-ids-dashboard.service",
    "deploy/iot-ids.service",
    "deploy/requirements-pi.txt",
    "deploy/setup_pi.sh",
    "models/README.md",
    "models/live_ids.h",
    "models/live_ids.onnx",
    "models/live_meta.json",
    "output/pdf/IOT_IDS_Corrected_Technical_Report.pdf",
    "output/presentation/IOT_IDS_Corrected_Project_Review.pptx",
    "src/dashboard.py",
    "src/flow_features.py",
    "src/ids_daemon.py",
    "src/ips_response.py",
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> tuple[Path, Path]:
    missing = [name for name in RELEASE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("missing release files: " + ", ".join(missing))

    DIST.mkdir(exist_ok=True)
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(RELEASE_FILES):
            path = ROOT / name
            info = archive.gettarinfo(str(path), arcname=f"iot-ids-capstone/{name}")
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)

    with BUNDLE.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(tar_bytes.getvalue())

    MANIFEST.write_text(f"{_digest(BUNDLE)}  {BUNDLE.name}\n", encoding="utf-8")
    return BUNDLE, MANIFEST


if __name__ == "__main__":
    bundle, manifest = build()
    print(bundle.relative_to(ROOT))
    print(manifest.relative_to(ROOT))
