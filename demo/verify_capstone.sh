#!/usr/bin/env bash
# Reproduce every laptop-available acceptance check. External datasets and
# Raspberry Pi measurements are intentionally excluded and remain separate gates.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

ruff check .
mypy
pytest -q tests/
python demo/preflight.py --runtime-only
bash demo/run_demo.sh
python demo/benchmark.py
python reports/build_report.py

if grep -n "section FAILED" demo/results/BENCHMARK.md; then
  echo "benchmark report contains a failed section" >&2
  exit 1
fi

echo "Laptop capstone verification passed. Pi, real-PCAP, and external-dataset gates remain."
