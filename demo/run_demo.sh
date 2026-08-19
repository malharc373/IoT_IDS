#!/usr/bin/env bash
# run_demo.sh — end-to-end IoT-IDS demonstration (no root required).
#
#   1. (re)generate a fresh mixed capture: benign traffic + 9 attack types
#   2. replay it through the IDS with live-style, aggregated alerts
#   3. run held-out validation on unseen scenarios and print the scorecard
#
# On the Raspberry Pi you would instead run the live sniffer:
#   sudo python src/ids_daemon.py --iface eth0
# and launch the attacks from another host (see attacks/README.md).

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore

BOLD=$'\033[1m'; CYA=$'\033[36m'; NC=$'\033[0m'

echo "${BOLD}${CYA}==> [1/3] Generating fresh mixed capture (benign + 9 attacks)${NC}"
python - <<'PY'
import sys; sys.path.insert(0, "attacks")
import build_corpus as bc
bc._build_demo_mixed("data/pcaps")
PY

echo
echo "${BOLD}${CYA}==> [2/3] Live-style detection replay${NC}"
python src/ids_daemon.py --replay data/pcaps/demo_mixed.pcap \
    --step 1.0 --min-conf 0.5 --log logs/alerts.jsonl

echo
echo "${BOLD}${CYA}==> [3/3] Held-out validation (unseen scenarios)${NC}"
python demo/validate.py

echo
echo "${BOLD}Artifacts:${NC}"
echo "  logs/alerts.jsonl                              (alert feed)"
echo "  demo/results/heldout_confusion_matrix.png      (validation)"
echo "  demo/results/live_confusion_matrix.png         (training test)"
echo "  models/live_ids.onnx                           (edge model, ~96 KB)"
