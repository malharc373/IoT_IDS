#!/usr/bin/env bash
# setup_pi.sh — one-shot Raspberry Pi setup for the IoT-IDS live sensor.
#
# Installs system + Python deps, creates a venv, and installs a systemd
# service that runs the live sniffer on boot. Idempotent — safe to re-run.
#
# Usage (on the Pi):
#   cd ~/IOT-IDS
#   sudo bash deploy/setup_pi.sh [INTERFACE]
#
# INTERFACE defaults to eth0 (use wlan0 for Wi-Fi).

set -euo pipefail

IFACE="${1:-eth0}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
VENV="$REPO_DIR/.venv"
PY="$VENV/bin/python"

echo "=== IoT-IDS Pi setup ==="
echo "  repo      : $REPO_DIR"
echo "  interface : $IFACE"
echo "  user      : $RUN_USER"

echo "--- [1/4] system packages ---"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y python3-venv python3-pip libpcap0.8 tcpdump
else
    echo "  (apt-get not found — skipping; install python3-venv + libpcap manually)"
fi

echo "--- [2/4] python venv + deps ---"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REPO_DIR/deploy/requirements-pi.txt"

echo "--- [3/4] smoke test the model loads ---"
"$PY" - <<PY
import onnxruntime as rt, json, os
m = os.path.join("$REPO_DIR", "models", "live_ids.onnx")
assert os.path.exists(m), "models/live_ids.onnx missing — copy it from your dev machine"
rt.InferenceSession(m)
meta = json.load(open(os.path.join("$REPO_DIR","models","live_meta.json")))
print("  model OK:", meta["metrics"])
PY

echo "--- [4/4] systemd service ---"
SERVICE=/etc/systemd/system/iot-ids.service
sudo tee "$SERVICE" >/dev/null <<UNIT
[Unit]
Description=IoT-IDS live intrusion detection sensor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$REPO_DIR
ExecStart=$PY $REPO_DIR/src/ids_daemon.py --iface $IFACE --log $REPO_DIR/logs/alerts.jsonl
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable iot-ids.service

echo
echo "=== Done ==="
echo "Start now :  sudo systemctl start iot-ids"
echo "Live logs :  journalctl -u iot-ids -f"
echo "Alerts    :  tail -f $REPO_DIR/logs/alerts.jsonl"
echo "Manual run:  sudo $PY src/ids_daemon.py --iface $IFACE"
