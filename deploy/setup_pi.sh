#!/usr/bin/env bash
# setup_pi.sh — one-shot Raspberry Pi setup for the IoT-IDS live sensor.
#
# Installs system + Python deps, creates a venv, and installs a systemd
# service that runs the live sniffer on boot. Idempotent — safe to re-run.
#
# Usage (on the Pi):
#   cd ~/IOT-IDS
#   sudo bash deploy/setup_pi.sh [INTERFACE] [DASHBOARD_PORT]
#
# INTERFACE defaults to eth0 (use wlan0 for Wi-Fi); DASHBOARD_PORT to 8080.
# Installs two services: iot-ids (sensor) and iot-ids-dashboard (web UI).

set -euo pipefail

IFACE="${1:-eth0}"
DASH_PORT="${2:-8080}"
# The dashboard exposes the alert feed (attacking hosts, blocked hosts, segment
# addressing), so reaching it over the LAN requires a token. Reuse the existing
# one across re-runs so bookmarked URLs keep working.
TOKEN_FILE="$(cd "$(dirname "$0")/.." && pwd)/logs/dashboard.token"
mkdir -p "$(dirname "$TOKEN_FILE")"
if [ -s "$TOKEN_FILE" ]; then
    DASH_TOKEN="$(cat "$TOKEN_FILE")"
else
    DASH_TOKEN="$(head -c 18 /dev/urandom | base64 | tr -d '/+=' )"
    printf '%s' "$DASH_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
fi
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
VENV="$REPO_DIR/.venv"
PY="$VENV/bin/python"

echo "=== IoT-IDS Pi setup ==="
echo "  repo      : $REPO_DIR"
echo "  interface : $IFACE"
echo "  dashboard : port $DASH_PORT"
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

echo "--- [4/4] systemd services ---"
# Sensor (needs root for packet capture)
sudo tee /etc/systemd/system/iot-ids.service >/dev/null <<UNIT
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

# Dashboard (no root needed — only reads the alert log; starts after the sensor)
sudo tee /etc/systemd/system/iot-ids-dashboard.service >/dev/null <<UNIT
[Unit]
Description=IoT-IDS web dashboard
After=iot-ids.service network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO_DIR
Environment=IOTIDS_DASHBOARD_TOKEN=$DASH_TOKEN
ExecStart=$PY $REPO_DIR/src/dashboard.py --host 0.0.0.0 --port $DASH_PORT --log $REPO_DIR/logs/alerts.jsonl
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable iot-ids.service iot-ids-dashboard.service

# Best-effort: discover the LAN IP to print a clickable dashboard URL
IP_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"; IP_ADDR="${IP_ADDR:-<pi-ip>}"

echo
echo "=== Done ==="
echo "Start both :  sudo systemctl start iot-ids iot-ids-dashboard"
echo "Dashboard  :  http://$IP_ADDR:$DASH_PORT/?token=$DASH_TOKEN"
echo "             (token stored at $TOKEN_FILE, mode 600)"
echo "             for no token at all, tunnel instead:"
echo "               ssh -L $DASH_PORT:127.0.0.1:$DASH_PORT $RUN_USER@$IP_ADDR"
echo "Sensor logs:  journalctl -u iot-ids -f"
echo "Dash logs  :  journalctl -u iot-ids-dashboard -f"
echo "Alerts     :  tail -f $REPO_DIR/logs/alerts.jsonl"
