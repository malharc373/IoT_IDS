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
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
VENV="$REPO_DIR/.venv"
PY="$VENV/bin/python"
# The dashboard exposes the alert feed (attacking hosts, blocked hosts, segment
# addressing), so reaching it over the LAN requires a token. Reuse the existing
# one across re-runs so bookmarked URLs keep working.
#
# The token lives ONLY in this 600 file. It is deliberately not placed in the
# systemd unit via Environment= (units are world-readable and the value shows up
# in `systemctl show`) and not passed as an argument (visible to any local user
# in `ps`). The service reads it with --token-file.
TOKEN_FILE="${IOTIDS_DASHBOARD_TOKEN_FILE:-$REPO_DIR/logs/dashboard.token}"
mkdir -p "$(dirname "$TOKEN_FILE")"
if [ -s "$TOKEN_FILE" ]; then
    DASH_TOKEN="$(cat "$TOKEN_FILE")"
else
    DASH_TOKEN="$(head -c 18 /dev/urandom | base64 | tr -d '/+=' )"
    printf '%s' "$DASH_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
fi
chown "$RUN_USER" "$TOKEN_FILE" 2>/dev/null || true

# CI exercises this initialization under `set -u` without installing packages
# or writing systemd units. It also gives operators a safe configuration check.
if [ "${IOTIDS_SETUP_PREFLIGHT_ONLY:-0}" = "1" ]; then
    exit 0
fi

echo "=== IoT-IDS Pi setup ==="
echo "  repo      : $REPO_DIR"
echo "  interface : $IFACE"
echo "  dashboard : port $DASH_PORT"
echo "  user      : $RUN_USER"

# onnxruntime has no 32-bit ARM wheels; there the daemon compiles the C export
# of the same model (src/c_backend.py), which needs gcc instead.
case "$(uname -m)" in
    armv6l|armv7l) REQS="$REPO_DIR/deploy/requirements-pi-armv7.txt"; EXTRA_PKGS="gcc libopenblas0" ;;
    *)             REQS="$REPO_DIR/deploy/requirements-pi.txt";       EXTRA_PKGS="" ;;
esac
echo "  runtime   : $(basename "$REQS")"

echo "--- [1/4] system packages ---"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    # Debian 13 (trixie, current Raspberry Pi OS) renamed the runtime library
    # to libpcap0.8t64 in the 64-bit time_t transition; older releases keep
    # the original name.
    PCAP_PKG=libpcap0.8
    if apt-cache show libpcap0.8t64 >/dev/null 2>&1; then
        PCAP_PKG=libpcap0.8t64
    fi
    # shellcheck disable=SC2086  # EXTRA_PKGS is intentionally word-split
    sudo apt-get install -y python3-venv python3-pip "$PCAP_PKG" tcpdump nftables $EXTRA_PKGS
else
    echo "  (apt-get not found — skipping; install python3-venv + libpcap manually)"
fi

echo "--- [2/4] python venv + deps ---"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REQS"

echo "--- [3/4] runtime, artifact, feature and inference preflight ---"
"$PY" "$REPO_DIR/demo/preflight.py" --runtime-only --iface "$IFACE"

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
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
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
ExecStart=$PY $REPO_DIR/src/dashboard.py --host 0.0.0.0 --port $DASH_PORT --token-file $TOKEN_FILE --log $REPO_DIR/logs/alerts.jsonl
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
