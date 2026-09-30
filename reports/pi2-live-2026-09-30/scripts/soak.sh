#!/usr/bin/env bash
# 24 h soak record: one CSV row every 5 minutes (deploy/PI_ACCEPTANCE.md §3).
cd ~/IOT-IDS || exit 1
F=acceptance/soak/soak.csv; mkdir -p acceptance/soak
[ -s "$F" ] || echo "ts,active,nrestarts,mem_current_mb,cpu_s,daemon_rss_mb,temp_c,throttled,eth0_rx_pkts,eth0_rx_dropped,sensor_pkts,flows_in_table,incidents_emitted,alert_lines,load1" > "$F"
for i in $(seq 1 288); do
  q() { systemctl show iot-ids -p "$1" --value; }
  act=$(q ActiveState); nr=$(q NRestarts); cpu=$(q CPUUsageNSec)
  mem=$(q MemoryCurrent); case "$mem" in ''|*[!0-9]*) mem=na ;; *) mem=$(( mem / 1048576 )) ;; esac
  case "$cpu" in ''|*[!0-9]*) cpu=na ;; *) cpu=$(( cpu / 1000000000 )) ;; esac
  hb=$(python3 -c "import json;d=json.load(open('logs/sensor_status.json'));h=d['host'];print(h['rss_mb'],d['pkts'],d['flows_in_table'],d['incidents_emitted'],h['load1'])" 2>/dev/null)
  set -- $hb
  rx=$(ip -s link show eth0 | awk '/RX:/{getline; print $2, $4}')
  thr=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
  temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
  echo "$(date -Is),$act,$nr,$mem,$cpu,$1,$temp,${thr:-na},$(echo $rx | tr ' ' ','),$2,$3,$4,$(wc -l < logs/alerts.jsonl),$5" >> "$F"
  sleep 300
done
