#!/usr/bin/env bash
# Real IPS enforcement test: the Mac attacks, the Pi blocks it for 45 s, the
# kernel timeout lifts the block. Everything is restored at the end.
PI=malharfalke@192.168.31.68; IP=192.168.31.68
S="ssh -o BatchMode=yes -o ConnectTimeout=5 $PI"
OUT=$(dirname "$0")/results/ips; mkdir -p "$OUT"

$S 'cd ~/IOT-IDS && sudo systemctl stop iot-ids && sudo nft delete table inet iot_ids 2>/dev/null; \
    sudo rm -f logs/ips_state.json; \
    sudo setsid -f .venv/bin/python src/ids_daemon.py --iface eth0,lo --log logs/alerts.jsonl \
      --ips --prevent --block-seconds 45 --ips-strikes 3 --ips-strike-window 120 \
      > acceptance/root/ips_enforce.log 2>&1 < /dev/null; sleep 8; \
    grep -m1 "\[IPS\]" acceptance/root/ips_enforce.log; sudo nft list table inet iot_ids | head -30' > "$OUT/01_start.txt" 2>&1

# probe SSH reachability once a second, in the background
( for i in $(seq 1 150); do
    if nc -z -G1 $IP 22 2>/dev/null; then echo "$(date +%H:%M:%S) up"; else echo "$(date +%H:%M:%S) DOWN"; fi
    sleep 1
  done ) > "$OUT/02_reachability.txt" &
PROBE=$!
sleep 3
echo "attack start $(date +%H:%M:%S)" > "$OUT/03_attack.txt"
nmap -sT -Pn -p1-2000 --max-rate 300 $IP >> "$OUT/03_attack.txt" 2>&1
echo "attack end $(date +%H:%M:%S)" >> "$OUT/03_attack.txt"
wait $PROBE

$S 'cd ~/IOT-IDS && grep -E "\[IPS\]" acceptance/root/ips_enforce.log | tail -40; echo ---nft; \
    sudo nft list table inet iot_ids; echo ---state; sudo cat logs/ips_state.json' > "$OUT/04_ips_log.txt" 2>&1

# restore: stop the enforcing daemon, drop the table, bring the service back
$S 'cd ~/IOT-IDS && sudo pkill -f "ids_daemon.py --iface eth0,lo --log logs/alerts.jsonl --ips" ; sleep 2; \
    sudo nft delete table inet iot_ids 2>/dev/null; sudo systemctl start iot-ids; sleep 5; \
    systemctl is-active iot-ids; sudo nft list tables' > "$OUT/05_restore.txt" 2>&1
echo DONE
