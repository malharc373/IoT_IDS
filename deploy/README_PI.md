# Deploying the IoT-IDS sensor on a Raspberry Pi

End-to-end walkthrough: flash → connect → copy → install → run → demonstrate.
Tested target: **Raspberry Pi 4 (64-bit Raspberry Pi OS)**. A Pi 3B+/Zero 2 W
also works (inference is ~microseconds; the sniffer is the only real load).

The edge model (`models/live_ids.onnx`, ~55 KB) has the feature scaler baked
in, so the Pi only needs `onnxruntime + numpy + scapy` — no training stack.

---

## 1. Flash Raspberry Pi OS

1. Install **Raspberry Pi Imager** (https://www.raspberrypi.com/software/).
2. Choose **Raspberry Pi OS Lite (64-bit)**.
3. In the ⚙ settings before writing:
   - set a hostname (e.g. `iot-ids`),
   - **enable SSH** (password or key),
   - set username/password and your Wi-Fi (if not using Ethernet).
4. Write the SD card, boot the Pi, and give it ~1 minute.

## 2. Connect over SSH

```bash
ssh <user>@iot-ids.local        # or ssh <user>@<PI_IP>
```

Find the interface name and IP you'll monitor:

```bash
ip -brief addr        # eth0 = wired, wlan0 = Wi-Fi
```

## 3. Copy the project onto the Pi

Either clone (if you pushed this branch to GitHub):

```bash
git clone <your-repo-url> ~/IOT-IDS
cd ~/IOT-IDS
```

…or copy from your dev machine with `scp` / `rsync`:

```bash
# from the Mac, in the repo root
rsync -av --exclude .git --exclude data/pcaps ./ <user>@iot-ids.local:~/IOT-IDS/
```

> Make sure `models/live_ids.onnx` and `models/live_meta.json` come across —
> they are the only runtime artifacts the sensor needs.

## 4. Install (one command)

```bash
cd ~/IOT-IDS
sudo bash deploy/setup_pi.sh eth0        # or wlan0
```

This installs deps into `.venv`, smoke-tests the model, and registers a
`iot-ids` systemd service.

## 5. Run the sensor

```bash
sudo systemctl start iot-ids      # start now
journalctl -u iot-ids -f          # watch live detections
tail -f ~/IOT-IDS/logs/alerts.jsonl
```

Or run it in the foreground to watch directly:

```bash
sudo .venv/bin/python src/ids_daemon.py --iface eth0
```

### Prevention (IPS) mode

To actively block attackers (not just alert), add `--prevent`. It uses
nftables/iptables with a confidence gate, an allowlist, and auto-expiry:

```bash
sudo .venv/bin/python src/ids_daemon.py --iface eth0 --prevent \
     --allow 192.168.1.0/24 --ips-min-conf 0.9 --block-seconds 300
```

Use `--ips` (instead of `--prevent`) for a dry-run that logs what it *would*
block without touching the firewall. To make the systemd service enforce, add
`--prevent` to `ExecStart` in `deploy/setup_pi.sh` before installing.

### Web dashboard

The sensor writes alerts to `logs/alerts.jsonl`; a stdlib dashboard renders them
live. Run it alongside the sensor and browse from any device on the LAN:

```bash
.venv/bin/python src/dashboard.py --port 8080     # http://<pi-ip>:8080
```

It shows active incidents, attack-type/category breakdown, top sources, a
per-minute timeline, and the current IPS blocklist — auto-refreshing every 2s.

## 6. Demonstrate detection (from another host on the LAN)

Install attacker tools on a second machine (your laptop):

```bash
sudo apt install -y nmap hping3 hydra slowhttptest   # Linux
brew install nmap hping                              # macOS (subset)
```

Point them at the Pi's IP (authorized lab use only):

```bash
nmap -sS -p1-1000 <PI_IP>                 # port scan
sudo hping3 -S --flood -p 80 <PI_IP>      # SYN flood     (Ctrl-C to stop)
sudo hping3 --icmp --flood <PI_IP>        # ICMP flood
sudo hping3 --udp  --flood -p 53 <PI_IP>  # UDP flood
hydra -l pi -P wordlist.txt ssh://<PI_IP> # SSH brute-force
slowhttptest -c 200 -H -u http://<PI_IP>/ # slowloris
```

Within a couple of seconds each attack raises an aggregated alert in
`journalctl -u iot-ids -f`, e.g.:

```
⚠ ATTACK portscan  src=192.168.1.50  TCP  873 flows  873 dst-ports ...
⚠ ATTACK synflood  src=192.168.1.50  TCP 4021 flows    1 dst-ports ...
```

No physical attacker handy? Replay a synthetic capture through the exact same
detection core (no root needed):

```bash
.venv/bin/python src/ids_daemon.py --replay data/pcaps/demo_mixed.pcap
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `model not found` | copy `models/live_ids.onnx` + `models/live_meta.json` to the Pi |
| no packets seen live | check the interface name with `ip -brief addr`; run with `sudo` |
| service not starting | `journalctl -u iot-ids -e` to see the error |
| high CPU on Pi Zero | raise `--step` (e.g. `--step 5`) to flush less often |
| only your own traffic seen | a switch isolates ports — mirror the port or run the Pi as the gateway/AP |
