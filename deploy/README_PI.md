# Deploying the IoT-IDS sensor on a Raspberry Pi

End-to-end walkthrough: flash → connect → copy → install → run → demonstrate.
Target: **Raspberry Pi 4 with 64-bit Raspberry Pi OS and glibc 2.28+**. The
dependency lock has CPython 3.10 aarch64 wheels available for its native
packages. Actual Pi throughput and soak evidence is still pending; host
inference timings are not a substitute for that acceptance run.

The edge model (`models/live_ids.onnx`, 91.8 KB) takes raw flow features — trees
are scale-invariant, so there is no scaler to ship or drift — and the Pi needs
only `onnxruntime + numpy + scapy`, no training stack.

> **Decide your topology before you install.** A sensor on a mirror port can
> *detect* attacks on other devices but can never *stop* them. Only an inline
> sensor can. See [§6 Placement](#6-placement-passive-vs-inline) — it decides
> whether `--ips-scope host` or `--ips-scope network` is the honest setting.

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

This installs deps into `.venv`, smoke-tests the model, and registers two
systemd services: **`iot-ids`** (the sensor, runs as root) and
**`iot-ids-dashboard`** (the web UI, runs as your user). Pass a second argument
to change the dashboard port: `sudo bash deploy/setup_pi.sh eth0 8080`.
The installer consumes the exact transitive `deploy/requirements-pi.txt` lock;
its reviewed direct inputs are kept in `deploy/requirements-pi.in`.

## 5. Run the sensor

```bash
sudo systemctl start iot-ids iot-ids-dashboard   # sensor + dashboard
journalctl -u iot-ids -f                          # watch live detections
tail -f ~/IOT-IDS/logs/alerts.jsonl
```

The dashboard is live at `http://<pi-ip>:8080/?token=<token>` — `setup_pi.sh`
mints a token on first run, stores it at `logs/dashboard.token` (mode 600) and
prints the full URL. The page exposes attacking hosts, active blocks and the
segment's addressing, so it refuses to serve a network without one. To skip the
token entirely, keep it on loopback and tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 <user>@iot-ids.local     # then browse localhost:8080
```

Both services start automatically on boot.

Or run it in the foreground to watch directly:

```bash
sudo .venv/bin/python src/ids_daemon.py --iface eth0
```

### Prevention (IPS) mode

The responder acts on a ladder — **monitor → throttle → block** — via
nftables/iptables, with an allowlist and auto-expiry:

```bash
sudo .venv/bin/python src/ids_daemon.py --iface eth0 --prevent \
     --allow 192.168.1.0/24 --ips-min-conf 0.9 --block-seconds 300 \
     --ips-scope host --ips-strikes 3
```

Use `--ips` (instead of `--prevent`) for a dry-run that logs what it *would*
do without touching the firewall. To make the systemd service enforce, add
`--prevent` to `ExecStart` in `deploy/setup_pi.sh` before installing.

Two flags change what actually happens, and neither has a safe default you can
ignore:

| flag | meaning |
|---|---|
| `--ips-scope host` (default) | rules on **INPUT** only — protects *this Pi*. Correct for a mirror-port sensor, and honest that attacks on other devices are not stopped. |
| `--ips-scope network` | rules on **INPUT + FORWARD** — protects devices that route *through* the Pi. Requires the inline setup in §6. |
| `--ips-strikes 3` | corroborating incidents required inside `--ips-strike-window` before blocking. Below that the source is **rate-limited** (`--ips-throttle-pps`, default 20) rather than blackholed. |

The strike gate exists because the model's confidence is **not calibrated** — a
reported 0.99 is not a 99% guarantee. Blackholing a host on one saturated score
is not a defensible action, so evidence has to accumulate first.

### Web dashboard

The sensor writes alerts to `logs/alerts.jsonl`; a stdlib dashboard renders them
live. Run it alongside the sensor and browse from any device on the LAN:

```bash
.venv/bin/python src/dashboard.py --port 8080                    # loopback only
.venv/bin/python src/dashboard.py --host 0.0.0.0 --token generate # LAN + token
```

It shows active incidents, attack-type/category breakdown, top sources, a
per-minute timeline, and the current IPS blocklist and throttle list —
auto-refreshing every 2s. There is no TLS (it is a stdlib HTTP server), so the
token authenticates but does not encrypt; over an untrusted network use the SSH
tunnel above.

## 6. Placement: passive vs inline

This is the decision that determines whether the "P" in IPS means anything.

### Passive — mirror/SPAN port (default, `--ips-scope host`)

```
        ┌────────┐
 LAN ───┤ switch ├─── IoT devices
        └───┬────┘
            │ mirror port
        ┌───┴───┐
        │  Pi   │   sees everything, forwards nothing
        └───────┘
```

The Pi observes a copy of the traffic. It can **detect** an attack on a camera
or a thermostat, and it can block traffic aimed at *itself* — but the attack
packets never traverse the Pi, so no firewall rule on it can stop them. This is
the right choice for monitoring, and `--ips-scope host` is the honest setting.

On a plain unmanaged switch you will only ever see broadcast traffic and your
own — that is the "only your own traffic seen" row in Troubleshooting.

### Inline — bridge (`--ips-scope network`)

```
        ┌───────┐            ┌────────┐
 uplink ┤ eth0  │            │        │
        │  Pi   │── br0 ─────┤ switch ├─── IoT devices
 (usb)  │ eth1  │            │        │
        └───────┘            └────────┘
```

Everything the devices send crosses the Pi, so a FORWARD-chain rule can
actually drop it. Needs a second NIC (a USB-Ethernet adapter is fine).

```bash
sudo apt install -y bridge-utils
```

Create the bridge with systemd-networkd (survives reboot):

```bash
sudo tee /etc/systemd/network/br0.netdev >/dev/null <<'EOF'
[NetDev]
Name=br0
Kind=bridge
EOF

sudo tee /etc/systemd/network/br0.network >/dev/null <<'EOF'
[Match]
Name=br0
[Network]
DHCP=yes
EOF

# enslave both NICs
for i in eth0 eth1; do
  sudo tee /etc/systemd/network/$i.network >/dev/null <<EOF
[Match]
Name=$i
[Network]
Bridge=br0
EOF
done

sudo systemctl enable --now systemd-networkd
```

Bridged frames must traverse the FORWARD chain for the IPS to see them:

```bash
sudo modprobe br_netfilter
echo br_netfilter | sudo tee /etc/modules-load.d/br_netfilter.conf
printf 'net.bridge.bridge-nf-call-iptables=1\nnet.bridge.bridge-nf-call-ip6tables=1\n' \
  | sudo tee /etc/sysctl.d/99-iot-ids.conf
sudo sysctl --system
```

Then sniff the bridge and enforce at network scope:

```bash
sudo .venv/bin/python src/ids_daemon.py --iface br0 --prevent \
     --ips-scope network --allow 192.168.1.0/24
```

> **Fail-open matters.** An inline Pi is now a single point of failure for the
> whole segment. The nft table this project installs has `policy accept` and
> drops only listed sources, so a *crashed* daemon leaves traffic flowing — but
> a *powered-off* Pi breaks the link. For anything beyond a lab, use a
> fail-open network tap or a managed switch's mirror port and stay passive.

> **Test in dry-run first.** Run with `--ips --ips-scope network` for a while
> and read `logs/alerts.jsonl`. A false positive in `host` scope costs you the
> sensor; in `network` scope it costs a device its connectivity.

To make it permanent, edit `ExecStart` in `/etc/systemd/system/iot-ids.service`
(or `deploy/setup_pi.sh` before installing) to use `--iface br0 --prevent
--ips-scope network`, then `sudo systemctl daemon-reload && sudo systemctl
restart iot-ids`.

### Verifying enforcement works

```bash
sudo nft list table inet iot_ids       # sets 'blocked'/'throttled' + chains
sudo nft list set inet iot_ids blocked # currently blocked sources
python src/ips_response.py --status    # what the responder thinks it can do
```

`status()` reports `"protects": "this sensor only"` or
`"this sensor + forwarded traffic"` — if it says the former while you expected
inline enforcement, the scope flag did not take effect.

## 7. Demonstrate detection (from another host on the LAN)

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

## Benchmark the Pi (real numbers, not projected)

`demo/benchmark.py` runs on the minimal Pi runtime — it degrades gracefully when
training libs are absent and prints *measured* Pi figures (no ×12 projection):

```bash
.venv/bin/python demo/benchmark.py        # writes demo/results/BENCHMARK.md
```

It reports model size, ONNX + native-C inference latency/throughput, feature-
extraction throughput, end-to-end pcap speed, held-out accuracy, and memory. The
tree/node counts fall back to the C header if xgboost isn't installed; the only
section that needs extra libs is the latency chart (matplotlib), which is skipped
cleanly if absent.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `model not found` | copy `models/live_ids.onnx` + `models/live_meta.json` to the Pi |
| no packets seen live | check the interface name with `ip -brief addr`; run with `sudo` |
| service not starting | `journalctl -u iot-ids -e` to see the error |
| high CPU on Pi Zero | raise `--step` (e.g. `--step 5`) to flush less often |
| only your own traffic seen | a switch isolates ports — mirror the port, or go inline (§6) |
| detections fire but nothing is blocked | you are in `--ips-scope host` on a mirror port: the attack traffic never crosses the Pi. See §6. |
| `nft list table inet iot_ids` is empty | the responder fell back to dry-run — check `python src/ips_response.py --status` for `backend` and whether it has root |
| dashboard refuses to start | binding off-loopback needs `--token` or `--insecure`; the error message lists the options |
| sources throttled but never blocked | expected — `--ips-strikes` corroborating incidents are required first; lower it or widen `--ips-strike-window` |
