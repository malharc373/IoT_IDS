# Attack simulation

Two ways to drive the IDS with malicious traffic:

## A. Synthetic pcaps (no root, cross-platform) — used for the Mac demo

`traffic_gen.py` crafts labeled captures with scapy. Writing pcaps needs no
privileges, so this runs anywhere and is fully reproducible.

```bash
# one capture of a single class
python attacks/traffic_gen.py --kind portscan --out /tmp/scan.pcap --seed 1

# build the full labeled training corpus + a mixed demo capture
python attacks/build_corpus.py --scenarios 25
```

Classes: `benign portscan synflood icmpflood udpflood ssh_bruteforce slowloris
mirai xmas_scan mqtt_flood` (+ hard-benign variants `benign_burst benign_multi`
that resemble attacks but are legitimate — false-positive traps).

## B. Real attacks against the Pi (root, live) — used for the on-device demo

Run the IDS on the Pi (`sudo python src/ids_daemon.py --iface eth0`), then
launch these from **another machine on the same lab network**, targeting the
Pi's IP. Only ever run these against hosts you own / are authorized to test.

| Class | Real tool command (attacker host) |
|-------|-----------------------------------|
| Port scan | `nmap -sS -p1-1000 <PI_IP>` |
| SYN flood | `sudo hping3 -S --flood -p 80 <PI_IP>` |
| ICMP flood | `sudo hping3 --icmp --flood <PI_IP>`  (or `ping -f <PI_IP>`) |
| UDP flood | `sudo hping3 --udp --flood -p 53 <PI_IP>` |
| SSH brute-force | `hydra -l pi -P wordlist.txt ssh://<PI_IP>` (or `ncrack`) |
| Slowloris | `slowhttptest -c 200 -H -u http://<PI_IP>/` |
| Xmas / NULL scan | `nmap -sX <PI_IP>` / `nmap -sN <PI_IP>` |
| Mirai-style spread | `nmap -p23,2323 --open <SUBNET>/24` (telnet sweep) |
| MQTT flood | `for i in $(seq 500); do mosquitto_pub -h <PI_IP> -t x -m y & done` |

Install on the attacker host:
```bash
sudo apt install -y nmap hping3 hydra slowhttptest
```

`generate_pcaps.py` note: the synthetic generators reproduce the *flow-level
signatures* these tools produce (SYN-only short flows, high-rate single-port
floods, many-port sweeps, long low-and-slow connections), which is what the
flow-based model keys on.

## C. One-command live demo — used for the presentation

`live_demo.py` drives a **running** sensor with real traffic so an audience can
watch the dashboard react: benign background chatter throughout, then every
attack class in turn with a gap after each. Run it on the same box as the
sensor; loopback is the default target, so nothing leaves the machine.

```bash
# terminal 1 — the sensor
sudo python src/ids_daemon.py --iface eth0,lo --log logs/alerts.jsonl
# terminal 2 — the demo (sudo enables the hping3 floods + Xmas scan)
sudo python attacks/live_demo.py            # all classes, ~4-5 min
sudo python attacks/live_demo.py --fast     # shorter, for a rehearsal
python  attacks/live_demo.py --only portscan slowloris   # pick classes
```

Without root it still runs the connect-scan, SSH brute force, slowloris, MQTT
and Mirai classes and skips the raw-packet floods with a note. It targets
loopback unless you pass `--target <ip> --i-own-the-target`, and even then it
refuses the packet floods against a remote host unless you add
`--allow-remote-flood`. Watch `logs/alerts.jsonl` or the dashboard as it runs.

> Authorized-use only. These commands generate hostile traffic and must be
> confined to a lab you control (your own Pi + host). Never point them at
> third-party systems.
