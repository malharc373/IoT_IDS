# Raspberry Pi 2: live traffic, real attacks, IPS, reboot (30 September 2026)

Follows [`pi2-acceptance-2026-09-30`](../pi2-acceptance-2026-09-30/README.md),
which covered what runs on the board. This run puts the sensor on a real
network and attacks it with real tools. **No synthetic traffic** is used
anywhere below.

| Setting | Value |
|---|---|
| Board | Raspberry Pi 2 Model B, 32-bit Raspberry Pi OS (Debian 13 trixie) |
| Network | home LAN, `192.168.31.0/24`; attacker laptop at `.178`, Pi at `.68` |
| Sensor | systemd service from `deploy/setup_pi.sh`, `--iface eth0,lo`, C engine, default thresholds (`--min-conf 0.5`) |
| Model | `models/live_ids.onnx` / `live_ids.h` unchanged (synthetic-trained) |

The home IPv6 prefix in the raw files is replaced by the documentation prefix
`2001:db8:0:1::/64`; the dashboard token printed by the installer is redacted.

## 1. Installer and services

- `sudo bash deploy/setup_pi.sh eth0,lo 8080` on a clean venv
  ([console](raw/setup_pi-console.txt)). The first attempt failed: Debian 13
  renamed `libpcap0.8` to `libpcap0.8t64`, so `apt-get` aborted. The installer
  now picks whichever name the release provides; the second run completed.
- The armv7 path installed gcc, libopenblas0 and piwheels numpy 2.2.6, and the
  preflight reported `backend=c` on both interfaces.
- **Reboot:** both services are enabled. After `systemctl reboot` the Pi
  answered SSH in 80 s and the sensor had a fresh heartbeat 99 s after the
  command (kernel + userspace boot 73 s), `NRestarts=0`.

## 2. Benign baseline: false alarms on real traffic

Ten minutes, no attacks ([snapshots](raw/baseline-snapshots.txt),
[alerts](raw/alerts.jsonl)):

| Measure | Value |
|---|---|
| Traffic | 2,475 packets (~4/s); eth0 dropped 0 |
| Distinct false-alarm incidents | **19** (~1.9 per minute): 14 `udpflood`, 5 `icmpflood` |
| From IPv6 link-local sources | 11 of 19 |
| Largest | router `192.168.31.1`, `icmpflood`, 3 flows, 78 packets |
| Surviving a ≥ 0.9 confidence gate | 13 of 19 |
| Surviving a ≥ 5 flows-per-incident gate | 0 of 19 |

The model was trained only on generated traffic and has never seen ordinary
LAN chatter (IPv6 neighbour discovery, multicast, IGMP, mDNS). It labels it
with high confidence, so confidence cannot filter it; incident size can.

## 3. Real attacks

`scripts/run_attacks.py`, one attack at a time with gaps
([results](raw/attacks.json)). Laptop-side attacks use unprivileged tools
over the LAN. Raw-packet attacks run on the Pi against its own address, so
floods never leave the board; the sensor watches `lo` for them.

| Attack (tool) | Reported for the attacker | First alert | Confidence |
|---|---|---|---|
| Port scan, `nmap -sT -p1-1000` (laptop) | **portscan**, 755 ports, 1,539 flows | 1 s | 1.00 |
| SSH brute force, 40 failed logins (laptop) | **ssh_bruteforce**, but only 2 flows of ~40 connections | 21 s | 1.00 |
| Slowloris, 40 held connections × 60 s (laptop) | **slowloris**, 29 flows | 1 s | 0.92 |
| Xmas scan, `nmap -sX` (Pi → itself) | **xmas_scan**, 706 ports | 3 s | 0.97 |
| SYN flood, `hping3 -S` 2k pps × 20 s (Pi → itself) | **synflood**, 2,618 flows | 2 s | 0.99 |
| UDP flood, `hping3 --udp` 2k pps × 20 s (Pi → itself) | `xmas_scan` (wrong type, still an attack) | < 1 s | 0.86 |
| ICMP flood, `hping3 --icmp` 2k pps × 20 s (Pi → itself) | **icmpflood**, weak | 13 s | 0.52 |

- All 7 attacks raised an incident; 5 of 7 carried the right label at
  confidence ≥ 0.9.
- **Capture loss under flood:** each flood sent about 30,000 packets; the
  sensor processed 3.5k–4.3k packets in the same windows (about 12–14 %,
  including other traffic). Python capture and parsing is the ceiling on
  this board (6.8k packets/s in the benchmark).
- During the floods the CPU stayed at or below 54 °C with no throttling, and
  the sensor's RSS at or below 70 MB.
- Smaller side labels on the attacker (for example 1–2 flow `synflood` or
  `mqtt_flood`) are the laptop's own background traffic, the same false
  alarms as in §2.

## 4. IPS in enforce mode

`scripts/ips_test.sh`: the sensor was restarted with
`--ips --prevent --block-seconds 45`, the laptop was **not** allowlisted, and
it port-scanned the Pi while probing the Pi's SSH port once a second
([reachability](raw/ips-reachability.txt),
[sensor console](raw/ips-enforce-console.txt),
[nftables at stop](raw/nft-at-stop.txt)).

| Measure | Value |
|---|---|
| Attacker blocked | yes, about 5 s after the scan began (throttled first, then blocked on the third strike) |
| While the attack continued | the block kept renewing, because the sensor captures before the firewall |
| After the attack stopped | lifted by the kernel timeout 21 s later |
| Block actions | 12: 11 on the laptop, **1 on an innocent IPv6 LAN device** |
| Throttled sources | 13, of which **12 innocent**: the router, 5 other IPv4 hosts including `0.0.0.0` (DHCP), 6 IPv6 link-local hosts |
| Admin traffic | after the scan, the laptop's SSH session and tunnel were read as `mqtt_flood` and re-blocked 5 times |

Blocking and expiry work mechanically, but on real traffic the false alarms
of §2 turn directly into throttles and blocks of normal devices, including
the gateway. **The IPS is not safe to enable on a real network with this
model.** Keep it in dry-run until false alarms are addressed.

Defects found in this run (both addressed after it, in PR #10; the fixes have
unit tests but have not been re-run in enforce mode on the Pi):

- When a throttle or block expires, `ips_response.py` deletes the nftables
  element the kernel has already removed and logs `backend cmd failed` for
  each one. Harmless, noisy.
- `0.0.0.0` (a DHCP client's source) can be throttled; unspecified,
  link-local and multicast sources should never be enforcement targets.
  The one innocent device that was blocked (`fe80::…`) was link-local.
  Replaying this run's recorded actions through the new rule removes 7 of
  the 12 innocent sources; the router and four other IPv4 hosts remain.

## 5. Soak

`scripts/soak.sh` records a CSV row every 5 minutes for 24 h
([first rows](raw/soak-start.csv)): service state, restarts, sensor RSS, CPU
temperature and throttle bits, eth0 drops, packet and incident counts. It
started at 20:25 on 30 Sep; results are added when it finishes.

## Recommendations

Ordered by effect on the false-alarm rate, none applied yet (each changes
detection behaviour and needs re-measurement):

1. Require at least 5 flows per incident before alerting, since no benign
   incident here exceeded 3.
2. Skip link-local, multicast, broadcast and unspecified addresses as
   incident sources and as IPS targets.
3. Parse IPv6 extension headers: MLD and similar traffic reached the model
   with `proto = 0`, a value it never saw in training.
4. Train on real benign captures, not only generated traffic.
5. For flood-rate capture, move packet capture off Python (for example
   AF_PACKET with a ring buffer, or a C front end).
