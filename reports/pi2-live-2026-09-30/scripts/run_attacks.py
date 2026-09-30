#!/usr/bin/env python3
"""Run real attacks one at a time against the Pi sensor and record what it
reports for each. Mac-side attacks use unprivileged tools over the LAN; raw
packet attacks run on the Pi against itself (loopback) so floods never leave it.
"""
import json
import os
import subprocess
import sys
import time

PI = "malharfalke@192.168.31.68"
PI_IP = "192.168.31.68"
MAC_IP = "192.168.31.178"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
SSH = ["ssh", "-o", "BatchMode=yes", PI]


def pi(cmd, timeout=180):
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)


def alert_count():
    return int(pi("wc -l < ~/IOT-IDS/logs/alerts.jsonl").stdout.strip())


def new_alerts(start_line):
    out = pi(f"tail -n +{start_line + 1} ~/IOT-IDS/logs/alerts.jsonl").stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def status():
    return json.loads(pi("cat ~/IOT-IDS/logs/sensor_status.json").stdout)


SSH_BRUTE = ("for i in $(seq 1 40); do ssh -o BatchMode=yes -o PubkeyAuthentication=no "
             "-o PreferredAuthentications=password -o ConnectTimeout=5 "
             f"-o StrictHostKeyChecking=no nosuchuser@{PI_IP} true 2>/dev/null; "
             "sleep 0.3; done; echo done 40 attempts")

ATTACKS = [
    # name, where, command, sources we expect the sensor to attribute it to
    ("portscan (nmap -sT, Mac->Pi)", "mac",
     f"nmap -sT -Pn -p1-1000 --max-rate 200 {PI_IP}", {MAC_IP}),
    ("ssh_bruteforce (40 failed logins, Mac->Pi)", "mac", SSH_BRUTE, {MAC_IP}),
    ("slowloris (40 conns x 60 s, Mac->Pi:8080)", "mac",
     f"python3 {HERE}/slowloris.py {PI_IP} 8080 40 60", {MAC_IP}),
    ("xmas_scan (nmap -sX, Pi->self)", "pi",
     f"sudo nmap -sX -Pn -p1-1000 --max-rate 200 {PI_IP}", {PI_IP, "127.0.0.1"}),
    ("synflood (hping3 -S, 2k pps x 20 s, Pi->self)", "pi",
     f"sudo timeout 20 /usr/sbin/hping3 -S -p 80 -i u500 {PI_IP} 2>&1 | tail -3",
     {PI_IP, "127.0.0.1"}),
    ("udpflood (hping3 --udp, 2k pps x 20 s, Pi->self)", "pi",
     f"sudo timeout 20 /usr/sbin/hping3 --udp -p 53 -i u500 {PI_IP} 2>&1 | tail -3",
     {PI_IP, "127.0.0.1"}),
    ("icmpflood (hping3 --icmp, 2k pps x 20 s, Pi->self)", "pi",
     f"sudo timeout 20 /usr/sbin/hping3 --icmp -i u500 {PI_IP} 2>&1 | tail -3",
     {PI_IP, "127.0.0.1"}),
]

only = set(sys.argv[1:])
summary = []
for name, where, cmd, sources in ATTACKS:
    if only and not any(o in name for o in only):
        continue
    line0 = alert_count()
    st0 = status()
    t0 = time.time()
    if where == "mac":
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=300)
    else:
        r = pi(cmd, timeout=300)
    t_end = time.time()
    time.sleep(15)                         # let the 1 s flush loop report
    st1 = status()
    alerts = new_alerts(line0)
    mine = [a for a in alerts if a["src_ip"] in sources]
    other = [a for a in alerts if a["src_ip"] not in sources]
    inc = {}
    for a in mine:
        k = a["kind"]
        if k not in inc or a["flows"] >= inc[k]["flows"]:
            inc[k] = a
    first = min((a["ts"] for a in mine), default=None)
    rec = {
        "attack": name, "tool_output": (r.stdout + r.stderr).strip()[-400:],
        "duration_s": round(t_end - t0, 1),
        "labels_from_attacker": {k: {"flows": v["flows"], "pkts": v["pkts"],
                                     "dst_ports": v["dst_ports"], "conf": v["confidence"]}
                                 for k, v in inc.items()},
        "first_alert": first, "attack_start": time.strftime("%H:%M:%S", time.localtime(t0)),
        "other_source_alerts": len(other),
        "sensor_pkts": st1["pkts"] - st0["pkts"],
        "cpu_temp_c": st1["host"]["cpu_temp_c"], "rss_mb": st1["host"]["rss_mb"],
        "load1": st1["host"]["load1"],
    }
    summary.append(rec)
    print(json.dumps(rec, indent=1), flush=True)
    time.sleep(20)

with open(os.path.join(OUT, f"attacks-{time.strftime('%H%M%S')}.json"), "w") as f:
    json.dump(summary, f, indent=1)
