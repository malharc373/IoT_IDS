#!/usr/bin/env python3
"""live_demo.py — one-command live demonstration of the IoT-IDS.

Drives a *running* sensor (`ids_daemon.py --iface ...`) with real traffic so an
audience can watch the dashboard react: benign background chatter the whole
time, then every attack class the model knows, one at a time, with a gap after
each so its incident is easy to see.

    benign portscan xmas_scan synflood udpflood icmpflood
    ssh_bruteforce slowloris mqtt_flood mirai

Run it ON the same box as the sensor (simplest for a demo — all traffic stays on
the machine and the floods hit loopback):

    # terminal 1: the sensor
    sudo python src/ids_daemon.py --iface eth0,lo --log logs/alerts.jsonl
    # terminal 2: the demo
    sudo python attacks/live_demo.py            # targets 127.0.0.1 by default

`sudo` is only needed for the raw-socket floods (hping3) and the SYN/Xmas scans;
without it those classes are skipped with a note and everything else still runs.

SAFETY / AUTHORIZED USE ONLY
----------------------------
This program emits hostile traffic. By default it targets loopback (127.0.0.1),
so nothing leaves the machine. Pointing it at any other host — with --target —
requires --i-own-the-target, and it still refuses to run the packet *floods*
against a non-local address unless you also pass --allow-remote-flood. Only ever
aim it at systems you own or are explicitly authorized to test in a lab you
control. See attacks/README.md.
"""
import argparse
import ipaddress
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOLD = "\033[1m"; CYA = "\033[36m"; YEL = "\033[33m"; GRN = "\033[32m"; NC = "\033[0m"


def say(msg, colour=CYA):
    print(f"{colour}{msg}{NC}", flush=True)


def have(tool):
    return shutil.which(tool) is not None


def is_local(target):
    """True if `target` is a loopback address or one of this host's own IPs."""
    try:
        addr = ipaddress.ip_address(target)
    except ValueError:
        return False
    if addr.is_loopback:
        return True
    own = {"127.0.0.1", "::1"}
    try:
        own.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        own.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return target in own


# ── benign background traffic ─────────────────────────────────────────────────
class Benign(threading.Thread):
    """Low-rate legitimate traffic to the target, for the whole demo.

    Real sockets, no root: short TCP connects to common service ports and a few
    UDP/DNS-shaped datagrams. This is what a normal device on the LAN looks like,
    and it lets the audience see that the IDS leaves ordinary traffic alone.
    """
    def __init__(self, target, stop):
        super().__init__(daemon=True)
        self.target = target
        self.stop = stop
        self.sent = 0

    def run(self):
        ports = [80, 443, 8080, 1883, 22, 53]
        while not self.stop.is_set():
            port = random.choice(ports)
            try:
                if port == 53:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.settimeout(0.5)
                    s.sendto(b"\x00\x01" + os.urandom(10), (self.target, port))
                    s.close()
                else:
                    s = socket.create_connection((self.target, port), timeout=0.5)
                    s.sendall(b"GET / HTTP/1.1\r\nHost: demo\r\n\r\n")
                    s.recv(64)
                    s.close()
                self.sent += 1
            except OSError:
                self.sent += 1          # a refused connect is still benign traffic
            time.sleep(random.uniform(0.3, 1.2))


# ── attacks ───────────────────────────────────────────────────────────────────
def run_cmd(cmd, timeout):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "(timed out — expected for held-open attacks)"
    except Exception as e:                                    # noqa: BLE001
        return f"(error: {e})"


def portscan(t, fast):
    if not have("nmap"):
        return "skip", "nmap not installed"
    n = 300 if fast else 1000
    return "portscan", run_cmd(f"nmap -sT -Pn -p1-{n} --max-rate 300 {t}", 120)


def xmas(t, fast):
    if not have("nmap"):
        return "skip", "nmap not installed"
    if os.geteuid() != 0:
        return "skip", "Xmas scan needs root (raw packets)"
    n = 300 if fast else 1000
    return "xmas_scan", run_cmd(f"nmap -sX -Pn -p1-{n} --max-rate 300 {t}", 120)


def _hping(t, args, secs):
    if not have("hping3"):
        return None, "hping3 not installed (apt install hping3)"
    if os.geteuid() != 0:
        return None, "flood needs root (raw sockets)"
    hp = shutil.which("hping3")
    return "ok", run_cmd(f"timeout {secs} {hp} {args} {t} 2>&1 | tail -2", secs + 5)


def synflood(t, fast):
    ok, out = _hping(t, "-S -p 80 -i u500", 15 if fast else 20)
    return ("synflood" if ok else "skip"), out


def udpflood(t, fast):
    ok, out = _hping(t, "--udp -p 53 -i u500", 15 if fast else 20)
    return ("udpflood" if ok else "skip"), out


def icmpflood(t, fast):
    ok, out = _hping(t, "--icmp -i u500", 15 if fast else 20)
    return ("icmpflood" if ok else "skip"), out


def ssh_bruteforce(t, fast):
    n = 20 if fast else 40
    cmd = (f"for i in $(seq 1 {n}); do "
           f"ssh -o BatchMode=yes -o PubkeyAuthentication=no "
           f"-o PreferredAuthentications=password -o ConnectTimeout=3 "
           f"-o StrictHostKeyChecking=no nosuchuser@{t} true 2>/dev/null; "
           f"sleep 0.2; done; echo '{n} login attempts'")
    return "ssh_bruteforce", run_cmd(cmd, 120)


def slowloris(t, fast):
    conns, dur = (20, 20) if fast else (40, 40)
    port = 8080
    return "slowloris", run_cmd(
        f"python3 {HERE}/slowloris.py {t} {port} {conns} {dur}", dur + 15)


def mqtt_flood(t, fast):
    """A burst of MQTT-shaped short TCP connects to the broker port (1883).

    Uses raw sockets so it needs neither mosquitto nor root; if the port is
    closed the rapid SYN/RST bursts still produce the flood flow signature.
    """
    n = 200 if fast else 500
    end = time.time() + (8 if fast else 12)
    made = 0
    while time.time() < end and made < n:
        try:
            s = socket.create_connection((t, 1883), timeout=0.3)
            # a minimal MQTT CONNECT-ish payload, then drop it
            s.sendall(bytes([0x10, 0x0c, 0x00, 0x04]) + b"MQTT" + os.urandom(6))
            s.close()
        except OSError:
            pass
        made += 1
    return "mqtt_flood", f"opened ~{made} MQTT connections to {t}:1883"


def mirai(t, fast):
    """Mirai-style telnet spread: rapid connects to 23/2323 (the ports Mirai
    scans for). Real sockets, no root; models the many-short-flow signature."""
    n = 150 if fast else 300
    made = 0
    end = time.time() + (8 if fast else 12)
    while time.time() < end and made < n:
        for port in (23, 2323):
            try:
                s = socket.create_connection((t, port), timeout=0.3)
                s.sendall(b"root\r\n" + os.urandom(4))
                s.close()
            except OSError:
                pass
            made += 1
    return "mirai", f"~{made} telnet (23/2323) connection attempts to {t}"


# name -> (function, one-line description shown before it runs)
ATTACKS = [
    ("portscan",       portscan,       "TCP connect scan (nmap -sT)"),
    ("xmas_scan",      xmas,           "Xmas scan (nmap -sX, root)"),
    ("synflood",       synflood,       "SYN flood (hping3 -S, root)"),
    ("udpflood",       udpflood,       "UDP flood (hping3 --udp, root)"),
    ("icmpflood",      icmpflood,      "ICMP flood (hping3 --icmp, root)"),
    ("ssh_bruteforce", ssh_bruteforce, "SSH brute force (failed logins)"),
    ("slowloris",      slowloris,      "Slowloris (held-open HTTP)"),
    ("mqtt_flood",     mqtt_flood,     "MQTT connection flood (:1883)"),
    ("mirai",          mirai,          "Mirai-style telnet spread (23/2323)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="127.0.0.1",
                    help="host to attack (default: loopback, stays on this box)")
    ap.add_argument("--only", nargs="*", metavar="CLASS",
                    help="run only these classes (default: all)")
    ap.add_argument("--fast", action="store_true",
                    help="shorter attacks — good for a rehearsal")
    ap.add_argument("--gap", type=float, default=20.0,
                    help="seconds of benign-only traffic between attacks")
    ap.add_argument("--no-benign", action="store_true",
                    help="do not run the benign background thread")
    ap.add_argument("--i-own-the-target", action="store_true",
                    help="required to aim at any non-local host")
    ap.add_argument("--allow-remote-flood", action="store_true",
                    help="also permit the packet floods against a remote target")
    args = ap.parse_args()

    target = args.target
    local = is_local(target)
    if not local and not args.i_own_the_target:
        say(f"Refusing to attack non-local target {target!r} without "
            f"--i-own-the-target.", YEL)
        say("Only aim this at hosts you own or are authorized to test.", YEL)
        return 2

    selected = ATTACKS
    if args.only:
        want = set(args.only)
        selected = [a for a in ATTACKS if a[0] in want]
        if not selected:
            say(f"No known classes in {sorted(want)}. Known: "
                f"{[a[0] for a in ATTACKS]}", YEL)
            return 2

    say(f"\n{BOLD}IoT-IDS live attack demo{NC}", BOLD)
    say(f"  target        : {target} ({'local — traffic stays on this box' if local else 'REMOTE'})")
    say(f"  classes       : {', '.join(a[0] for a in selected)}")
    say(f"  benign traffic: {'off' if args.no_benign else 'on (background)'}")
    say(f"  root          : {'yes' if os.geteuid() == 0 else 'no — floods/Xmas will be skipped'}")
    say("  Watch the dashboard / `tail -f logs/alerts.jsonl` while this runs.\n")

    stop = threading.Event()
    bg = None
    if not args.no_benign:
        bg = Benign(target, stop)
        bg.start()
        say("Benign background traffic started; warming up 5 s...", GRN)
        time.sleep(5)

    try:
        for name, fn, desc in selected:
            floodish = name in ("synflood", "udpflood", "icmpflood")
            if floodish and not local and not args.allow_remote_flood:
                say(f"— skip {name}: remote flood needs --allow-remote-flood", YEL)
                continue
            say(f"\n{BOLD}▶ {name}{NC} — {desc}", CYA)
            t0 = time.time()
            label, out = fn(target, args.fast)
            dt = time.time() - t0
            tail = (out or "").splitlines()[-1:] or [""]
            if label == "skip":
                say(f"  skipped: {out}", YEL)
            else:
                say(f"  done in {dt:.0f}s. tool said: {tail[0][:100]}", GRN)
                say(f"  → expect an incident labelled ~'{name}' on the dashboard.")
            say(f"  benign-only for {args.gap:.0f}s (watch the alert settle)...")
            time.sleep(args.gap)
    except KeyboardInterrupt:
        say("\nInterrupted.", YEL)
    finally:
        stop.set()
        if bg:
            bg.join(timeout=2)
            say(f"\nBenign thread sent ~{bg.sent} legitimate exchanges.", GRN)

    say(f"\n{BOLD}Demo complete.{NC} Review logs/alerts.jsonl or the dashboard "
        f"for the incident timeline.", GRN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
