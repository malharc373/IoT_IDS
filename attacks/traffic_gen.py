"""
traffic_gen.py — synthesize labeled network traffic as .pcap files.

Writing pcaps with scapy does NOT require root (only sending/sniffing does),
so this runs anywhere and produces a fully reproducible, labeled corpus for
training and demoing the IDS.

Each generator returns a list of scapy packets with realistic timestamps,
sizes, and TCP flag sequences.  Six attack classes + benign background:

    benign         normal TCP sessions, DNS, HTTP-like request/response
    portscan       one host SYN-scanning many ports of a target (recon)
    synflood       high-rate half-open SYNs to one service (DoS)
    icmpflood      high-rate ICMP echo to one target (volumetric DoS)
    udpflood       high-rate UDP to one service port (volumetric DoS)
    ssh_bruteforce many short auth sessions against tcp/22
    slowloris      few very long-lived, low-and-slow partial HTTP flows

Real-tool equivalents (used on the Pi where root is available) are documented
in attacks/README.md.

CLI:
    python attacks/traffic_gen.py --kind portscan --out out.pcap --seed 1
"""
from __future__ import annotations

import argparse
import random
from typing import List

from scapy.all import Ether, IP, TCP, UDP, ICMP, Raw, wrpcap

ATTACK_KINDS = [
    "benign", "portscan", "synflood", "icmpflood",
    "udpflood", "ssh_bruteforce", "slowloris",
]

# Label ids (0 = benign). Kept explicit so the trainer + daemon agree.
LABELS = {k: i for i, k in enumerate(ATTACK_KINDS)}


def _mac(rng) -> str:
    return "02:%02x:%02x:%02x:%02x:%02x" % tuple(rng.randint(0, 255) for _ in range(5))


def _lan_ip(rng, subnet="192.168.1") -> str:
    return f"{subnet}.{rng.randint(2, 254)}"


def _stamp(pkts, t0=0.0):
    """Assign monotonically increasing timestamps already stored per-packet.
    Packets carry a relative offset in .time; shift by a random epoch base."""
    base = 1_700_000_000.0 + t0
    for p in pkts:
        p.time = base + float(p.time)
    return pkts


# ── BENIGN ────────────────────────────────────────────────────────────────────
def gen_benign(rng, n_flows=40) -> List:
    pkts = []
    t = 0.0
    server_ips = [_lan_ip(rng) for _ in range(4)]
    for _ in range(n_flows):
        cli, srv = _mac(rng), _mac(rng)
        c_ip, s_ip = _lan_ip(rng), rng.choice(server_ips)
        sport = rng.randint(1024, 65535)
        kind = rng.random()
        t += rng.uniform(0.01, 0.4)

        if kind < 0.25:
            # DNS query/response (UDP/53)
            q = Ether(src=cli, dst=srv) / IP(src=c_ip, dst=s_ip) / \
                UDP(sport=sport, dport=53) / Raw(load=bytes(rng.randint(40, 70)))
            q.time = t
            r = Ether(src=srv, dst=cli) / IP(src=s_ip, dst=c_ip) / \
                UDP(sport=53, dport=sport) / Raw(load=bytes(rng.randint(80, 200)))
            r.time = t + rng.uniform(0.002, 0.03)
            pkts += [q, r]
            continue

        # TCP session: handshake → data exchange → teardown
        dport = rng.choice([80, 443, 8080, 1883, 8883])  # incl. MQTT IoT ports
        rtt = rng.uniform(0.002, 0.05)
        seq_t = t

        def add(src_mac, dst_mac, sip, dip, sp, dp, flags, size, dt):
            nonlocal seq_t
            seq_t += dt
            p = Ether(src=src_mac, dst=dst_mac) / IP(src=sip, dst=dip) / \
                TCP(sport=sp, dport=dp, flags=flags)
            if size > 0:
                p = p / Raw(load=bytes(size))
            p.time = seq_t
            pkts.append(p)

        add(cli, srv, c_ip, s_ip, sport, dport, "S", 0, 0)
        add(srv, cli, s_ip, c_ip, dport, sport, "SA", 0, rtt)
        add(cli, srv, c_ip, s_ip, sport, dport, "A", 0, rtt)
        # request/response rounds
        for _ in range(rng.randint(2, 6)):
            add(cli, srv, c_ip, s_ip, sport, dport, "PA", rng.randint(40, 300),
                rng.uniform(0.005, 0.08))
            add(srv, cli, s_ip, c_ip, dport, sport, "PA", rng.randint(200, 1400),
                rng.uniform(0.005, 0.06))
        add(cli, srv, c_ip, s_ip, sport, dport, "FA", 0, rng.uniform(0.01, 0.1))
        add(srv, cli, s_ip, c_ip, dport, sport, "FA", 0, rtt)
        add(cli, srv, c_ip, s_ip, sport, dport, "A", 0, rtt)
    return _stamp(pkts, rng.uniform(0, 5))


# ── PORT SCAN ─────────────────────────────────────────────────────────────────
def gen_portscan(rng, n_ports=None) -> List:
    pkts = []
    attacker_mac, victim_mac = _mac(rng), _mac(rng)
    a_ip, v_ip = _lan_ip(rng), _lan_ip(rng)
    n_ports = n_ports or rng.randint(150, 600)
    ports = rng.sample(range(1, 9000), n_ports)
    t = 0.0
    sport = rng.randint(1024, 65535)
    for dp in ports:
        t += rng.uniform(0.0005, 0.006)     # fast sweep
        syn = Ether(src=attacker_mac, dst=victim_mac) / IP(src=a_ip, dst=v_ip) / \
            TCP(sport=sport, dport=dp, flags="S")
        syn.time = t
        pkts.append(syn)
        # most ports closed → RST; a few open → SYN-ACK
        if rng.random() < 0.1:
            rep = Ether(src=victim_mac, dst=attacker_mac) / IP(src=v_ip, dst=a_ip) / \
                TCP(sport=dp, dport=sport, flags="SA")
        else:
            rep = Ether(src=victim_mac, dst=attacker_mac) / IP(src=v_ip, dst=a_ip) / \
                TCP(sport=dp, dport=sport, flags="RA")
        rep.time = t + rng.uniform(0.0002, 0.002)
        pkts.append(rep)
        sport = (sport + 1) if sport < 65535 else 1024
    return _stamp(pkts, rng.uniform(0, 5))


# ── SYN FLOOD ─────────────────────────────────────────────────────────────────
def gen_synflood(rng, n=None) -> List:
    pkts = []
    victim_mac = _mac(rng)
    v_ip = _lan_ip(rng)
    dport = rng.choice([80, 443, 22, 8080])
    n = n or rng.randint(800, 2500)
    # classic flood: a handful of (possibly spoofed) sources, ONE target port,
    # extremely high rate, no completion.
    srcs = [(_mac(rng), _lan_ip(rng)) for _ in range(rng.randint(1, 4))]
    t = 0.0
    for _ in range(n):
        smac, sip = rng.choice(srcs)
        t += rng.uniform(0.00005, 0.0008)   # very fast
        p = Ether(src=smac, dst=victim_mac) / IP(src=sip, dst=v_ip) / \
            TCP(sport=rng.randint(1024, 65535), dport=dport, flags="S")
        p.time = t
        pkts.append(p)
    return _stamp(pkts, rng.uniform(0, 5))


# ── ICMP FLOOD ────────────────────────────────────────────────────────────────
def gen_icmpflood(rng, n=None) -> List:
    # Botnet-style: several sources hammer one victim with ICMP echo, so the
    # capture contains many (src->victim) flows rather than a single giant one.
    pkts = []
    v_mac = _mac(rng)
    v_ip = _lan_ip(rng)
    n_sources = rng.randint(20, 45)
    per_src = rng.randint(30, 120)
    t = 0.0
    for _ in range(n_sources):
        a_mac, a_ip = _mac(rng), _lan_ip(rng)
        for _ in range(per_src):
            t += rng.uniform(0.00008, 0.001)
            size = rng.choice([56, 64, 128, 512, 1024])
            p = Ether(src=a_mac, dst=v_mac) / IP(src=a_ip, dst=v_ip) / \
                ICMP(type=8) / Raw(load=bytes(size))
            p.time = t
            pkts.append(p)
    return _stamp(pkts, rng.uniform(0, 5))


# ── UDP FLOOD ─────────────────────────────────────────────────────────────────
def gen_udpflood(rng, n=None) -> List:
    pkts = []
    a_mac, v_mac = _mac(rng), _mac(rng)
    a_ip, v_ip = _lan_ip(rng), _lan_ip(rng)
    dport = rng.choice([53, 123, 1900, 5353, 19])
    n = n or rng.randint(700, 2200)
    t = 0.0
    for _ in range(n):
        t += rng.uniform(0.00006, 0.0009)
        p = Ether(src=a_mac, dst=v_mac) / IP(src=a_ip, dst=v_ip) / \
            UDP(sport=rng.randint(1024, 65535), dport=dport) / \
            Raw(load=bytes(rng.randint(64, 1400)))
        p.time = t
        pkts.append(p)
    return _stamp(pkts, rng.uniform(0, 5))


# ── SSH BRUTE-FORCE ───────────────────────────────────────────────────────────
def gen_ssh_bruteforce(rng, n_attempts=None) -> List:
    pkts = []
    a_mac, v_mac = _mac(rng), _mac(rng)
    a_ip, v_ip = _lan_ip(rng), _lan_ip(rng)
    n_attempts = n_attempts or rng.randint(40, 120)
    t = 0.0
    for _ in range(n_attempts):
        sport = rng.randint(1024, 65535)
        t += rng.uniform(0.02, 0.15)
        rtt = rng.uniform(0.002, 0.02)
        seq_t = t

        def add(smac, dmac, sip, dip, sp, dp, flags, size, dt):
            nonlocal seq_t
            seq_t += dt
            p = Ether(src=smac, dst=dmac) / IP(src=sip, dst=dip) / \
                TCP(sport=sp, dport=dp, flags=flags)
            if size:
                p = p / Raw(load=bytes(size))
            p.time = seq_t
            pkts.append(p)

        # short session: connect, banner, a couple auth packets, server RST
        add(a_mac, v_mac, a_ip, v_ip, sport, 22, "S", 0, 0)
        add(v_mac, a_mac, v_ip, a_ip, 22, sport, "SA", 0, rtt)
        add(a_mac, v_mac, a_ip, v_ip, sport, 22, "A", 0, rtt)
        add(v_mac, a_mac, v_ip, a_ip, 22, sport, "PA", rng.randint(20, 40), rtt)   # banner
        add(a_mac, v_mac, a_ip, v_ip, sport, 22, "PA", rng.randint(20, 60), rtt)   # auth attempt
        add(v_mac, a_mac, v_ip, a_ip, 22, sport, "PA", rng.randint(20, 40), rtt)   # deny
        add(v_mac, a_mac, v_ip, a_ip, 22, sport, "R", 0, rng.uniform(0.005, 0.03)) # reset
    return _stamp(pkts, rng.uniform(0, 5))


# ── SLOWLORIS ─────────────────────────────────────────────────────────────────
def gen_slowloris(rng, n_conns=None) -> List:
    pkts = []
    a_mac, v_mac = _mac(rng), _mac(rng)
    a_ip, v_ip = _lan_ip(rng), _lan_ip(rng)
    n_conns = n_conns or rng.randint(20, 60)
    for _ in range(n_conns):
        sport = rng.randint(1024, 65535)
        t = rng.uniform(0, 2)
        rtt = rng.uniform(0.002, 0.02)
        p = Ether(src=a_mac, dst=v_mac) / IP(src=a_ip, dst=v_ip) / \
            TCP(sport=sport, dport=80, flags="S"); p.time = t; pkts.append(p)
        p = Ether(src=v_mac, dst=a_mac) / IP(src=v_ip, dst=a_ip) / \
            TCP(sport=80, dport=sport, flags="SA"); p.time = t + rtt; pkts.append(p)
        p = Ether(src=a_mac, dst=v_mac) / IP(src=a_ip, dst=v_ip) / \
            TCP(sport=sport, dport=80, flags="A"); p.time = t + 2 * rtt; pkts.append(p)
        # trickle partial headers slowly to keep the socket open
        tt = t + 2 * rtt
        for _ in range(rng.randint(6, 15)):
            tt += rng.uniform(5.0, 15.0)     # very slow
            p = Ether(src=a_mac, dst=v_mac) / IP(src=a_ip, dst=v_ip) / \
                TCP(sport=sport, dport=80, flags="PA") / Raw(load=bytes(rng.randint(2, 12)))
            p.time = tt; pkts.append(p)
            p = Ether(src=v_mac, dst=a_mac) / IP(src=v_ip, dst=a_ip) / \
                TCP(sport=80, dport=sport, flags="A"); p.time = tt + rtt; pkts.append(p)
    pkts.sort(key=lambda x: x.time)
    return _stamp(pkts, rng.uniform(0, 5))


GENERATORS = {
    "benign": gen_benign,
    "portscan": gen_portscan,
    "synflood": gen_synflood,
    "icmpflood": gen_icmpflood,
    "udpflood": gen_udpflood,
    "ssh_bruteforce": gen_ssh_bruteforce,
    "slowloris": gen_slowloris,
}


def generate(kind: str, seed: int = 0) -> List:
    rng = random.Random(seed)
    return GENERATORS[kind](rng)


def main():
    ap = argparse.ArgumentParser(description="Synthesize labeled traffic pcaps")
    ap.add_argument("--kind", required=True, choices=ATTACK_KINDS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pkts = generate(args.kind, args.seed)
    wrpcap(args.out, pkts)
    print(f"[{args.kind}] wrote {len(pkts)} packets -> {args.out}")


if __name__ == "__main__":
    main()
