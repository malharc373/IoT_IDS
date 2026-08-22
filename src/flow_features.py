"""
flow_features.py — shared flow-feature extraction for IoT-IDS.

This is the SINGLE SOURCE OF TRUTH for how raw packets become the feature
vector fed to the model.  The trainer, the offline pcap classifier, and the
live sniffer daemon all import from here, which guarantees zero train/serve
skew.

Pipeline:
    raw packet bytes ──parse_raw()──▶ normalized packet dict
    normalized dict  ──FlowTable──▶ bidirectional flow records
    flow record      ──features()──▶ 22-dim float vector (FEATURE_NAMES)

The 22 features are chosen to make different attack classes separable:
  * flow-level shape/rate/flag stats catch floods and malformed flows
  * host-context rolling stats (distinct dst ports / IPs per source) catch
    reconnaissance (port scans) that is invisible at the single-flow level.

Both IPv4 and IPv6 are parsed. No third-party dependency is required for
parsing (pure struct), so this file runs unchanged on a Raspberry Pi.
"""
from __future__ import annotations

import math
import socket
import struct
import threading
import collections
from typing import Dict, List, Optional

# ── Protocol numbers ──────────────────────────────────────────────────────────
PROTO_ICMP = 1
PROTO_TCP = 6
PROTO_UDP = 17
PROTO_ICMPV6 = 58

ETH_IPV4 = 0x0800
ETH_IPV6 = 0x86DD
ETH_VLAN = 0x8100
ETH_QINQ = 0x88A8
LINKTYPE_ETHERNET = 1

# IPv6 extension headers to walk past to reach the transport header
_V6_EXT = {0, 43, 44, 51, 60, 135}   # hop-by-hop, routing, fragment, AH, dstopts, mobility

# TCP flag bit masks
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20

# Rolling window (seconds) used for host-context features.
HOST_WINDOW_S = 60.0

# After a TCP teardown the 5-tuple stays attached to the finished flow for this
# long, so the trailing ACK of a FIN/FIN exchange lands where it belongs instead
# of opening a spurious one-packet flow. A SYN reuses the tuple immediately.
TCP_REUSE_GAP_S = 5.0

# ── Feature vector definition (ORDER MATTERS — do not reorder) ────────────────
FEATURE_NAMES: List[str] = [
    "proto",                # 6 TCP / 17 UDP / 1 ICMP / 0 other
    "duration",             # flow lifetime (s)
    "tot_pkts",             # total packets both directions
    "tot_bytes",            # total bytes both directions
    "pkts_per_sec",         # packet rate
    "bytes_per_sec",        # byte rate
    "mean_pkt_len",         # mean packet size
    "std_pkt_len",          # std packet size
    "min_pkt_len",          # smallest packet
    "max_pkt_len",          # largest packet
    "mean_iat",             # mean inter-arrival time
    "std_iat",              # std inter-arrival time
    "syn_ratio",            # SYN packets / total (TCP)
    "fin_ratio",            # FIN packets / total
    "rst_ratio",            # RST packets / total
    "ack_ratio",            # ACK packets / total
    "fwd_bwd_pkt_ratio",    # fwd_pkts / (bwd_pkts+1) — asymmetry
    "down_up_bytes_ratio",  # bwd_bytes / (fwd_bytes+1)
    "host_dst_ports",       # distinct dst ports this src hit in window
    "host_dst_ips",         # distinct dst IPs this src hit in window
    "host_flow_count",      # flows opened by this src in window
    "dst_port",             # service port the initiator targeted (23/1883/53/…)
]
N_FEATURES = len(FEATURE_NAMES)
# Increment whenever feature *semantics* change without changing names/order.
# Version 2 anchors fwd/bwd and dst_port to the observed connection initiator;
# version 1 incorrectly used lexicographic endpoint ordering.
FEATURE_CONTRACT_VERSION = 2


# ── Packet parsing ────────────────────────────────────────────────────────────
def parse_raw(raw: bytes, orig_len: Optional[int] = None) -> Optional[dict]:
    """Ethernet → IPv4/IPv6 → TCP/UDP/ICMP. Returns a normalized dict or None.

    `orig_len` is the on-the-wire frame length. Pass it when reading a capture
    taken with a snaplen: the stored bytes are truncated, so `len(raw)` would
    understate every packet-length and byte-rate feature (vault/Findings/F16).
    """
    if len(raw) < 14:
        return None
    eth_type = struct.unpack("!H", raw[12:14])[0]
    # Walk up to two stacked VLAN tags (802.1Q / 802.1ad QinQ) transparently.
    offset = 14
    for _ in range(2):
        if eth_type not in (ETH_VLAN, ETH_QINQ):
            break
        if len(raw) < offset + 4:
            return None
        eth_type = struct.unpack("!H", raw[offset + 2:offset + 4])[0]
        offset += 4

    ip = raw[offset:]
    length = orig_len if orig_len else len(raw)

    if eth_type == ETH_IPV4:
        if len(ip) < 20:
            return None
        ihl = (ip[0] & 0x0F) * 4
        if ihl < 20 or len(ip) < ihl:
            return None
        proto = ip[9]
        frag = struct.unpack("!H", ip[6:8])[0]
        # A non-initial fragment has no transport header. Treating its first
        # payload bytes as ports corrupts flow identity and directional stats.
        if frag & 0x1FFF:
            return None
        src_ip = socket.inet_ntoa(ip[12:16])
        dst_ip = socket.inet_ntoa(ip[16:20])
        payload = ip[ihl:]
    elif eth_type == ETH_IPV6:
        if len(ip) < 40:
            return None
        proto = ip[6]
        try:
            src_ip = socket.inet_ntop(socket.AF_INET6, ip[8:24])
            dst_ip = socket.inet_ntop(socket.AF_INET6, ip[24:40])
        except (OSError, ValueError):
            return None
        payload = ip[40:]
        # skip extension headers to reach the transport header
        for _ in range(8):
            if proto not in _V6_EXT or len(payload) < 8:
                break
            if proto == 44:                      # fragment header: fixed 8 bytes
                # Non-initial fragments do not contain a transport header.
                frag = struct.unpack("!H", payload[2:4])[0]
                if frag & 0xFFF8:
                    return None
                nxt, hlen = payload[0], 8
            elif proto == 51:                    # AH: (Payload Len + 2) * 4
                nxt, hlen = payload[0], (payload[1] + 2) * 4
            else:
                nxt, hlen = payload[0], (payload[1] + 1) * 8
            if len(payload) < hlen:
                return None
            proto, payload = nxt, payload[hlen:]
    else:
        return None

    src_port = dst_port = 0
    flags = 0
    if proto == PROTO_TCP and len(payload) >= 14:
        src_port, dst_port = struct.unpack("!HH", payload[0:4])
        flags = payload[13] & 0x3F
    elif proto == PROTO_UDP and len(payload) >= 8:
        src_port, dst_port = struct.unpack("!HH", payload[0:4])
    elif proto in (PROTO_ICMP, PROTO_ICMPV6):
        src_port = dst_port = 0
        # normalise ICMPv6 onto the ICMP feature value so the model sees one
        # protocol identity for "control message" regardless of IP version
        proto = PROTO_ICMP

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "proto": proto,
        "length": length,
        "flags": flags,
    }


def normalize_scapy(pkt) -> Optional[dict]:
    """Convert a scapy packet to the same normalized dict used everywhere.

    Imported lazily so this module has no hard scapy dependency for offline use.
    """
    try:
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.inet6 import IPv6, IPv6ExtHdrFragment
    except Exception:
        return None
    if IP in pkt:
        ip = pkt[IP]
        if int(getattr(ip, "frag", 0)) > 0:
            return None
        proto = ip.proto
    elif IPv6 in pkt:
        ip = pkt[IPv6]
        if IPv6ExtHdrFragment in pkt and int(pkt[IPv6ExtHdrFragment].offset) > 0:
            return None
        proto = ip.nh
    else:
        return None
    src_port = dst_port = 0
    flags = 0
    if TCP in pkt:
        t = pkt[TCP]
        src_port, dst_port = int(t.sport), int(t.dport)
        flags = int(t.flags) & 0x3F
        proto = PROTO_TCP
    elif UDP in pkt:
        u = pkt[UDP]
        src_port, dst_port = int(u.sport), int(u.dport)
        proto = PROTO_UDP
    elif ICMP in pkt or proto == PROTO_ICMPV6:
        proto = PROTO_ICMP
    return {
        "src_ip": ip.src,
        "dst_ip": ip.dst,
        "src_port": src_port,
        "dst_port": dst_port,
        "proto": int(proto),
        "length": len(pkt),
        "flags": flags,
    }


# ── capture readers ───────────────────────────────────────────────────────────
PCAP_MAGICS = {0xA1B2C3D4, 0xA1B23C4D, 0xD4C3B2A1, 0x4D3CB2A1}
PCAPNG_SHB = 0x0A0D0D0A


def read_pcapng(filename: str) -> List[tuple]:
    """Return (ts, raw_bytes, orig_len) from a pcapng file.

    Modern tshark/dumpcap write pcapng by default. The classic reader would
    accept such a file, mis-detect the endianness from the Section Header Block
    magic, and emit garbage records rather than fail — so the format is parsed
    properly here (vault/Findings/F16).

    Handles Enhanced Packet Blocks and Simple Packet Blocks, honours per-
    interface timestamp resolution (if_tsresol), and tolerates multiple
    sections with differing byte order.
    """
    out = []
    with open(filename, "rb") as f:
        endian = "<"
        # per-interface (linktype, snaplen, ts divisor); index is interface id
        ifaces: List[tuple] = []
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            btype = struct.unpack(endian + "I", hdr[0:4])[0]
            if btype == PCAPNG_SHB:
                bom = f.read(4)
                if len(bom) < 4:
                    break
                # byte-order magic decides the endianness for this section
                endian = "<" if struct.unpack("<I", bom)[0] == 0x1A2B3C4D else ">"
                blen = struct.unpack(endian + "I", hdr[4:8])[0]
                if blen < 12:
                    break
                f.read(blen - 12)          # rest of the SHB + trailing length
                ifaces = []
                continue
            blen = struct.unpack(endian + "I", hdr[4:8])[0]
            if blen < 12:
                break
            body = f.read(blen - 12)
            f.read(4)                       # trailing block total length
            if len(body) < blen - 12:
                break

            if btype == 0x00000001:         # Interface Description Block
                linktype = struct.unpack(endian + "H", body[0:2])[0] if len(body) >= 2 else -1
                snaplen = struct.unpack(endian + "I", body[4:8])[0] if len(body) >= 8 else 0
                divisor = 1e6               # if_tsresol default: microseconds
                opt = body[8:]
                while len(opt) >= 4:
                    ocode, olen = struct.unpack(endian + "HH", opt[0:4])
                    if ocode == 0:
                        break
                    val = opt[4:4 + olen]
                    if ocode == 9 and val:  # if_tsresol
                        r = val[0]
                        divisor = float(2 ** (r & 0x7F)) if r & 0x80 else float(10 ** r)
                    opt = opt[4 + olen + ((-olen) % 4):]
                ifaces.append((linktype, snaplen, divisor))

            elif btype == 0x00000006:       # Enhanced Packet Block
                if len(body) < 20:
                    continue
                iid, ts_hi, ts_lo, cap_len, orig_len = struct.unpack(
                    endian + "IIIII", body[0:20])
                if iid >= len(ifaces):
                    raise ValueError(f"{filename}: packet references unknown interface {iid}")
                linktype, _snaplen, divisor = ifaces[iid]
                if linktype != LINKTYPE_ETHERNET:
                    raise ValueError(
                        f"{filename}: unsupported pcapng link type {linktype}; "
                        "only Ethernet (DLT_EN10MB/LINKTYPE_ETHERNET) is supported")
                ts = ((ts_hi << 32) | ts_lo) / divisor
                out.append((ts, body[20:20 + cap_len], orig_len))

            elif btype == 0x00000003:       # Simple Packet Block (no timestamp)
                if len(body) < 4:
                    continue
                orig_len = struct.unpack(endian + "I", body[0:4])[0]
                if not ifaces:
                    raise ValueError(f"{filename}: simple packet block has no interface")
                linktype, snaplen, _divisor = ifaces[0]
                if linktype != LINKTYPE_ETHERNET:
                    raise ValueError(
                        f"{filename}: unsupported pcapng link type {linktype}; "
                        "only Ethernet (DLT_EN10MB/LINKTYPE_ETHERNET) is supported")
                cap_len = min(orig_len, snaplen) if snaplen else orig_len
                out.append((0.0, body[4:4 + cap_len], orig_len))
    return out


# ── pcap reader (classic little/big-endian .pcap; pcapng handled above) ───────
def read_pcap(filename: str) -> List[tuple]:
    """Return a list of (ts, raw_bytes, orig_len) from a capture file.

    Dispatches to read_pcapng() for pcapng input and raises on anything that is
    neither, rather than silently mis-parsing it.
    """
    out = []
    with open(filename, "rb") as f:
        hdr = f.read(24)
        if len(hdr) < 24:
            return out
        magic = struct.unpack("<I", hdr[:4])[0]
        if magic == PCAPNG_SHB:
            return read_pcapng(filename)
        if magic not in PCAP_MAGICS:
            raise ValueError(
                f"{filename}: not a pcap or pcapng file "
                f"(magic 0x{magic:08X}). Classic pcap and pcapng are supported.")
        endian = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
        ts_divisor = 1e9 if magic in (0xA1B23C4D, 0x4D3CB2A1) else 1e6
        linktype = struct.unpack(endian + "I", hdr[20:24])[0]
        if linktype != LINKTYPE_ETHERNET:
            raise ValueError(
                f"{filename}: unsupported pcap link type {linktype}; only "
                "Ethernet (DLT_EN10MB/LINKTYPE_ETHERNET) is supported")
        while True:
            rec = f.read(16)
            if len(rec) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", rec)
            raw = f.read(incl_len)
            if len(raw) < incl_len:
                break
            # orig_len is the on-the-wire length; incl_len is what the snaplen
            # let through. Callers need the former or every length-derived
            # feature shrinks silently on a snaplen'd capture (F16).
            out.append((ts_sec + ts_usec / ts_divisor, raw, orig_len))
    return out


# ── Flow record ───────────────────────────────────────────────────────────────
class Flow:
    __slots__ = (
        "src_ip", "dst_ip", "src_port", "dst_port", "proto",
        "ts_start", "ts_end", "last_ts",
        "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
        "lengths", "iats",
        "syn", "fin", "rst", "ack",
        "dirty", "ctx_sig", "fin_fwd", "fin_bwd", "closed",
    )

    def __init__(self, pkt, ts):
        # forward direction = whoever sent the first packet
        self.src_ip = pkt["src_ip"]
        self.dst_ip = pkt["dst_ip"]
        self.src_port = pkt["src_port"]
        self.dst_port = pkt["dst_port"]
        self.proto = pkt["proto"]
        self.ts_start = ts
        self.ts_end = ts
        self.last_ts = ts
        self.fwd_pkts = 0
        self.bwd_pkts = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.lengths: List[int] = []
        self.iats: List[float] = []
        self.syn = self.fin = self.rst = self.ack = 0
        # set whenever the flow gains a packet; cleared once its feature vector
        # has been handed out. Lets the live path re-score only what changed.
        self.dirty = True
        # host-context triple used the last time this flow was scored. Three of
        # the 22 features are host-context, so a flow's vector changes when its
        # PEERS change even if the flow itself gained no packets — the cache
        # must invalidate on that too, or a scan's verdicts go stale.
        self.ctx_sig = None
        # TCP teardown tracking: a flow is closed by RST, or by FIN in both
        # directions. A later packet on the same 5-tuple then starts a NEW
        # flow instead of being merged into the finished one (F16) — but only
        # once it actually looks like a new connection, see _starts_new_flow.
        self.fin_fwd = False
        self.fin_bwd = False
        self.closed = False

    def update(self, pkt, ts, forward: bool):
        self.dirty = True
        if ts > self.last_ts:
            self.iats.append(ts - self.last_ts)
        self.last_ts = ts
        self.ts_end = ts
        self.lengths.append(pkt["length"])
        if forward:
            self.fwd_pkts += 1
            self.fwd_bytes += pkt["length"]
        else:
            self.bwd_pkts += 1
            self.bwd_bytes += pkt["length"]
        fl = pkt["flags"]
        if fl & SYN:
            self.syn += 1
        if fl & FIN:
            self.fin += 1
        if fl & RST:
            self.rst += 1
            self.closed = True
        if fl & ACK:
            self.ack += 1
        if fl & FIN:
            if forward:
                self.fin_fwd = True
            else:
                self.fin_bwd = True
            if self.fin_fwd and self.fin_bwd:
                self.closed = True

    @property
    def tot_pkts(self) -> int:
        return self.fwd_pkts + self.bwd_pkts

    def features(self, host_ctx: dict) -> List[float]:
        n = self.tot_pkts
        dur = max(self.ts_end - self.ts_start, 1e-6)
        lengths = self.lengths
        mean_len = sum(lengths) / n if n else 0.0
        var_len = sum((x - mean_len) ** 2 for x in lengths) / n if n else 0.0
        std_len = math.sqrt(var_len)
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        if self.iats:
            mean_iat = sum(self.iats) / len(self.iats)
            var_iat = sum((x - mean_iat) ** 2 for x in self.iats) / len(self.iats)
            std_iat = math.sqrt(var_iat)
        else:
            mean_iat = std_iat = 0.0
        tot_bytes = self.fwd_bytes + self.bwd_bytes
        return [
            float(self.proto),
            round(dur, 6),
            float(n),
            float(tot_bytes),
            round(n / dur, 3),
            round(tot_bytes / dur, 3),
            round(mean_len, 3),
            round(std_len, 3),
            float(min_len),
            float(max_len),
            round(mean_iat, 6),
            round(std_iat, 6),
            round(self.syn / n, 4) if n else 0.0,
            round(self.fin / n, 4) if n else 0.0,
            round(self.rst / n, 4) if n else 0.0,
            round(self.ack / n, 4) if n else 0.0,
            round(self.fwd_pkts / (self.bwd_pkts + 1), 4),
            round(self.bwd_bytes / (self.fwd_bytes + 1), 4),
            float(host_ctx["dst_ports"]),
            float(host_ctx["dst_ips"]),
            float(host_ctx["flow_count"]),
            float(self.dst_port),
        ]


# ── Flow table ────────────────────────────────────────────────────────────────
def _flow_key(pkt):
    """Canonical bidirectional key: sort the two endpoints so A→B and B→A map
    to the same flow.

    The boolean describes only whether this packet matches the canonical sort
    order. It must never be used as packet direction: canonical ordering is a
    storage detail, while forward means initiator → responder.
    """
    a = (pkt["src_ip"], pkt["src_port"])
    b = (pkt["dst_ip"], pkt["dst_port"])
    if a <= b:
        return (a, b, pkt["proto"]), True
    return (b, a, pkt["proto"]), False


class FlowTable:
    """Aggregates packets into bidirectional flows and computes host context.

    Works for both batch (whole pcap) and streaming (live) use.

    THREAD SAFETY (see vault/Findings/F05)
    --------------------------------------
    In live mode scapy's AsyncSniffer calls add_packet() from its own thread
    while the daemon's flush loop iterates the same dict. On CPython that
    raises "dictionary changed size during iteration" — an intermittent crash
    under exactly the traffic volume the sensor exists to handle. Every method
    that touches `self.flows` takes `self.lock`, and the lock is re-entrant and
    public so a caller can hold it across a compound operation.
    """

    def __init__(self):
        # every flow ever seen, keyed by (5-tuple, generation) so a 5-tuple
        # reused after a teardown yields a second, distinct record
        self.flows: Dict[tuple, Flow] = collections.OrderedDict()
        self.active: Dict[tuple, tuple] = {}      # 5-tuple -> current flows key
        self._gen: Dict[tuple, int] = collections.defaultdict(int)
        self.lock = threading.RLock()

    @staticmethod
    def _starts_new_flow(f, pkt, ts):
        """Should this packet open a new flow on a torn-down 5-tuple?

        Yes on a SYN without ACK — that is unambiguously a new connection. Yes
        after a quiet gap, which covers UDP and ICMP where there is no
        handshake to look for. NO otherwise: the trailing ACK that completes a
        FIN/FIN exchange belongs to the flow that just closed, and treating it
        as a new one manufactures a one-packet flow that the model then has to
        classify (it reads as neither benign session nor attack).
        """
        if not f.closed:
            return False
        if pkt["proto"] == PROTO_TCP and (pkt["flags"] & SYN) and not (pkt["flags"] & ACK):
            return True
        return (ts - f.last_ts) > TCP_REUSE_GAP_S

    def add_packet(self, pkt: dict, ts: float):
        with self.lock:
            key, _canonical_order = _flow_key(pkt)
            uk = self.active.get(key)
            f = self.flows.get(uk) if uk is not None else None
            if f is None or self._starts_new_flow(f, pkt, ts):
                gen = self._gen[key]
                self._gen[key] = gen + 1
                uk = (key, gen)
                # First observed sender is the best general fallback for a
                # partial capture. A SYN+ACK is the one case that proves the
                # first sender is the responder, so orient that flow in reverse.
                syn_ack = (pkt["proto"] == PROTO_TCP
                           and (pkt["flags"] & (SYN | ACK)) == (SYN | ACK))
                if syn_ack:
                    initiator = dict(pkt)
                    initiator["src_ip"], initiator["dst_ip"] = (
                        pkt["dst_ip"], pkt["src_ip"])
                    initiator["src_port"], initiator["dst_port"] = (
                        pkt["dst_port"], pkt["src_port"])
                    f = Flow(initiator, ts)
                    forward = False
                else:
                    f = Flow(pkt, ts)
                    forward = True
                self.flows[uk] = f
            else:
                forward = (
                    pkt["src_ip"] == f.src_ip
                    and pkt["src_port"] == f.src_port
                    and pkt["dst_ip"] == f.dst_ip
                    and pkt["dst_port"] == f.dst_port
                )
            self.active[key] = uk
            f.update(pkt, ts, forward)

    def __len__(self):
        with self.lock:
            return len(self.flows)

    # -- host context ----------------------------------------------------------
    def _host_context(self, window: Optional[float] = None) -> Dict[str, dict]:
        """For each source IP, distinct dst ports / dst IPs / flow count.
        `window` limits to flows whose ts_end is within `window` s of the latest
        packet (used live); None = use all flows (batch).

        Caller must hold self.lock.
        """
        latest = max((f.ts_end for f in self.flows.values()), default=0.0)
        per_src = collections.defaultdict(
            lambda: {"dst_ports": set(), "dst_ips": set(), "flow_count": 0}
        )
        for f in self.flows.values():
            if window is not None and (latest - f.ts_end) > window:
                continue
            # attribute host context to the flow *initiator* (fwd src)
            rec = per_src[f.src_ip]
            rec["dst_ports"].add(f.dst_port)
            rec["dst_ips"].add(f.dst_ip)
            rec["flow_count"] += 1
        return {
            ip: {
                "dst_ports": len(v["dst_ports"]),
                "dst_ips": len(v["dst_ips"]),
                "flow_count": v["flow_count"],
            }
            for ip, v in per_src.items()
        }

    def _rows(self, min_pkts, window, clear_dirty):
        """Core extraction. Caller must hold self.lock.

        Yields (key, meta, vector, was_dirty). `window` bounds BOTH the host
        context and the set of flows returned — the pre-2026-08 code applied it
        only to the host context, so every flush re-scored the entire table
        regardless of age (vault/Findings/F04).
        """
        ctx = self._host_context(window=window)
        latest = max((f.ts_end for f in self.flows.values()), default=0.0)
        out = []
        for key, f in self.flows.items():
            if f.tot_pkts < min_pkts:
                continue
            if window is not None and (latest - f.ts_end) > window:
                continue
            host = ctx.get(f.src_ip, {"dst_ports": 1, "dst_ips": 1, "flow_count": 1})
            meta = {
                "src": f"{f.src_ip}:{f.src_port}",
                "dst": f"{f.dst_ip}:{f.dst_port}",
                "proto": f.proto,
                "pkts": f.tot_pkts,
                "bytes": f.fwd_bytes + f.bwd_bytes,
                "ts_start": f.ts_start,
                "ts_end": f.ts_end,
            }
            sig = (host["dst_ports"], host["dst_ips"], host["flow_count"])
            # re-score iff the feature vector could have changed: new packets,
            # or a changed host-context triple, or never scored at all
            needs = f.dirty or f.ctx_sig != sig
            if clear_dirty:
                f.dirty = False
                f.ctx_sig = sig
            out.append((key, meta, f.features(host), needs))
        return out

    def extract(self, min_pkts: int = 1, window: Optional[float] = None):
        """Return list of (flow_meta, feature_vector) for every live flow.

        Batch interface — does not touch the dirty flags.
        """
        with self.lock:
            return [(m, v) for _, m, v, _ in
                    self._rows(min_pkts, window, clear_dirty=False)]

    def extract_live(self, min_pkts: int = 1, window: Optional[float] = None):
        """Return list of (key, meta, vector, needs_scoring) and clear dirty.

        `needs_scoring` is True for flows that gained packets since the last
        call. The daemon scores only those and reuses cached verdicts for the
        rest, so per-flush inference cost tracks new traffic rather than table
        size. Host context is still recomputed over the whole window, because a
        flow's context features change when its *peers* change.
        """
        with self.lock:
            return self._rows(min_pkts, window, clear_dirty=True)

    def prune(self, older_than: float, now: float):
        """Drop flows idle longer than `older_than` seconds (live memory cap).

        Returns the list of evicted keys so callers can drop cached verdicts.

        All three maps are pruned together. `_gen` is the subtle one: it is
        keyed by the *bare* 5-tuple rather than by flow, so it does not shrink
        when flows are evicted, and on a busy segment every ephemeral source
        port leaves a permanent entry — an unbounded leak in a daemon meant to
        run for weeks on a Pi (vault/Findings/F20). Once a 5-tuple has no live
        flow and the caller has dropped its cached verdicts, restarting its
        generation counter at 0 cannot collide with anything, so the entry is
        simply removed.
        """
        with self.lock:
            dead = [k for k, f in self.flows.items() if (now - f.last_ts) > older_than]
            for k in dead:
                del self.flows[k]
            live = set(self.flows)
            for base, uk in list(self.active.items()):
                if uk not in live:
                    del self.active[base]
            live_bases = {base for base, _gen_no in live}
            for base in list(self._gen):
                if base not in live_bases:
                    del self._gen[base]
            return dead


# ── Convenience: pcap → features ──────────────────────────────────────────────
def features_from_pcap(pcap_path: str, min_pkts: int = 1):
    """Parse a .pcap and return (meta, vector) list. Batch/host-context = all."""
    table = FlowTable()
    for ts, raw, orig_len in read_pcap(pcap_path):
        pkt = parse_raw(raw, orig_len)
        if pkt is not None:
            table.add_packet(pkt, ts)
    return table.extract(min_pkts=min_pkts, window=None)


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python flow_features.py <capture.pcap>")
        raise SystemExit(1)
    rows = features_from_pcap(sys.argv[1])
    print(f"Flows: {len(rows)}  |  Features/flow: {N_FEATURES}")
    for meta, vec in rows[:5]:
        print(json.dumps(meta), "->", [round(x, 3) for x in vec])
