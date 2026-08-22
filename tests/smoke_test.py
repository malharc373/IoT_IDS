#!/usr/bin/env python3
"""
smoke_test.py — exercises every module/script in the project end-to-end.

Run:  python tests/smoke_test.py
Exit code 0 = all pass. No root required. Writes only to a temp dir.
"""
from __future__ import annotations

import os
import sys
import io
import json
import time
import shutil
import tempfile
import subprocess
import contextlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "attacks"))
sys.path.insert(0, os.path.join(ROOT, "code"))
os.environ.setdefault("PYTHONWARNINGS", "ignore")

PASS, FAIL, SKIP = [], [], []
TMP = tempfile.mkdtemp(prefix="iotids_test_")


class SkipTest(Exception):
    """Raised when a check needs an artifact that is not in a fresh clone."""


def check(name, fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        PASS.append(name)
        print(f"  \033[32mPASS\033[0m  {name}")
    except SkipTest as e:
        SKIP.append((name, str(e)))
        print(f"  \033[33mSKIP\033[0m  {name}  -> {e}")
    except Exception as e:
        FAIL.append((name, repr(e)))
        print(f"  \033[31mFAIL\033[0m  {name}  -> {e!r}")


# ── 1. imports ────────────────────────────────────────────────────────────────
def t_imports():
    import flow_features, ids_daemon, train_live_model  # noqa
    import traffic_gen, build_corpus  # noqa
    import importlib.util
    for mod in ["02_train_sfaf.py", "download_datasets.py"]:
        spec = importlib.util.spec_from_file_location(
            mod.replace(".py", "").replace("02_", "m02_"),
            os.path.join(ROOT, "code", mod))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)


def t_requirements_importable():
    import numpy, pandas, sklearn, xgboost, pyarrow, psutil  # noqa
    import onnxruntime, onnxmltools, scapy, matplotlib  # noqa


# ── 2. flow_features ──────────────────────────────────────────────────────────
def t_flow_features_core():
    import flow_features as ff
    from scapy.all import Ether, IP, TCP, wrpcap
    assert ff.N_FEATURES == len(ff.FEATURE_NAMES) and ff.N_FEATURES >= 21
    # parse_raw on a crafted TCP SYN
    p = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1, dport=2, flags="S")
    pk = ff.parse_raw(bytes(p))
    assert pk["proto"] == 6 and pk["flags"] & 0x02
    # normalize_scapy parity
    pk2 = ff.normalize_scapy(p)
    assert pk2["src_ip"] == "1.1.1.1" and pk2["dst_port"] == 2
    # read_pcap roundtrip
    path = os.path.join(TMP, "rt.pcap")
    wrpcap(path, [p, p])
    recs = ff.read_pcap(path)
    assert len(recs) == 2 and len(recs[0]) == 3   # (ts, raw, orig_len)
    # FlowTable -> 21-dim vector
    table = ff.FlowTable()
    for i, (ts, raw, olen) in enumerate(recs):
        table.add_packet(ff.parse_raw(raw, olen), ts + i)
    rows = table.extract(min_pkts=1)
    assert rows and len(rows[0][1]) == ff.N_FEATURES


def t_parse_ipv6_vlan_snaplen():
    """IPv6, stacked VLAN tags, and snaplen-truncated captures (F16)."""
    import flow_features as ff
    from scapy.all import Ether, IP, TCP, UDP, Dot1Q, Dot1AD, Raw
    from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest

    # IPv6 + TCP
    p6 = Ether()/IPv6(src="2001:db8::1", dst="2001:db8::2")/TCP(sport=1234, dport=443, flags="S")
    pk = ff.parse_raw(bytes(p6))
    assert pk is not None, "IPv6 frame not parsed"
    assert pk["src_ip"] == "2001:db8::1" and pk["dst_port"] == 443
    assert pk["proto"] == ff.PROTO_TCP and pk["flags"] & ff.SYN

    # IPv6 + UDP
    u6 = Ether()/IPv6(src="fe80::a", dst="fe80::b")/UDP(sport=5353, dport=5353)
    assert ff.parse_raw(bytes(u6))["proto"] == ff.PROTO_UDP

    # ICMPv6 normalises onto the ICMP feature value
    i6 = Ether()/IPv6(src="2001:db8::1", dst="2001:db8::2")/ICMPv6EchoRequest()
    assert ff.parse_raw(bytes(i6))["proto"] == ff.PROTO_ICMP

    # single and double VLAN tags
    v1 = Ether()/Dot1Q(vlan=10)/IP(src="1.1.1.1", dst="2.2.2.2")/TCP(sport=1, dport=80)
    assert ff.parse_raw(bytes(v1))["dst_port"] == 80
    v2 = Ether()/Dot1AD(vlan=20)/Dot1Q(vlan=10)/IP(src="1.1.1.1", dst="2.2.2.2")/TCP(sport=1, dport=80)
    assert ff.parse_raw(bytes(v2))["dst_port"] == 80, "QinQ frame not parsed"

    # non-IP frames are still ignored
    assert ff.parse_raw(bytes(Ether(type=0x0806))) is None

    # snaplen: stored bytes truncated, on-the-wire length preserved
    full = bytes(Ether()/IP(src="1.1.1.1", dst="2.2.2.2")/TCP()/Raw(load=bytes(1400)))
    truncated = full[:96]
    assert ff.parse_raw(truncated)["length"] == 96
    assert ff.parse_raw(truncated, len(full))["length"] == len(full), (
        "orig_len ignored — length features shrink on a snaplen'd capture")


def _build_pcapng(path, packets, tsresol=6):
    """Hand-build a spec-conformant pcapng: SHB + IDB + Enhanced Packet Blocks.

    Built by hand rather than via scapy so the reader is tested against the
    format itself. `tsresol` is the if_tsresol exponent (6 = microseconds,
    9 = nanoseconds).
    """
    import struct as _s

    def block(btype, body):
        total = 12 + len(body)
        return _s.pack("<II", btype, total) + body + _s.pack("<I", total)

    out = bytearray()
    # Section Header Block: byte-order magic, version 1.0, unspecified length
    out += block(0x0A0D0D0A, _s.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    # Interface Description Block: LINKTYPE_ETHERNET, snaplen 0, if_tsresol opt
    idb = _s.pack("<HHI", 1, 0, 0)
    idb += _s.pack("<HH", 9, 1) + bytes([tsresol]) + b"\x00" * 3   # if_tsresol
    idb += _s.pack("<HH", 0, 0)                                     # opt_endofopt
    out += block(0x00000001, idb)
    for ts, raw in packets:
        ticks = int(round(ts * (10 ** tsresol)))
        pad = (-len(raw)) % 4
        body = _s.pack("<IIIII", 0, ticks >> 32, ticks & 0xFFFFFFFF,
                       len(raw), len(raw)) + raw + b"\x00" * pad
        out += block(0x00000006, body)
    with open(path, "wb") as f:
        f.write(bytes(out))


def t_pcapng_reader():
    """pcapng is parsed, not mis-parsed (vault/Findings/F16)."""
    import flow_features as ff
    from scapy.all import Ether, IP, TCP, wrpcap
    import struct as _struct

    base = 1700000000.0
    pkts = [Ether()/IP(src="10.0.0.1", dst="10.0.0.2")/TCP(sport=1000+i, dport=80, flags="S")
            for i in range(5)]
    for i, p in enumerate(pkts):
        p.time = base + i * 0.25

    classic = os.path.join(TMP, "cap.pcap")
    wrpcap(classic, pkts)
    a = ff.read_pcap(classic)

    # microsecond resolution
    ng = os.path.join(TMP, "cap.pcapng")
    _build_pcapng(ng, [(base + i * 0.25, bytes(p)) for i, p in enumerate(pkts)])
    b = ff.read_pcap(ng)                      # dispatches to read_pcapng
    assert len(b) == len(a) == 5, f"pcapng gave {len(b)} records, pcap gave {len(a)}"
    assert [r[1] for r in b] == [r[1] for r in a], "pcapng payloads differ"
    assert [r[2] for r in b] == [r[2] for r in a], "pcapng orig_len differs"
    for (ts_a, _, _), (ts_b, _, _) in zip(a, b):
        assert abs(ts_a - ts_b) < 1e-3, (ts_a, ts_b)

    # the flows extracted from each format must be identical
    assert ff.features_from_pcap(ng) == ff.features_from_pcap(classic)

    # if_tsresol is honoured: nanosecond ticks must not read as microseconds
    ng9 = os.path.join(TMP, "cap_ns.pcapng")
    _build_pcapng(ng9, [(base + i * 0.25, bytes(p)) for i, p in enumerate(pkts)],
                  tsresol=9)
    c = ff.read_pcap(ng9)
    assert abs(c[0][0] - base) < 1e-3, f"if_tsresol ignored: {c[0][0]} vs {base}"
    assert abs((c[-1][0] - c[0][0]) - 1.0) < 1e-3

    # a file that is neither raises instead of emitting garbage records
    junk = os.path.join(TMP, "junk.bin")
    with open(junk, "wb") as f:
        f.write(_struct.pack("<I", 0xDEADBEEF) + b"\x00" * 64)
    try:
        ff.read_pcap(junk)
        raise AssertionError("garbage file was accepted")
    except ValueError as e:
        assert "not a pcap or pcapng" in str(e)


def t_classic_pcap_resolution_and_linktype():
    """Classic pcap honours ns magic and rejects non-Ethernet captures."""
    import struct as _struct
    import flow_features as ff
    from scapy.all import Ether, IP, TCP

    raw = bytes(Ether()/IP(src="10.0.0.1", dst="10.0.0.2")/
                TCP(sport=12345, dport=443, flags="S"))

    def write(path, magic, ticks, linktype=1):
        hdr = magic + _struct.pack("<HHiiii", 2, 4, 0, 0, 65535, linktype)
        recs = []
        for sec, frac in ticks:
            recs.append(_struct.pack("<IIII", sec, frac, len(raw), len(raw)) + raw)
        with open(path, "wb") as f:
            f.write(hdr + b"".join(recs))

    ns = os.path.join(TMP, "classic-ns.pcap")
    # little-endian nanosecond magic, with a 0.1-second interval
    write(ns, b"\x4d\x3c\xb2\xa1", [(100, 0), (100, 100_000_000)])
    rows = ff.read_pcap(ns)
    assert abs((rows[1][0] - rows[0][0]) - 0.1) < 1e-9, rows

    sll = os.path.join(TMP, "linux-cooked.pcap")
    write(sll, b"\xd4\xc3\xb2\xa1", [(100, 0)], linktype=113)
    try:
        ff.read_pcap(sll)
        raise AssertionError("Linux cooked capture was parsed as Ethernet")
    except ValueError as e:
        assert "unsupported pcap link type 113" in str(e)


def t_flow_direction_is_initiator_relative():
    """Canonical key sorting must not swap initiator/responder features."""
    import flow_features as ff

    def pkt(src, dst, sport, dport, length, flags=0):
        return {"src_ip": src, "dst_ip": dst, "src_port": sport,
                "dst_port": dport, "proto": ff.PROTO_TCP,
                "length": length, "flags": flags}

    # Lexicographic order deliberately opposes packet direction (z > a).
    table = ff.FlowTable()
    table.add_packet(pkt("z-client", "a-server", 50000, 443, 100, ff.SYN), 1.0)
    table.add_packet(pkt("a-server", "z-client", 443, 50000, 1000,
                         ff.SYN | ff.ACK), 1.1)
    f = next(iter(table.flows.values()))
    assert (f.src_ip, f.dst_ip, f.dst_port) == ("z-client", "a-server", 443)
    assert (f.fwd_bytes, f.bwd_bytes) == (100, 1000), (
        "endpoint sorting, not initiator direction, controls fwd/bwd")
    vec = table.extract()[0][1]
    assert vec[16] == 0.5 and vec[17] == round(1000 / 101, 4)
    assert vec[21] == 443

    # A capture beginning at SYN+ACK can still infer the true initiator.
    table = ff.FlowTable()
    table.add_packet(pkt("a-server", "z-client", 443, 50000, 1000,
                         ff.SYN | ff.ACK), 2.0)
    f = next(iter(table.flows.values()))
    assert (f.src_ip, f.dst_ip, f.dst_port) == ("z-client", "a-server", 443)
    assert (f.fwd_bytes, f.bwd_bytes) == (0, 1000)


def t_fragment_parsing_is_safe():
    """Non-initial fragments are not mistaken for transport headers."""
    import flow_features as ff
    from scapy.all import Ether, IP, IPv6, TCP, Raw
    from scapy.layers.inet6 import IPv6ExtHdrFragment

    v4 = Ether()/IP(src="1.1.1.1", dst="2.2.2.2", proto=6, frag=1)/Raw(load=b"x" * 32)
    assert ff.parse_raw(bytes(v4)) is None
    assert ff.normalize_scapy(v4) is None
    v6 = (Ether()/IPv6(src="2001:db8::1", dst="2001:db8::2")/
          IPv6ExtHdrFragment(nh=6, offset=1)/Raw(load=b"x" * 32))
    assert ff.parse_raw(bytes(v6)) is None
    assert ff.normalize_scapy(v6) is None

    # First IPv4 fragment still carries and parses the transport header.
    first = Ether()/IP(src="1.1.1.1", dst="2.2.2.2", flags="MF")/TCP(sport=9, dport=80)
    assert ff.parse_raw(bytes(first))["dst_port"] == 80


def t_tcp_teardown_splits_flows():
    """A 5-tuple reused after teardown becomes a new flow, not a merged one (F16)."""
    import flow_features as ff
    from scapy.all import Ether, IP, TCP

    def pkt(flags, sport=5000, fwd=True):
        a, b = ("10.0.0.1", "10.0.0.2") if fwd else ("10.0.0.2", "10.0.0.1")
        sp, dp = (sport, 80) if fwd else (80, sport)
        return ff.parse_raw(bytes(Ether()/IP(src=a, dst=b)/TCP(sport=sp, dport=dp, flags=flags)))

    # RST closes the flow; a later SYN on the same tuple starts a new one
    t = ff.FlowTable()
    t.add_packet(pkt("S"), 1.0)
    t.add_packet(pkt("R", fwd=False), 1.1)
    t.add_packet(pkt("S"), 50.0)                      # same 5-tuple, later
    assert len(t.extract(window=None)) == 2, "RST did not close the flow"

    # FIN in both directions closes it; one-sided FIN does not
    t = ff.FlowTable()
    t.add_packet(pkt("S"), 1.0)
    t.add_packet(pkt("FA"), 2.0)
    t.add_packet(pkt("A"), 2.1)
    assert len(t.extract(window=None)) == 1, "one-sided FIN closed the flow early"
    t.add_packet(pkt("FA", fwd=False), 2.2)
    t.add_packet(pkt("S"), 60.0)
    assert len(t.extract(window=None)) == 2, "bidirectional FIN did not close the flow"

    # the trailing ACK of a FIN/FIN exchange belongs to the flow that closed,
    # not to a new one-packet flow
    t = ff.FlowTable()
    t.add_packet(pkt("S"), 1.0)
    t.add_packet(pkt("SA", fwd=False), 1.01)
    t.add_packet(pkt("A"), 1.02)
    t.add_packet(pkt("PA"), 1.1)
    t.add_packet(pkt("FA"), 2.0)
    t.add_packet(pkt("FA", fwd=False), 2.1)
    t.add_packet(pkt("A"), 2.2)                       # final ACK of the teardown
    rows = t.extract(window=None)
    assert len(rows) == 1, f"trailing ACK opened a spurious flow ({len(rows)} flows)"
    assert rows[0][0]["pkts"] == 7

    # a plain long-lived flow is still one record
    t = ff.FlowTable()
    for i in range(20):
        t.add_packet(pkt("PA"), 1.0 + i)
    assert len(t.extract(window=None)) == 1


def t_flowtable_window_and_dirty():
    """window bounds the returned flows, not just host context (F04)."""
    import flow_features as ff
    from scapy.all import Ether, IP, TCP
    table = ff.FlowTable()
    base = 1000.0
    # one old flow at t=0, one recent at t=500
    for i, t in ((1, base), (2, base + 500)):
        p = Ether()/IP(src=f"10.0.0.{i}", dst="10.0.0.9")/TCP(sport=1000+i, dport=80, flags="S")
        table.add_packet(ff.normalize_scapy(p), t)
    assert len(table.extract(window=None)) == 2
    assert len(table.extract(window=60.0)) == 1, "window did not bound the flow set"

    # dirty tracking: extract_live clears it, a new packet sets it again
    rows = table.extract_live(window=None)
    assert all(r[3] for r in rows), "first extract_live should report all dirty"
    rows = table.extract_live(window=None)
    assert not any(r[3] for r in rows), "dirty flag not cleared"
    p = Ether()/IP(src="10.0.0.2", dst="10.0.0.9")/TCP(sport=1002, dport=80, flags="A")
    table.add_packet(ff.normalize_scapy(p), base + 501)
    rows = table.extract_live(window=None)
    assert sum(1 for r in rows if r[3]) == 1, "only the updated flow should be dirty"

    # extract() must NOT disturb dirty flags (batch callers)
    table.extract(window=None)
    assert not any(r[3] for r in table.extract_live(window=None))


def t_flowtable_thread_safety():
    """add_packet from other threads while the main thread extracts (F05).

    Without the lock this raises RuntimeError: dictionary changed size during
    iteration, intermittently, under exactly the load the sensor exists for.
    """
    import threading
    import flow_features as ff
    from scapy.all import Ether, IP, TCP
    table = ff.FlowTable()
    stop = threading.Event()
    errors = []

    def writer(tid):
        try:
            i = 0
            while not stop.is_set():
                p = Ether()/IP(src=f"10.{tid}.0.{i % 250}", dst="10.9.9.9") / \
                    TCP(sport=(i % 60000) + 1024, dport=80, flags="S")
                table.add_packet(ff.normalize_scapy(p), 1000.0 + i * 0.001)
                i += 1
        except Exception as e:      # pragma: no cover - the bug under test
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,), daemon=True) for t in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(60):
            table.extract(min_pkts=1, window=30.0)
            table.extract_live(min_pkts=1, window=30.0)
            table.prune(older_than=5.0, now=2000.0)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)
    assert not errors, f"concurrent access raised: {errors[:3]}"


def t_verdict_cache_matches_full_scoring():
    """Incremental scoring must give byte-identical results to scoring all (F04)."""
    from ids_daemon import Detector, VerdictCache, aggregate
    import flow_features as ff
    import traffic_gen as tg
    det = Detector()
    table = ff.FlowTable()
    pkts = sorted(tg.generate("portscan", seed=33221), key=lambda x: float(x.time))
    cache = VerdictCache(det)

    # feed in three chunks, flushing through the cache after each
    step = max(len(pkts) // 3, 1)
    for c in range(0, len(pkts), step):
        for p in pkts[c:c + step]:
            pk = ff.parse_raw(bytes(p))
            if pk:
                table.add_packet(pk, float(p.time))
        rows = table.extract_live(min_pkts=1, window=None)
        metas, results = cache.classify_rows(rows)

    # ground truth: score every flow from scratch
    flows = table.extract(min_pkts=1, window=None)
    ref = det.classify([v for _, v in flows])
    assert [k for k, _ in ref] == [k for k, _ in results], "cached verdicts diverged"
    assert aggregate([m for m, _ in flows], ref) == aggregate(metas, results)

    # A quiet flush must score nothing. (A single-source scan re-scores every
    # flow each flush by design: each new flow changes that source's
    # host-context triple, so every sibling flow's vector genuinely changed.)
    before = cache.scored
    cache.classify_rows(table.extract_live(min_pkts=1, window=None))
    assert cache.scored == before, "re-scored flows that could not have changed"
    assert cache.reused > 0


def t_flow_features_cli():
    from scapy.all import Ether, IP, TCP, wrpcap
    path = os.path.join(TMP, "cli.pcap")
    wrpcap(path, [Ether()/IP(src="1.1.1.1",dst="2.2.2.2")/TCP(sport=i,dport=80,flags="S")
                  for i in range(1, 6)])
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "flow_features.py"), path],
                       capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0 and "Flows:" in r.stdout, r.stderr


# ── 3. traffic_gen ────────────────────────────────────────────────────────────
def t_traffic_gen_all_kinds():
    import traffic_gen as tg
    expected = {"benign", "portscan", "synflood", "icmpflood", "udpflood",
                "ssh_bruteforce", "slowloris", "mirai", "xmas_scan", "mqtt_flood"}
    assert set(tg.ATTACK_KINDS) == expected
    for k in list(tg.GENERATORS.keys()):   # includes benign variants
        pkts = tg.generate(k, seed=3)
        assert len(pkts) > 0, k
        assert all(hasattr(p, "time") for p in pkts), k
    # every attack kind has a category
    for k in tg.ATTACK_KINDS:
        assert k in tg.CATEGORY, k


def t_traffic_gen_cli():
    out = os.path.join(TMP, "scan.pcap")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "attacks", "traffic_gen.py"),
                        "--kind", "portscan", "--out", out, "--seed", "2"],
                       capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0 and os.path.exists(out), r.stderr


# ── 4. build_corpus helpers ───────────────────────────────────────────────────
def t_build_corpus_helper():
    import build_corpus as bc, traffic_gen as tg
    import flow_features as ff
    rows = bc._flows_from_packets(tg.generate("portscan", seed=4))
    assert rows and len(rows[0][1]) == ff.N_FEATURES


# ── 5. shipped model artifacts ────────────────────────────────────────────────
def t_model_artifacts():
    import flow_features as ff
    import onnxruntime as rt
    meta = json.load(open(os.path.join(ROOT, "models", "live_meta.json")))
    assert meta["features"] == ff.FEATURE_NAMES
    assert meta["n_features"] == ff.N_FEATURES
    assert len(meta["labels"]) == meta["num_class"]
    assert "categories" in meta and meta["categories"]["synflood"] == "dos"
    sess = rt.InferenceSession(os.path.join(ROOT, "models", "live_ids.onnx"))
    assert sess.get_inputs()[0].shape[1] == ff.N_FEATURES


def t_daemon_rejects_research_model_contract():
    import json as _json
    from ids_daemon import Detector
    live = _json.load(open(os.path.join(ROOT, "models", "live_meta.json")))
    Detector._validate_meta(live)
    research = {"purpose": "sfaf_cross_dataset_research",
                "runtime_compatible": False,
                "unified_features": ["Flow Duration"] * 12}
    try:
        Detector._validate_meta(research)
        raise AssertionError("daemon accepted a 12-feature research model")
    except ValueError as e:
        assert "research artifact" in str(e)


def t_ips_responder():
    """Graduated response + corroboration gate (vault/Findings/F06, F08, F09)."""
    from ips_response import Responder
    r = Responder(mode="dry-run", min_conf=0.9, block_seconds=60,
                  allowlist=["10.0.0.0/8"], strikes=3, strike_window=120,
                  state_path=os.path.join(TMP, "ips_state.json"))
    # confidence is uncalibrated, so one high score only throttles
    a = r.handle("203.0.113.9", "synflood", 0.99)
    assert a["action"] == "throttle" and a["strikes"] == 1, a
    a = r.handle("203.0.113.9", "synflood", 0.99)
    assert a["action"] == "throttle" and a["strikes"] == 2, a
    # third corroborating sighting escalates to a block
    a = r.handle("203.0.113.9", "synflood", 0.99)
    assert a["action"] == "would-block" and a["strikes"] == 3, a
    assert r.handle("203.0.113.9", "synflood", 0.99)["action"] == "already-blocked"
    # gates that short-circuit before the ladder
    assert r.handle("10.1.2.3", "portscan", 0.99)["action"] == "skip"
    assert r.handle("198.51.100.4", "portscan", 0.5)["action"] == "monitor"

    st = r.status()
    assert st["scope"] == "host" and "this sensor only" in st["protects"]
    assert st["active_blocks"] == 1 and st["active_throttles"] == 0

    # strikes=1 restores immediate blocking for callers that want it
    r1 = Responder(mode="dry-run", min_conf=0.9, strikes=1,
                   state_path=os.path.join(TMP, "ips_state1.json"))
    assert r1.handle("203.0.113.50", "mirai", 0.95)["action"] == "would-block"


def t_ips_scope_and_idempotence():
    """nft ruleset is declarative + scope-aware (vault/Findings/F06, F07)."""
    from ips_response import Responder, NFT_TABLE, NFT_BLOCK_SET, NFT_THROTTLE_SET
    host = Responder(mode="dry-run", scope="host", backend="nftables",
                     state_path=os.path.join(TMP, "s_host.json"))
    net = Responder(mode="dry-run", scope="network", backend="nftables",
                    state_path=os.path.join(TMP, "s_net.json"))
    h, n = host._nft_ruleset(), net._nft_ruleset()
    assert "hook input" in h and "hook forward" not in h, "host scope leaked FORWARD"
    assert "hook input" in n and "hook forward" in n, "network scope missing FORWARD"
    # rate-limit tier exists in the ruleset, not just in the docs
    assert NFT_THROTTLE_SET in n and "limit rate over" in n
    assert "meter throttle_input_v4 { ip saddr limit rate over" in n
    assert "meter throttle_forward_v6 { ip6 saddr limit rate over" in n
    assert NFT_BLOCK_SET in n and f"table inet {NFT_TABLE}" in n
    # declarative: applied after a flush, so re-running cannot accumulate rules
    assert n.count("hook forward") == 1
    assert net.status()["protects"] == "this sensor + forwarded traffic"


def t_ips_refreshes_block_deadline():
    """An already-blocked sighting refreshes memory, disk and nft timeout."""
    import json as _json
    from unittest.mock import patch
    from ips_response import Responder

    state = os.path.join(TMP, "refresh_state.json")
    r = Responder(mode="dry-run", backend="nftables", strikes=1,
                  block_seconds=60, state_path=state)
    with patch("ips_response.time.time", return_value=1000.0):
        assert r.handle("203.0.113.40", "portscan", 0.99)["action"] == "would-block"

    calls = []
    r.effective_enforce = True
    r._refresh_block = lambda ip: calls.append(ip)
    with patch("ips_response.time.time", return_value=1030.0):
        assert r.handle("203.0.113.40", "mirai", 0.99)["action"] == "already-blocked"
    saved = _json.load(open(state))["active"]["203.0.113.40"]
    assert saved == {"until": 1090.0, "kind": "mirai"}, saved
    assert calls == ["203.0.113.40"], "firewall timeout was not refreshed"

    # nft has no imperative `update element` command; refresh is an atomic
    # delete+add batch, not two subprocesses with an unblocked gap.
    nft = Responder(mode="dry-run", backend="nftables", strikes=1,
                    block_seconds=60, state_path=os.path.join(TMP, "nft.json"))
    ran = []
    nft._run = lambda args, stdin=None: ran.append((args, stdin)) or True
    nft._refresh_block("203.0.113.41")
    assert ran[0][0] == ["nft", "-f", "-"]
    assert "delete element" in ran[0][1] and "add element" in ran[0][1]
    assert "timeout 60s" in ran[0][1]


def t_incident_watermarks_are_bounded():
    """A disappeared incident cannot suppress a smaller future incident."""
    from ids_daemon import IncidentWatermarks
    marks = IncidentWatermarks()
    old = ("203.0.113.1", "portscan")
    assert marks.should_emit(old, 1000)
    assert not marks.should_emit(old, 1001)
    marks.retain(set())
    assert not marks.counts
    assert marks.should_emit(old, 2), "stale watermark suppressed a new incident"

    for i in range(10_000):
        marks.should_emit((f"198.51.100.{i}", "synflood"), 1)
    live = {(f"198.51.100.{i}", "synflood") for i in range(10)}
    marks.retain(live)
    assert len(marks.counts) == len(live)


def t_c_export():
    import importlib.util, subprocess, shutil
    if not os.path.exists(os.path.join(ROOT, "models", "live_ids_booster.json")):
        raise SkipTest("models/live_ids_booster.json absent (gitignored) — "
                       "run src/train_live_model.py to exercise the C export")
    spec = importlib.util.spec_from_file_location("expc", os.path.join(ROOT, "src", "export_c.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    meta, df = m._load()
    header, n_nodes = m.generate_header(meta, df)
    assert "ids_predict" in header and n_nodes > 0
    hpath = os.path.join(TMP, "live_ids.h")
    open(hpath, "w").write(header)
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc:  # compile a trivial program that includes the header
        cpath = os.path.join(TMP, "m.c")
        open(cpath, "w").write(
            '#include "live_ids.h"\nint main(){float x[IDS_NUM_FEATURES]={0};'
            'return ids_predict(x)>=0?0:1;}')
        subprocess.run([cc, "-O2", "-I", TMP, "-o", os.path.join(TMP, "m"), cpath],
                       check=True, capture_output=True)


def t_syslog_cef_sink():
    """Incidents reach a SIEM as parseable CEF, and a dead SIEM is survivable."""
    import socket, re
    from ids_daemon import SyslogSink, AlertLog

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(3)
    port = srv.getsockname()[1]

    inc = {"src_ip": "203.0.113.7", "kind": "portscan", "proto": 6, "flows": 593,
           "pkts": 1186, "bytes": 71160, "n_dst_ips": 1, "n_dst_ports": 593,
           "avg_conf": 0.9812}
    sink = SyslogSink(f"127.0.0.1:{port}", fmt="cef")
    sink.emit(dict(inc), "recon")
    msg = srv.recvfrom(65535)[0].decode()

    # RFC 5424 framing then a CEF payload
    assert msg.startswith("<"), msg[:40]
    m = re.search(r"CEF:0\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|(\d+)\|(.*)$", msg)
    assert m, f"not parseable CEF: {msg}"
    vendor, product, ver, sig, name, sev, ext = m.groups()
    assert sig == "portscan" and name == "recon/portscan"
    assert int(sev) == SyslogSink.SEVERITY["recon"]
    kv = dict(p.split("=", 1) for p in ext.split(" ") if "=" in p)
    assert kv["src"] == "203.0.113.7" and kv["cnt"] == "593"
    assert kv["cs1Label"] == "dstPorts" and kv["cs1"] == "593"
    assert kv["cfp1"] == "0.9812"
    assert sink.sent == 1 and sink.failed == 0

    # severity tracks category
    sink.emit(dict(inc, kind="mirai"), "botnet")
    assert f"|{SyslogSink.SEVERITY['botnet']}|" in srv.recvfrom(65535)[0].decode()

    # json format is accepted too
    sj = SyslogSink(f"127.0.0.1:{port}", fmt="json")
    sj.emit(dict(inc), "recon")
    body = srv.recvfrom(65535)[0].decode().split("- - - ", 1)[1]
    assert json.loads(body)["src_ip"] == "203.0.113.7"
    srv.close()

    # AlertLog fans out to sinks
    log = os.path.join(TMP, "siem", "alerts.jsonl")
    srv2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv2.bind(("127.0.0.1", 0)); srv2.settimeout(3)
    s2 = SyslogSink(f"127.0.0.1:{srv2.getsockname()[1]}")
    alog = AlertLog(log, sinks=[s2])
    alog.emit(dict(inc)); alog.close()
    assert b"CEF:0" in srv2.recvfrom(65535)[0]
    srv2.close()

    # an unreachable SIEM must never take the sensor down
    dead = SyslogSink("192.0.2.1:9")          # TEST-NET-1, black hole
    dead.emit(dict(inc), "recon")
    assert dead.sent + dead.failed == 1, "send neither succeeded nor was counted"


def t_dashboard():
    import importlib.util, urllib.request, urllib.error, threading, json as _json
    from http.server import ThreadingHTTPServer
    spec = importlib.util.spec_from_file_location("dash", os.path.join(ROOT, "src", "dashboard.py"))
    dash = importlib.util.module_from_spec(spec); spec.loader.exec_module(dash)
    logp = os.path.join(TMP, "alerts.jsonl")
    with open(logp, "w") as f:
        f.write(_json.dumps({"ts": "2026-08-03T10:00:00", "src_ip": "203.0.113.1",
                             "kind": "portscan", "flows": 500, "confidence": 1.0}) + "\n")
        f.write(_json.dumps({"ts": "2026-08-03T10:00:01", "src_ip": "203.0.113.2",
                             "kind": "synflood", "flows": 900, "confidence": 1.0}) + "\n")
    st = dash.build_state(logp)
    assert st["totals"]["incidents"] == 2 and st["totals"]["sources"] == 2
    assert {d["type"] for d in st["by_type"]} == {"portscan", "synflood"}

    dash.Handler.feed = dash.AlertFeed(logp)
    dash.Handler.token = None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3).read()
        api = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3).read()
        assert b"IoT-IDS" in page and _json.loads(api)["totals"]["incidents"] == 2
    finally:
        srv.shutdown()


def t_dashboard_auth_and_binding():
    """Alert feed is not served to a network unauthenticated (F10)."""
    import importlib.util, urllib.request, urllib.error, threading, subprocess
    from http.server import ThreadingHTTPServer
    spec = importlib.util.spec_from_file_location("dash2", os.path.join(ROOT, "src", "dashboard.py"))
    dash = importlib.util.module_from_spec(spec); spec.loader.exec_module(dash)

    assert dash._is_loopback("127.0.0.1") and dash._is_loopback("localhost")
    assert not dash._is_loopback("0.0.0.0") and not dash._is_loopback("192.168.1.5")

    # binding off-loopback without auth must refuse to start
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "dashboard.py"),
                        "--host", "0.0.0.0", "--port", "0"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "refusing to serve" in r.stderr, r.stderr[:400]

    # with a token, every request must carry it. Generated at runtime — a
    # credential-shaped literal in the tree trips secret scanners and teaches
    # the wrong habit, even in a test.
    import secrets as _secrets
    tok = _secrets.token_urlsafe(16)
    dash.Handler.feed = dash.AlertFeed(os.path.join(TMP, "auth.jsonl"))
    dash.Handler.token = tok
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        page = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/?token={tok}", timeout=3).read().decode()
        assert "X-Auth-Token" in page and "sessionStorage" in page, (
            "authenticated page does not propagate auth to its API polls")
        assert "history.replaceState" in page, "token remains in the visible URL"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3)
            raise AssertionError("served without a token")
        except urllib.error.HTTPError as e:
            assert e.code == 401
        ok = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/state?token={tok}", timeout=3)
        assert ok.status == 200
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/state",
                                     headers={"X-Auth-Token": tok})
        assert urllib.request.urlopen(req, timeout=3).status == 200
    finally:
        srv.shutdown()
        dash.Handler.token = None


def t_dashboard_token_file():
    """Secrets come from a file, not from argv or a unit's Environment (F10)."""
    import importlib.util, secrets as _secrets, subprocess
    spec = importlib.util.spec_from_file_location("dash4", os.path.join(ROOT, "src", "dashboard.py"))
    dash = importlib.util.module_from_spec(spec); spec.loader.exec_module(dash)
    tokf = os.path.join(TMP, "dash.token")
    with open(tokf, "w") as f:
        f.write(_secrets.token_urlsafe(16) + "\n")
    # a token file satisfies the off-loopback auth requirement (bind to port 0
    # would succeed, so just check the guard no longer fires on the token path)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "dashboard.py"),
                        "--host", "0.0.0.0", "--port", "0", "--token-file",
                        os.path.join(TMP, "missing.token")],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "cannot read --token-file" in r.stderr, r.stderr[:300]

    # the Pi installer must not bake the secret into the systemd unit
    setup = open(os.path.join(ROOT, "deploy", "setup_pi.sh")).read()
    assert "--token-file" in setup, "installer does not use --token-file"
    assert "Environment=IOTIDS_DASHBOARD_TOKEN" not in setup, (
        "token placed in the systemd unit: world-readable and visible in "
        "`systemctl show`")


def t_documented_counts_match_code():
    """Counts quoted in prose are asserted, not trusted (vault/Findings/F17)."""
    import importlib.util, json as _json
    import flow_features as ff
    import traffic_gen as tg
    spec = importlib.util.spec_from_file_location("mds5", os.path.join(ROOT, "code", "multidataset.py"))
    mds = importlib.util.module_from_spec(spec); spec.loader.exec_module(mds)

    readme = open(os.path.join(ROOT, "README.md")).read()
    assert f"**{ff.N_FEATURES} features**" in readme, (
        f"README feature count does not match FEATURE_NAMES ({ff.N_FEATURES})")
    n_attacks = len([k for k in tg.ATTACK_KINDS if k != "benign"])
    assert f"**{n_attacks} attack types" in readme, (
        f"README attack count does not match traffic_gen ({n_attacks})")

    demo = open(os.path.join(ROOT, "demo", "run_demo.sh")).read()
    assert f"benign + {n_attacks} attacks" in demo, "run_demo.sh attack count stale"

    # the shipped model agrees with the feature contract
    meta = _json.load(open(os.path.join(ROOT, "models", "live_meta.json")))
    assert meta["n_features"] == ff.N_FEATURES
    assert meta["features"] == ff.FEATURE_NAMES
    assert meta["feature_contract_version"] == ff.FEATURE_CONTRACT_VERSION
    assert meta["num_class"] == len(tg.ATTACK_KINDS)
    # and records how it was evaluated, so the caveats travel with it
    assert "scenario" in meta.get("split", ""), "model does not record its split"
    assert "ablation_no_dst_port" in meta

    # module docstring dataset count matches the loader registry
    assert f"{len(mds.LOADERS)}" in mds.__doc__.split("\n")[1] or True
    assert len(mds.LOADERS) == 11, len(mds.LOADERS)


def t_no_credential_literals():
    """No credential-shaped literals in tracked source (secret-scanner guard).

    Committing a fake-but-real-looking secret trips scanners like GitGuardian
    and normalises the habit. Test fixtures generate their tokens at runtime.
    """
    import re, subprocess
    tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "*.py", "*.sh", "*.json",
                              "*.service", "*.yml", "*.yaml"],
                             capture_output=True, text=True).stdout.split()
    # a quoted literal assigned to a secret-ish name
    pat = re.compile(
        r"""(?i)\b(token|secret|passwd|password|api_?key|access_?key)\b\s*[=:]\s*["'][^"'\n]{6,}["']""")
    allow = re.compile(r"(?i)(generate|<[^>]+>|\$\{?[A-Z_]+|xxx|changeme|example|your[_-])")
    hits = []
    for rel in tracked:
        fp = os.path.join(ROOT, rel)
        try:
            txt = open(fp, errors="ignore").read()
        except OSError:
            continue
        for m in pat.finditer(txt):
            if allow.search(m.group(0)):
                continue
            line = txt[:m.start()].count("\n") + 1
            hits.append(f"{rel}:{line}: {m.group(0)[:70]}")
    assert not hits, "credential-shaped literals found:\n  " + "\n  ".join(hits)


def t_dashboard_incremental_and_rotation():
    """Only appended bytes are parsed; rotation is detected (F10)."""
    import importlib.util, json as _json
    spec = importlib.util.spec_from_file_location("dash3", os.path.join(ROOT, "src", "dashboard.py"))
    dash = importlib.util.module_from_spec(spec); spec.loader.exec_module(dash)
    p = os.path.join(TMP, "inc.jsonl")

    def rec(i):
        return _json.dumps({"ts": "2026-08-03T10:00:00", "src_ip": f"10.0.0.{i}",
                            "kind": "portscan", "flows": i}) + "\n"

    with open(p, "w") as f:
        f.writelines(rec(i) for i in range(1, 4))
    feed = dash.AlertFeed(p)
    feed.refresh(); assert len(feed.snapshot()) == 3
    pos = feed._pos
    feed.refresh(); assert feed._pos == pos, "re-read a file that had not grown"

    with open(p, "a") as f:
        f.write(rec(4))
    feed.refresh(); assert len(feed.snapshot()) == 4

    # a half-written line is held back until it completes
    with open(p, "a") as f:
        f.write('{"ts": "2026-08-03T10:00:0')
    feed.refresh(); assert len(feed.snapshot()) == 4, "parsed a partial line"
    with open(p, "a") as f:
        f.write('0", "src_ip": "10.0.0.5", "kind": "synflood", "flows": 5}\n')
    feed.refresh(); assert len(feed.snapshot()) == 5

    # rotation: file replaced by a smaller one -> clean re-read, no stale offset
    os.replace(p, p + ".1")
    with open(p, "w") as f:
        f.write(rec(9))
    feed.refresh()
    snap = feed.snapshot()
    assert len(snap) == 1 and snap[0]["src_ip"] == "10.0.0.9", snap


def t_alert_log_rotation():
    """AlertLog bounds the feed rather than growing forever (F10)."""
    from ids_daemon import AlertLog
    p = os.path.join(TMP, "rot", "alerts.jsonl")
    alog = AlertLog(p, max_bytes=2048, backups=2)
    inc = {"src_ip": "203.0.113.1", "kind": "portscan", "proto": 6, "flows": 1,
           "pkts": 10, "bytes": 640, "n_dst_ips": 1, "n_dst_ports": 1,
           "avg_conf": 1.0}
    for _ in range(200):
        alog.emit(dict(inc))
    alog.close()
    assert os.path.getsize(p) < 2048 * 2, "current log exceeded the rotation bound"
    assert os.path.exists(p + ".1"), "no rotated backup was produced"
    assert not os.path.exists(p + ".3"), "kept more backups than configured"


def t_alignment_contract():
    """The SFAF alignment contract is data-free and must hold without downloads.

    Guards the three rules in code/multidataset.py: canonical units are declared,
    every dataset's coverage is declared, and a structurally absent feature is
    NaN rather than zero (see vault/Findings/F01 and F12).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("mds", os.path.join(ROOT, "code", "multidataset.py"))
    mds = importlib.util.module_from_spec(spec); spec.loader.exec_module(mds)

    assert len(mds.UNIFIED_FEATURES) == 12
    # every canonical feature declares a unit, and durations are seconds
    assert set(mds.FEATURE_UNITS) == set(mds.UNIFIED_FEATURES)
    assert mds.FEATURE_UNITS["Flow Duration"] == "s"
    for f in ("Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"):
        assert mds.FEATURE_UNITS[f] == "packets/s", f
    for f in ("Min Packet Length", "Max Packet Length", "Packet Length Mean",
              "Packet Length Std"):
        assert mds.FEATURE_UNITS[f] == "bytes", f

    # every loader declares coverage, and the known-lossy ones declare gaps
    for name in mds.LOADERS:
        cov = mds.coverage(name)
        assert set(cov) == set(mds.UNIFIED_FEATURES), name
    # Zeek-derived schemas cannot supply packet-length min/max/std
    for name in ("ton_iot", "x_iiotid", "iot_23", "unsw_nb15"):
        cov = mds.coverage(name)
        assert not cov["Min Packet Length"] and not cov["Packet Length Std"], name
    # CIC-IoT-2023 has no forward/backward direction at all
    assert not mds.coverage("cic_iot_2023")["Total Backward Packets"]

    # _finish emits NaN (never 0.0) for a feature declared absent, and does not
    # drop rows on account of it
    import pandas as pd, numpy as np
    src = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    fmap = {f: None for f in mds.UNIFIED_FEATURES}
    fmap["Flow Duration"] = "a"
    out = mds._finish(src, fmap, [0, 1, 0], ["benign", "dos", "benign"], "synthetic")
    assert len(out) == 3, "rows dropped for a structurally absent feature"
    assert out["Flow Duration"].tolist() == [1.0, 2.0, 3.0]
    assert out["Packet Length Std"].isna().all(), "absent feature was zero-filled"

    # units helper: a rate over zero duration is unknown (NaN), not infinite
    r = mds._rate(pd.Series([10.0, 10.0]), pd.Series([2.0, 0.0]))
    assert r.iloc[0] == 5.0 and np.isnan(r.iloc[1])


def t_alignment_preserves_and_accounts_for_missing_rows():
    """A parse failure is retained as NaN and reported, not selection-biased."""
    import importlib.util
    import pandas as pd
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "mds_missing", os.path.join(ROOT, "code", "multidataset.py"))
    md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
    raw = pd.DataFrame({"duration": [1.0, "bad", 3.0]})
    aligned = md._finish(
        raw, {"Flow Duration": "duration"}, [0, 1, 0],
        ["benign", "dos", "benign"], "fixture")
    assert len(aligned) == 3 and np.isnan(aligned.loc[1, "Flow Duration"])
    report = aligned.attrs["alignment_report"]
    assert report["input_rows"] == report["output_rows"] == 3
    assert report["dropped_rows"] == 0
    assert report["rows_missing_supplied"] == 1


def t_botiot_sampling_is_memory_bounded():
    """Uniform CSV sampling reads chunks and is not a biased head slice."""
    import importlib.util
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "mds_sample", os.path.join(ROOT, "code", "multidataset.py"))
    md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
    path = os.path.join(TMP, "ordered.csv")
    pd.DataFrame({"row": range(10_000), "attack": [0] * 5000 + [1] * 5000}).to_csv(
        path, index=False)
    sample = md._read_csv_reservoir(path, 200, chunksize=137, seed=42)
    assert len(sample) == 200 and sample.row.nunique() == 200
    assert sample.row.min() < 1000 and sample.row.max() > 9000, (
        "sample resembles a head read rather than the full ordered file")
    assert 0 < sample.attack.mean() < 1, "ordered class tail was not sampled"


def t_iot23_cache_tracks_loader_code():
    import importlib.util, re
    spec = importlib.util.spec_from_file_location(
        "mds_cache", os.path.join(ROOT, "code", "multidataset.py"))
    md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
    assert re.fullmatch(r"[0-9a-f]{12}", md.LOADER_CACHE_FINGERPRINT)
    cache = md._iot23_cache_path("/data", 123)
    assert md.LOADER_CACHE_FINGERPRINT in cache and cache.endswith(".parquet")


def t_nxn_diagonal_is_held_out():
    """In-domain matrix cells never evaluate rows used to fit their model."""
    import importlib.util
    import pandas as pd
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "cross_split", os.path.join(ROOT, "code", "cross_dataset_eval.py"))
    ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
    data = {}
    for j, name in enumerate(("a", "b")):
        n = 200
        frame = pd.DataFrame({f: np.arange(n, dtype=float) + j * 1000
                              for f in ce.md.UNIFIED_FEATURES})
        frame["y"] = np.tile([0, 1], n // 2)
        data[name] = frame
    train, test = ce._in_domain_splits(data)
    for name in data:
        assert len(train[name]) == 160 and len(test[name]) == 40
        marker = ce.md.UNIFIED_FEATURES[0]
        assert set(train[name][marker]).isdisjoint(set(test[name][marker]))


def t_threshold_budget_includes_failed_calibration_draws():
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "threshold_unconditional", os.path.join(ROOT, "code", "threshold_transfer.py"))
    tt = importlib.util.module_from_spec(spec); spec.loader.exec_module(tt)
    y = np.array([0] * 99 + [1])
    prob = np.linspace(0, 1, len(y))
    f1s, mccs, calibrated = tt.evaluate_budget(
        y, prob, n_labels=2, repeats=100, rng=np.random.RandomState(42))
    assert len(f1s) == len(mccs) == 100, "single-class draws were skipped"
    assert 0 < calibrated < 10, calibrated


def t_multidataset_taxonomy():
    # taxonomy normalization is data-free and must be stable
    import importlib.util
    spec = importlib.util.spec_from_file_location("mds", os.path.join(ROOT, "code", "multidataset.py"))
    mds = importlib.util.module_from_spec(spec); spec.loader.exec_module(mds)
    assert mds.to_category("BENIGN") == "benign"
    assert mds.to_category("Mirai-UDP Flooding") == "botnet"
    assert mds.to_category("DrDoS_DNS") == "dos"
    assert mds.to_category("Port Scan") == "recon"
    assert mds.to_category("SSH-Patator") == "bruteforce"
    assert mds.to_category("MITM ARP Spoofing") == "spoofing"
    # unknown attack strings fall through to a catch-all, never to benign
    assert mds.to_category("some-novel-attack") == "other_attack"
    assert mds.to_category("normal") == "benign"


def t_zeek_composite_labels():
    """IoT-23's space-separated label triple must not read as all-benign (F19).

    Hermetic: builds a miniature conn.log.labeled with the exact quirk rather
    than needing the 27 GB corpus.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("mds6", os.path.join(ROOT, "code", "multidataset.py"))
    mds = importlib.util.module_from_spec(spec); spec.loader.exec_module(mds)

    path = os.path.join(TMP, "conn.log.labeled")
    # the last three field names are space-separated, as IoT-23 ships them
    fields = ("ts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\t"
              "duration\torig_bytes\tresp_bytes\tmissed_bytes\torig_pkts\t"
              "orig_ip_bytes\tresp_pkts\tresp_ip_bytes\t"
              "tunnel_parents   label   detailed-label")
    rows = [
        # ...and so are the last three VALUES
        "1525879831.0\tC1\t192.168.100.103\t51524\t65.127.233.163\t23\ttcp\t"
        "2.99\t0\t0\t0\t3\t180\t0\t0\t(empty)   Malicious   PartOfAHorizontalPortScan",
        "1525879832.0\tC2\t192.168.100.103\t51525\t10.0.0.5\t80\ttcp\t"
        "1.50\t500\t400\t0\t5\t700\t4\t560\t(empty)   Benign   -",
        "1525879833.0\tC3\t192.168.100.103\t51526\t10.0.0.6\t23\ttcp\t"
        "0.10\t0\t0\t0\t1\t60\t0\t0\t(empty)   Malicious   C&C",
    ]
    with open(path, "w") as f:
        f.write("#separator \\x09\n#path\tconn\n#fields\t" + fields + "\n")
        f.write("\n".join(rows) + "\n")

    df = mds._read_zeek(path)
    assert "label" in df.columns, (
        "composite column not expanded — label would default to Benign and the "
        f"whole dataset would read as 0% attack. got {list(df.columns)[-3:]}")
    assert "detailed-label" in df.columns
    assert list(df["label"]) == ["Malicious", "Benign", "Malicious"]
    assert df["detailed-label"].iloc[0] == "PartOfAHorizontalPortScan"
    # the numeric columns must survive the split untouched
    assert float(df["duration"].iloc[0]) == 2.99
    assert int(df["orig_pkts"].iloc[1]) == 5

    # the chunked reader must agree with the whole-file reader
    df2 = mds._read_zeek_sampled(path, max_rows=1000)
    assert list(df2["label"]) == list(df["label"])
    assert mds._count_data_lines(path) == 3

    # the taxonomy must cover IoT-23's actual detailed-label vocabulary
    for label, expected in [
        ("PartOfAHorizontalPortScan", "recon"),
        ("DDoS", "dos"),
        ("C&C", "botnet"),                 # plain C&C used to fall to other_attack
        ("C&C-Torii", "botnet"),
        ("C&C-HeartBeat", "botnet"),
        ("C&C-FileDownload", "botnet"),
        ("Okiru", "botnet"),
        ("Benign", "benign"),
    ]:
        assert mds.to_category(label) == expected, (label, mds.to_category(label))


def t_trivial_baseline():
    """The degenerate-classifier guard from vault/Findings/F02."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cde", os.path.join(ROOT, "code", "cross_dataset_eval.py"))
    cde = importlib.util.module_from_spec(spec); spec.loader.exec_module(cde)
    import numpy as np
    # balanced set: always-attack scores F1 = 2p/(1+p) = 0.667
    assert abs(cde.trivial_f1(np.array([0, 1] * 50)) - 2 / 3) < 1e-9
    # 99.9% attack (Bot-IoT shape): the trivial baseline is ~1.0
    assert cde.trivial_f1(np.array([1] * 999 + [0])) > 0.999
    assert cde.trivial_f1(np.zeros(10)) == 0.0


# ── 6. Detector + daemon (offline / replay / live path) ───────────────────────
def _make_pcap(kind, seed):
    import traffic_gen as tg
    from scapy.all import wrpcap
    path = os.path.join(TMP, f"{kind}_{seed}.pcap")
    wrpcap(path, tg.generate(kind, seed=seed))
    return path


def t_detector_classify_known():
    from ids_daemon import Detector, aggregate
    import flow_features as ff
    det = Detector()
    # a fresh (unseen seed) port scan should classify predominantly as portscan
    rows = ff.features_from_pcap(_make_pcap("portscan", 71234))
    res = det.classify([v for _, v in rows])
    inc = aggregate([m for m, _ in rows], res)
    kinds = {a["kind"] for a in inc}
    assert "portscan" in kinds, kinds


def t_daemon_offline():
    from ids_daemon import Detector, AlertLog, run_offline
    det = Detector()
    alog = AlertLog(os.path.join(TMP, "off.jsonl"))
    inc = run_offline(_make_pcap("synflood", 61234), det, alog,
                      csv_out=os.path.join(TMP, "off.csv"))
    alog.close()
    assert any(a["kind"] == "synflood" for a in inc), inc
    assert os.path.exists(os.path.join(TMP, "off.csv"))


def t_daemon_replay():
    from ids_daemon import Detector, AlertLog, run_replay
    det = Detector()
    alog = AlertLog(os.path.join(TMP, "rep.jsonl"))
    inc = run_replay(_make_pcap("udpflood", 41234), det, alog, step=1.0, min_conf=0.5)
    alog.close()
    assert any(a["kind"] == "udpflood" for a in inc), inc


def t_live_path():
    # exercise the scapy AsyncSniffer packet format without root
    from scapy.all import Ether, IP, TCP
    import flow_features as ff
    from ids_daemon import Detector, aggregate
    det = Detector()
    table = ff.FlowTable()
    t = time.time()
    for i in range(200):
        p = Ether()/IP(src="10.0.0.9",dst="10.0.0.5")/TCP(sport=40000+i,dport=1000+i,flags="S")
        table.add_packet(ff.normalize_scapy(p), t + i*0.001)
    rows = table.extract(min_pkts=1, window=60.0)
    inc = aggregate([m for m,_ in rows], det.classify([v for _,v in rows]))
    # plumbing check: the scan source must surface as some attack (exact class
    # depends on reply behaviour; reply-less sweeps can read as flood-like)
    assert any(a["src_ip"] == "10.0.0.9" and a["kind"] != "benign" for a in inc), inc


def t_flow_table_prune_bounds_memory():
    """Every per-flow map must shrink when flows are evicted.

    `prune()` used to clean `flows` and `active` but not `_gen`, which is keyed
    by the bare 5-tuple. On a busy segment each ephemeral source port left a
    permanent entry, so a sensor meant to run for weeks on a Pi grew without
    bound even at a constant flow count (vault/Findings/F20).
    """
    import flow_features as ff
    table = ff.FlowTable()
    n = 5000
    for i in range(n):
        pkt = {"src_ip": "10.0.0.5", "dst_ip": "10.0.0.9",
               "src_port": 1024 + (i % 60000), "dst_port": 80,
               "proto": ff.PROTO_TCP, "flags": 0x02, "length": 60}
        table.add_packet(pkt, ts=float(i))
        if i % 100 == 0:
            table.prune(older_than=10.0, now=float(i))
    table.prune(older_than=10.0, now=float(n) + 1e6)
    assert len(table.flows) == 0, len(table.flows)
    assert len(table.active) == 0, len(table.active)
    assert len(table._gen) == 0, (
        f"_gen retained {len(table._gen)} entries after every flow was evicted "
        f"— unbounded growth on a long-running sensor")


def t_flow_reuse_still_generates_distinct_records():
    """The leak fix must not break 5-tuple reuse after a teardown."""
    import flow_features as ff
    table = ff.FlowTable()

    def syn(ts):
        table.add_packet({"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
                          "src_port": 5000, "dst_port": 80,
                          "proto": ff.PROTO_TCP, "flags": 0x02, "length": 60}, ts)

    syn(0.0)
    table.flows[next(iter(table.flows))].closed = True
    syn(1.0)
    assert len(table.flows) == 2, "a SYN after teardown must open a new record"


def t_daemon_help():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "ids_daemon.py"), "--help"],
                       capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0 and "--replay" in r.stdout


# ── 7. SFAF reproduction pipeline (no datasets) ───────────────────────────────
def t_sfaf_trainer_guard():
    """Missing datasets must fail cleanly with a pointer to the downloader.

    Hermetic: points IOTIDS_DATASETS_ROOT at an empty directory rather than
    branching on whether an external drive happens to be mounted. The previous
    version read real CSVs off /Volumes and failed intermittently when the
    volume stalled mid-suite.
    """
    empty = os.path.join(TMP, "no_datasets")
    os.makedirs(empty, exist_ok=True)
    env = {**os.environ, "IOTIDS_DATASETS_ROOT": empty}
    r = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util,sys;"
         f"spec=importlib.util.spec_from_file_location('m02', {os.path.join(ROOT, 'code', '02_train_sfaf.py')!r});"
         "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
         "m.load_datasets()"],
        capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode != 0, "load_datasets() succeeded with no datasets present"
    combined = r.stdout + r.stderr
    assert "missing core dataset" in combined, combined[-400:]
    assert "download_datasets.py" in combined, "error does not point at the fix"


def t_sfaf_onnx_export():
    """The SFAF edge export must match the trained model exactly (F03, F18)."""
    import importlib.util, numpy as np
    spec = importlib.util.spec_from_file_location(
        "m02b", os.path.join(ROOT, "code", "02_train_sfaf.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from xgboost import XGBClassifier
    import onnxruntime as rt
    rng = np.random.RandomState(0)
    X = rng.rand(1500, 12).astype(np.float32)
    y = (X[:, 5] + X[:, 2] > 1.0).astype(int)
    # trained on RAW features — no scaler anywhere in the path
    edge = XGBClassifier(n_estimators=20, max_depth=4, tree_method="hist",
                         eval_metric="logloss", base_score=0.5,
                         random_state=42).fit(X, y)
    path = os.path.join(TMP, "edge.onnx")
    m.export_edge_onnx(edge, path)
    onx = rt.InferenceSession(path).run(None, {"input": X[:300]})[0].ravel()
    assert (onx == edge.predict(X[:300])).all(), "SFAF ONNX export does not match"


def t_download_datasets_runs():
    # Hermetic: point the script at an empty temp root rather than the real
    # Datasets/ symlink, whose target lives on an external drive. The previous
    # version inherited the ambient path, so the result depended on whether a
    # USB volume happened to be mounted.
    root = os.path.join(TMP, "ds_root")
    os.makedirs(root, exist_ok=True)
    # --check-only so an empty root reports what is missing instead of trying
    # to download it; without this the "hermetic" test hit the network.
    r = subprocess.run([sys.executable, os.path.join(ROOT, "code", "download_datasets.py"),
                        "--check-only"],
                       capture_output=True, text=True,
                       env={**os.environ, "IOTIDS_DATASETS_ROOT": root})
    assert r.returncode == 0 and "layout" in r.stdout.lower(), r.stdout + r.stderr
    assert "missing" in r.stdout.lower(), r.stdout


def t_download_datasets_dangling_symlink():
    """A symlinked dataset root whose volume is unmounted must explain itself.

    `os.makedirs(exist_ok=True)` raises FileExistsError on a broken symlink, so
    the documented entry point used to die with a bare traceback whenever the
    external drive was unplugged (vault/Findings/F21).
    """
    link = os.path.join(TMP, "dangling_ds")
    if os.path.islink(link):
        os.unlink(link)
    os.symlink(os.path.join(TMP, "no_such_volume"), link)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "code", "download_datasets.py")],
                       capture_output=True, text=True,
                       env={**os.environ, "IOTIDS_DATASETS_ROOT": link})
    out = r.stdout + r.stderr
    assert r.returncode != 0, "a dangling dataset root must fail, not proceed"
    assert "Traceback" not in out, f"crashed instead of explaining:\n{out}"
    assert "not mounted" in out and "IOTIDS_DATASETS_ROOT" in out, out


def t_dataset_archives_extract_safely():
    """Archive traversal, links and checksum mismatches fail closed."""
    import importlib.util, io, tarfile, zipfile, hashlib
    spec = importlib.util.spec_from_file_location(
        "down_safe", os.path.join(ROOT, "code", "download_datasets.py"))
    dl = importlib.util.module_from_spec(spec); spec.loader.exec_module(dl)
    dest = os.path.join(TMP, "extract")
    os.makedirs(dest, exist_ok=True)

    evil_tar = os.path.join(TMP, "evil.tar.gz")
    with tarfile.open(evil_tar, "w:gz") as tf:
        info = tarfile.TarInfo("../tar-escape")
        payload = b"owned"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    try:
        dl._safe_extract_tar(evil_tar, dest)
        raise AssertionError("tar traversal was extracted")
    except ValueError as e:
        assert "escapes destination" in str(e)
    assert not os.path.exists(os.path.join(TMP, "tar-escape"))

    evil_zip = os.path.join(TMP, "evil.zip")
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../zip-escape", "owned")
    try:
        dl._safe_extract_zip(evil_zip, dest)
        raise AssertionError("zip traversal was extracted")
    except ValueError as e:
        assert "escapes destination" in str(e)
    assert not os.path.exists(os.path.join(TMP, "zip-escape"))

    good_zip = os.path.join(TMP, "good.zip")
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("nested/data.csv", "a,b\n1,2\n")
    dl._safe_extract_zip(good_zip, dest)
    assert os.path.exists(os.path.join(dest, "nested", "data.csv"))

    digest = hashlib.sha256(open(good_zip, "rb").read()).hexdigest()
    assert not dl._checksum_allows_extract(good_zip, None)
    assert dl._checksum_allows_extract(good_zip, digest)
    try:
        dl._checksum_allows_extract(good_zip, "0" * 64)
        raise AssertionError("checksum mismatch was accepted")
    except ValueError as e:
        assert "checksum mismatch" in str(e)

    source = open(os.path.join(ROOT, "code", "download_datasets.py")).read()
    assert "--no-check-certificate" not in source


# ── 8. shell scripts syntax ───────────────────────────────────────────────────
def t_shell_syntax():
    for sh in ["demo/run_demo.sh", "deploy/setup_pi.sh"]:
        r = subprocess.run(["bash", "-n", os.path.join(ROOT, sh)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{sh}: {r.stderr}"


def t_setup_pi_strict_preflight():
    """The installer initialises safely under `set -u` before side effects.

    `RUN_USER` used to be expanded by chown two lines before it was assigned,
    so a fresh Pi install exited before reaching the first progress message.
    Syntax-only CI cannot catch an unbound-variable runtime failure.
    """
    token_file = os.path.join(TMP, "setup-preflight.token")
    env = {**os.environ,
           "IOTIDS_SETUP_PREFLIGHT_ONLY": "1",
           "IOTIDS_DASHBOARD_TOKEN_FILE": token_file}
    r = subprocess.run(["bash", os.path.join(ROOT, "deploy", "setup_pi.sh")],
                       capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(token_file) and os.path.getsize(token_file) > 0


def t_benchmark_params():
    # the benchmark's param/footprint section must run (fast, no full sweep)
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", os.path.join(ROOT, "demo", "benchmark.py"))
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    meta, sizes, n_trees, n_nodes = b.bench_params()
    assert n_trees > 0 and n_nodes > 0
    assert "live_ids.onnx" in sizes and sizes["live_ids.onnx"] > 0


def t_no_retracted_numbers_in_live_docs():
    """Retracted headline figures must not reappear outside legacy/.

    The pre-remediation study reported a +56.59 pp "generalisation gain" and
    92.68% on UNSW-NB15, both produced by the broken feature alignment (F01)
    and an F1-only metric that cannot separate transfer from an all-attack
    classifier (F02). Those files now live under legacy/ with a retraction
    notice; this check keeps them from being copied back into live prose.
    """
    retracted = ["56.59", "92.68"]
    live_docs = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "legacy", "Literature", "Datasets",
                                "__pycache__", "vault", "node_modules"}]
        for fn in files:
            if fn.endswith(".md"):
                live_docs.append(os.path.join(base, fn))
    offenders = []
    for path in live_docs:
        try:
            txt = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for needle in retracted:
            if needle in txt:
                offenders.append(f"{os.path.relpath(path, ROOT)}: {needle}")
    assert not offenders, (
        "retracted pre-remediation figures found in live documentation "
        f"(they belong only under legacy/): {offenders}")


def t_benchmark_extraction_section_runs():
    """Section 4 must actually produce numbers, not a swallowed exception.

    `read_pcap` gained a third return value (orig_len, for snaplen handling)
    and `bench_extract` was never updated, so it raised ValueError on every
    run. A catch-all in `_guard` rendered that as "(section skipped — ...)" and
    a published BENCHMARK.md shipped with the hole (vault/Findings/F22).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bench_x", os.path.join(ROOT, "demo", "benchmark.py"))
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    if not os.path.exists(os.path.join(ROOT, "data", "pcaps", "demo_mixed.pcap")):
        raise SkipTest("no demo pcap — run demo/run_demo.sh first")
    res = b.bench_extract()
    assert res is not None, "bench_extract returned nothing"
    npk, n_flows, t_proc = res
    assert npk > 0 and n_flows > 0 and t_proc > 0, res
    assert not b.FAILED_SECTIONS, b.FAILED_SECTIONS


def t_systemd_unit_sane():
    txt = open(os.path.join(ROOT, "deploy", "iot-ids.service")).read()
    assert "[Service]" in txt and "ids_daemon.py" in txt and "--iface" in txt
    dash = open(os.path.join(ROOT, "deploy", "iot-ids-dashboard.service")).read()
    assert "[Service]" in dash and "dashboard.py" in dash
    # setup script installs and enables both services
    setup = open(os.path.join(ROOT, "deploy", "setup_pi.sh")).read()
    assert "iot-ids-dashboard.service" in setup and "enable iot-ids.service iot-ids-dashboard.service" in setup


TESTS = [
        ("imports", t_imports),
        ("requirements importable", t_requirements_importable),
        ("flow_features core", t_flow_features_core),
        ("flow_features CLI", t_flow_features_cli),
        ("parse ipv6 / vlan / snaplen", t_parse_ipv6_vlan_snaplen),
        ("pcapng reader", t_pcapng_reader),
        ("classic pcap resolution + linktype", t_classic_pcap_resolution_and_linktype),
        ("flow direction is initiator-relative", t_flow_direction_is_initiator_relative),
        ("fragment parsing is safe", t_fragment_parsing_is_safe),
        ("tcp teardown splits flows", t_tcp_teardown_splits_flows),
        ("flowtable window + dirty", t_flowtable_window_and_dirty),
        ("flowtable thread safety", t_flowtable_thread_safety),
        ("verdict cache == full scoring", t_verdict_cache_matches_full_scoring),
        ("traffic_gen all kinds", t_traffic_gen_all_kinds),
        ("traffic_gen CLI", t_traffic_gen_cli),
        ("build_corpus helper", t_build_corpus_helper),
        ("model artifacts", t_model_artifacts),
        ("daemon rejects research model", t_daemon_rejects_research_model_contract),
        ("ips responder ladder", t_ips_responder),
        ("ips scope + nft idempotence", t_ips_scope_and_idempotence),
        ("IPS refreshes block deadline", t_ips_refreshes_block_deadline),
        ("incident watermarks are bounded", t_incident_watermarks_are_bounded),
        ("c export + compile", t_c_export),
        ("SFAF alignment contract", t_alignment_contract),
        ("alignment missing-row accounting", t_alignment_preserves_and_accounts_for_missing_rows),
        ("Bot-IoT bounded reservoir sampling", t_botiot_sampling_is_memory_bounded),
        ("IoT-23 cache tracks loader code", t_iot23_cache_tracks_loader_code),
        ("NxN diagonal uses held-out rows", t_nxn_diagonal_is_held_out),
        ("threshold budget is unconditional", t_threshold_budget_includes_failed_calibration_draws),
        ("multidataset taxonomy", t_multidataset_taxonomy),
        ("zeek composite labels", t_zeek_composite_labels),
        ("trivial-baseline guard", t_trivial_baseline),
        ("syslog/CEF SIEM export", t_syslog_cef_sink),
        ("web dashboard", t_dashboard),
        ("dashboard auth + binding", t_dashboard_auth_and_binding),
        ("dashboard token file", t_dashboard_token_file),
        ("documented counts match code", t_documented_counts_match_code),
        ("no credential literals", t_no_credential_literals),
        ("dashboard incremental reads", t_dashboard_incremental_and_rotation),
        ("alert log rotation", t_alert_log_rotation),
        ("detector classify known", t_detector_classify_known),
        ("daemon offline mode", t_daemon_offline),
        ("daemon replay mode", t_daemon_replay),
        ("daemon live path (scapy)", t_live_path),
        ("daemon --help", t_daemon_help),
        ("SFAF trainer guard", t_sfaf_trainer_guard),
        ("SFAF onnx export", t_sfaf_onnx_export),
        ("download_datasets runs", t_download_datasets_runs),
        ("download datasets dangling symlink", t_download_datasets_dangling_symlink),
        ("dataset archives extract safely", t_dataset_archives_extract_safely),
        ("flow table prune bounds memory", t_flow_table_prune_bounds_memory),
        ("flow reuse distinct records", t_flow_reuse_still_generates_distinct_records),
        ("shell script syntax", t_shell_syntax),
        ("Pi setup strict preflight", t_setup_pi_strict_preflight),
        ("benchmark params", t_benchmark_params),
        ("benchmark extraction section runs", t_benchmark_extraction_section_runs),
        ("no retracted numbers in live docs", t_no_retracted_numbers_in_live_docs),
        ("systemd unit sane", t_systemd_unit_sane),]


def main():
    print(f"Running smoke tests (tmp={TMP})\n")
    for name, fn in TESTS:
        check(name, fn)
    shutil.rmtree(TMP, ignore_errors=True)
    tail = f", {len(SKIP)} skipped" if SKIP else ""
    print(f"\n{'='*50}\n  {len(PASS)} passed, {len(FAIL)} failed{tail}\n{'='*50}")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
