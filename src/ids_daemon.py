#!/usr/bin/env python3
"""
ids_daemon.py — real-time IoT Intrusion Detection daemon.

One detection core, three feeding modes:

  offline :  python src/ids_daemon.py --pcap  data/pcaps/demo_mixed.pcap
             classify a capture and print an aggregated report (no root)

  replay  :  python src/ids_daemon.py --replay data/pcaps/demo_mixed.pcap
             stream a pcap through the live windowed logic with progressive,
             aggregated alerts — a live feel with no root (Mac demo)

  live    :  sudo python src/ids_daemon.py --iface eth0
             sniff a real interface (needs root — Raspberry Pi)

Alerts are aggregated per (source, attack-type): a port scan that touches 500
ports becomes ONE "portscan from X — 500 ports" alert, not 500 lines — the way
a real sensor reports.  Runtime deps: onnxruntime + numpy (+ scapy for --iface).
The ONNX model has the scaler baked in, so nothing else is needed on the edge.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import datetime as dt
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from flow_features import (  # noqa: E402
    FlowTable, parse_raw, normalize_scapy, read_pcap, FEATURE_NAMES,
)

MODELS = os.path.join(ROOT, "models")
DEFAULT_MODEL = os.path.join(MODELS, "live_ids.onnx")
DEFAULT_META = os.path.join(MODELS, "live_meta.json")
PROTO_NAME = {6: "TCP", 17: "UDP", 1: "ICMP"}

_TTY = sys.stdout.isatty()
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s
RED = lambda s: _c("1;31", s)
GRN = lambda s: _c("1;32", s)
YEL = lambda s: _c("1;33", s)
CYA = lambda s: _c("36", s)
DIM = lambda s: _c("2", s)


class Detector:
    def __init__(self, model_path=DEFAULT_MODEL, meta_path=DEFAULT_META):
        import onnxruntime as rt
        if not os.path.exists(model_path):
            sys.exit(f"[ERROR] model not found: {model_path}\n"
                     f"        Run: python src/train_live_model.py")
        self.sess = rt.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.labels = {int(k): v for k, v in self.meta["labels"].items()}
        assert self.meta["features"] == FEATURE_NAMES, "feature order mismatch!"

    def classify(self, vectors):
        if not vectors:
            return []
        X = np.asarray(vectors, dtype=np.float32)
        out = self.sess.run(None, {self.input_name: X})
        labels = np.asarray(out[0]).ravel().astype(int)
        probs = np.asarray(out[1]) if len(out) > 1 else None
        res = []
        for i, lab in enumerate(labels):
            conf = float(probs[i][lab]) if probs is not None else 1.0
            res.append((self.labels.get(int(lab), str(lab)), conf))
        return res


def aggregate(metas, results):
    """Group attack flows by (src_ip, kind) into one incident each."""
    inc = collections.OrderedDict()
    for meta, (kind, conf) in zip(metas, results):
        if kind == "benign":
            continue
        src_ip = meta["src"].rsplit(":", 1)[0]
        key = (src_ip, kind)
        a = inc.get(key)
        if a is None:
            a = {"src_ip": src_ip, "kind": kind, "flows": 0, "pkts": 0,
                 "bytes": 0, "dst_ips": set(), "dst_ports": set(),
                 "proto": meta["proto"], "conf_sum": 0.0}
            inc[key] = a
        dip, dport = meta["dst"].rsplit(":", 1)
        a["flows"] += 1
        a["pkts"] += meta["pkts"]
        a["bytes"] += meta["bytes"]
        a["dst_ips"].add(dip)
        a["dst_ports"].add(dport)
        a["conf_sum"] += conf
    for a in inc.values():
        a["n_dst_ips"] = len(a["dst_ips"])
        a["n_dst_ports"] = len(a["dst_ports"])
        a["avg_conf"] = a["conf_sum"] / max(a["flows"], 1)
        del a["dst_ips"], a["dst_ports"], a["conf_sum"]
    return list(inc.values())


def fmt_incident(a):
    proto = PROTO_NAME.get(a["proto"], str(a["proto"]))
    conf = DIM("conf~%.2f" % a["avg_conf"])
    return (f"  {RED('⚠ ATTACK')} {YEL(a['kind']):<22} "
            f"src={a['src_ip']:<15} {proto:<4} "
            f"{a['flows']:>5} flows  {a['n_dst_ports']:>4} dst-ports  "
            f"{a['n_dst_ips']:>3} dst-ips  {a['pkts']:>6} pkts  {conf}")


class AlertLog:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fh = open(path, "a")
        self.n = 0
        self.by_kind = collections.Counter()

    def emit(self, a):
        self.n += 1
        self.by_kind[a["kind"]] += 1
        rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
               "src_ip": a["src_ip"], "kind": a["kind"], "proto": a["proto"],
               "flows": a["flows"], "pkts": a["pkts"], "bytes": a["bytes"],
               "dst_ips": a["n_dst_ips"], "dst_ports": a["n_dst_ports"],
               "confidence": round(a["avg_conf"], 4)}
        self.fh.write(json.dumps(rec) + "\n"); self.fh.flush()

    def close(self):
        self.fh.close()


def _summary(n_flows, n_benign, incidents, alog, n_pkts, infer_ms):
    n_attack = n_flows - n_benign
    print("\n" + "=" * 64)
    print("  DETECTION SUMMARY")
    print("=" * 64)
    print(f"  Packets parsed     : {n_pkts:,}")
    print(f"  Flows classified   : {n_flows:,}")
    print(f"  {GRN('Benign flows')}       : {n_benign:,}")
    print(f"  {RED('Attack flows')}       : {n_attack:,}")
    print(f"  {RED('Incidents (src×type)')}: {len(incidents)}")
    kinds = collections.Counter(a["kind"] for a in incidents)
    for k, c in kinds.most_common():
        tot = sum(a["flows"] for a in incidents if a["kind"] == k)
        print(f"       {RED('•')} {k:<16} {c} source(s), {tot:,} flows")
    if n_flows and infer_ms:
        print(f"  Avg inference      : {infer_ms/n_flows*1000:.2f} µs/flow "
              f"({infer_ms:.1f} ms total)")
    print("=" * 64)


# ── OFFLINE MODE ──────────────────────────────────────────────────────────────
def run_offline(pcap_path, det, alog, csv_out=None):
    print(CYA(f"\n[*] Offline analysis: {pcap_path}"))
    table = FlowTable()
    n_pkts = 0
    for ts, raw in read_pcap(pcap_path):
        pk = parse_raw(raw)
        if pk is not None:
            table.add_packet(pk, ts); n_pkts += 1
    flows = table.extract(min_pkts=1, window=None)
    metas = [m for m, _ in flows]
    vecs = [v for _, v in flows]
    t1 = time.perf_counter()
    results = det.classify(vecs)
    infer_ms = (time.perf_counter() - t1) * 1000

    n_benign = sum(1 for k, _ in results if k == "benign")
    incidents = sorted(aggregate(metas, results), key=lambda a: -a["flows"])
    for a in incidents:
        print(fmt_incident(a)); alog.emit(a)
    _summary(len(flows), n_benign, incidents, alog, n_pkts, infer_ms)

    if csv_out:
        import csv
        rows = [{**m, "verdict": k, "confidence": round(c, 4)}
                for m, (k, c) in zip(metas, results)]
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(DIM(f"[*] Per-flow results -> {csv_out}"))
    return incidents


# ── REPLAY MODE (offline pcap, live-style progressive alerts) ─────────────────
def run_replay(pcap_path, det, alog, window=60.0, step=1.0, speed=0.0, min_conf=0.5):
    print(CYA(f"\n[*] Replay (live-style): {pcap_path}  "
              f"window={window}s step={step}s min_conf={min_conf}"))
    packets = sorted(read_pcap(pcap_path), key=lambda x: x[0])
    if not packets:
        print("[WARN] empty pcap"); return []
    t0 = packets[0][0]
    table = FlowTable()
    seen = {}          # (src,kind) -> last flow count alerted
    next_ckpt = t0 + step
    n_pkts = 0

    def flush(now):
        flows = table.extract(min_pkts=1, window=window)
        metas = [m for m, _ in flows]
        results = det.classify([v for _, v in flows])
        for a in sorted(aggregate(metas, results), key=lambda x: -x["flows"]):
            if a["avg_conf"] < min_conf:
                continue
            key = (a["src_ip"], a["kind"])
            prev = seen.get(key, 0)
            # alert on first sight or when the incident grows materially
            if a["flows"] >= prev * 1.5 or prev == 0:
                seen[key] = a["flows"]
                rel = now - t0
                print(f"  {DIM(f'[t+{rel:6.1f}s]')}{fmt_incident(a)}")
                alog.emit(a)

    for ts, raw in packets:
        pk = parse_raw(raw)
        if pk is not None:
            table.add_packet(pk, ts); n_pkts += 1
        if ts >= next_ckpt:
            flush(ts)
            if speed > 0:
                time.sleep(step * speed)
            next_ckpt += step
    flush(packets[-1][0])

    flows = table.extract(min_pkts=1, window=None)
    results = det.classify([v for _, v in flows])
    n_benign = sum(1 for k, _ in results if k == "benign")
    incidents = aggregate([m for m, _ in flows], results)
    _summary(len(flows), n_benign, incidents, alog, n_pkts, 0)
    return incidents


# ── LIVE MODE ─────────────────────────────────────────────────────────────────
def run_live(iface, det, alog, window=60.0, flush_s=2.0, idle_evict=120.0, min_conf=0.5):
    try:
        from scapy.all import AsyncSniffer
    except Exception:
        sys.exit("[ERROR] scapy required for --iface mode: pip install scapy")

    print(CYA(f"\n[*] Live IDS on {iface}  (Ctrl-C to stop)"))
    print(DIM(f"    window={window}s flush={flush_s}s model=live_ids.onnx"))
    table = FlowTable()
    seen = {}
    stats = {"pkts": 0}

    def on_pkt(p):
        pk = normalize_scapy(p)
        if pk is not None:
            table.add_packet(pk, time.time()); stats["pkts"] += 1

    sniffer = AsyncSniffer(iface=iface, prn=on_pkt, store=False)
    sniffer.start()
    try:
        while True:
            time.sleep(flush_s)
            now = time.time()
            flows = table.extract(min_pkts=1, window=window)
            metas = [m for m, _ in flows]
            results = det.classify([v for _, v in flows])
            for a in sorted(aggregate(metas, results), key=lambda x: -x["flows"]):
                if a["avg_conf"] < min_conf:
                    continue
                key = (a["src_ip"], a["kind"])
                if a["flows"] >= seen.get(key, 0) * 1.5 or key not in seen:
                    seen[key] = a["flows"]
                    print(fmt_incident(a)); alog.emit(a)
            table.prune(older_than=idle_evict, now=now)
            print(DIM(f"  [{dt.datetime.now():%H:%M:%S}] pkts={stats['pkts']:,} "
                      f"flows={len(table.flows):,} incidents={alog.n}"), end="\r")
    except KeyboardInterrupt:
        print(CYA("\n[*] stopping..."))
    finally:
        sniffer.stop()
        flows = table.extract(min_pkts=1, window=None)
        results = det.classify([v for _, v in flows])
        n_benign = sum(1 for k, _ in results if k == "benign")
        incidents = aggregate([m for m, _ in flows], results)
        _summary(len(flows), n_benign, incidents, alog, stats["pkts"], 0)


def main():
    ap = argparse.ArgumentParser(description="Real-time IoT IDS daemon")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pcap", help="offline: classify a capture file")
    g.add_argument("--replay", help="offline: replay a pcap, live-style alerts")
    g.add_argument("--iface", help="live: sniff an interface (needs root)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "alerts.jsonl"))
    ap.add_argument("--csv", default=None, help="offline: write per-flow CSV")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=1.0, help="replay/live flush interval")
    ap.add_argument("--speed", type=float, default=0.0, help="replay wall-clock factor")
    ap.add_argument("--min-conf", type=float, default=0.5, dest="min_conf",
                    help="live/replay: min confidence to raise an alert")
    args = ap.parse_args()

    det = Detector(args.model, args.meta)
    alog = AlertLog(args.log)
    try:
        if args.pcap:
            run_offline(args.pcap, det, alog, csv_out=args.csv)
        elif args.replay:
            run_replay(args.replay, det, alog, window=args.window,
                       step=args.step, speed=args.speed, min_conf=args.min_conf)
        else:
            run_live(args.iface, det, alog, window=args.window,
                     flush_s=args.step, min_conf=args.min_conf)
    finally:
        alog.close()


if __name__ == "__main__":
    main()
