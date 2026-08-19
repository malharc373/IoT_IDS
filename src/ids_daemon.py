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
CATEGORIES = {}   # kind -> coarse category, populated when a Detector loads meta

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
        self.categories = self.meta.get("categories", {})
        CATEGORIES.update(self.categories)
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


class VerdictCache:
    """Remembers each flow's last verdict so a flush only scores what changed.

    The live loop used to re-run ONNX over every flow in the table on every
    flush (default 2 s) until the 120 s idle eviction — so the same port scan
    was re-classified ~60 times, and per-flush cost tracked table size rather
    than new traffic (vault/Findings/F04).

    Aggregation still needs a verdict for *every* windowed flow, not just the
    changed ones, or an incident's flow count would reset each flush. So the
    cache returns cached verdicts for clean flows and scores only dirty ones.
    """

    def __init__(self, detector):
        self.det = detector
        self.verdicts = {}          # flow key -> (kind, confidence)
        self.scored = 0             # flows actually run through the model
        self.reused = 0             # flows served from cache

    def classify_rows(self, rows):
        """rows: list of (key, meta, vector, needs_scoring) from extract_live.

        Returns (metas, results) aligned, exactly as if everything was scored.
        """
        todo = [(i, r) for i, r in enumerate(rows)
                if r[3] or r[0] not in self.verdicts]
        if todo:
            fresh = self.det.classify([r[2] for _, r in todo])
            for (i, r), verdict in zip(todo, fresh):
                self.verdicts[r[0]] = verdict
        self.scored += len(todo)
        self.reused += len(rows) - len(todo)
        metas = [r[1] for r in rows]
        results = [self.verdicts[r[0]] for r in rows]
        return metas, results

    def forget(self, keys):
        for k in keys:
            self.verdicts.pop(k, None)

    def stats(self):
        tot = self.scored + self.reused
        pct = (self.reused / tot * 100) if tot else 0.0
        return f"scored={self.scored:,} reused={self.reused:,} ({pct:.0f}% cached)"


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
    cat = CATEGORIES.get(a["kind"], "")
    label = f"{cat}/{a['kind']}" if cat and cat != "benign" else a["kind"]
    return (f"  {RED('⚠ ATTACK')} {YEL(label):<30} "
            f"src={a['src_ip']:<15} {proto:<4} "
            f"{a['flows']:>5} flows  {a['n_dst_ports']:>4} dst-ports  "
            f"{a['n_dst_ips']:>3} dst-ips  {a['pkts']:>6} pkts  {conf}")


class AlertLog:
    """Append-only JSONL alert feed with size-based rotation.

    The feed had no bound: a long-running sensor grew logs/alerts.jsonl
    forever, and the dashboard re-parsed the whole thing on every 2 s poll from
    every client (vault/Findings/F10). Rotation keeps both costs bounded, and
    the dashboard detects the rotation and re-reads cleanly.
    """

    def __init__(self, path, max_bytes=32 * 1024 * 1024, backups=3):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self.fh = open(path, "a")
        self.n = 0
        self.by_kind = collections.Counter()

    def _rotate_if_needed(self):
        if self.max_bytes <= 0:
            return
        try:
            if self.fh.tell() < self.max_bytes:
                return
        except Exception:
            return
        self.fh.close()
        for i in range(self.backups - 1, 0, -1):
            src, dst = f"{self.path}.{i}", f"{self.path}.{i+1}"
            if os.path.exists(src):
                os.replace(src, dst)
        os.replace(self.path, f"{self.path}.1")
        self.fh = open(self.path, "a")

    def emit(self, a):
        self.n += 1
        self.by_kind[a["kind"]] += 1
        rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
               "src_ip": a["src_ip"], "kind": a["kind"], "proto": a["proto"],
               "flows": a["flows"], "pkts": a["pkts"], "bytes": a["bytes"],
               "dst_ips": a["n_dst_ips"], "dst_ports": a["n_dst_ports"],
               "confidence": round(a["avg_conf"], 4)}
        self.fh.write(json.dumps(rec) + "\n"); self.fh.flush()
        self._rotate_if_needed()

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
    for ts, raw, orig_len in read_pcap(pcap_path):
        pk = parse_raw(raw, orig_len)
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
def run_replay(pcap_path, det, alog, window=60.0, step=1.0, speed=0.0, min_conf=0.5,
               responder=None):
    print(CYA(f"\n[*] Replay (live-style): {pcap_path}  "
              f"window={window}s step={step}s min_conf={min_conf}"))
    packets = sorted(read_pcap(pcap_path), key=lambda x: x[0])
    if not packets:
        print("[WARN] empty pcap"); return []
    t0 = packets[0][0]
    table = FlowTable()
    cache = VerdictCache(det)
    seen = {}          # (src,kind) -> last flow count alerted
    next_ckpt = t0 + step
    n_pkts = 0

    def flush(now):
        rows = table.extract_live(min_pkts=1, window=window)
        metas, results = cache.classify_rows(rows)
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
                if responder is not None:
                    responder.handle(a["src_ip"], a["kind"], a["avg_conf"])
        if responder is not None:
            responder.expire()

    for ts, raw, orig_len in packets:
        pk = parse_raw(raw, orig_len)
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
    print(DIM(f"  [cache] {cache.stats()}"))
    return incidents


# ── LIVE MODE ─────────────────────────────────────────────────────────────────
def run_live(iface, det, alog, window=60.0, flush_s=2.0, idle_evict=120.0, min_conf=0.5,
             responder=None):
    try:
        from scapy.all import AsyncSniffer
    except Exception:
        sys.exit("[ERROR] scapy required for --iface mode: pip install scapy")

    print(CYA(f"\n[*] Live IDS on {iface}  (Ctrl-C to stop)"))
    print(DIM(f"    window={window}s flush={flush_s}s model=live_ids.onnx"))
    table = FlowTable()
    cache = VerdictCache(det)
    seen = {}
    stats = {"pkts": 0, "last_ts": time.time()}

    def on_pkt(p):
        pk = normalize_scapy(p)
        if pk is None:
            return
        # Use the CAPTURE timestamp, not the time this callback happened to run.
        # Under burst load the callback lags the wire by a variable amount, which
        # distorts every inter-arrival feature relative to training — where pcap
        # timestamps are used (vault/Findings/F05).
        ts = float(getattr(p, "time", 0.0)) or time.time()
        table.add_packet(pk, ts)
        stats["pkts"] += 1
        if ts > stats["last_ts"]:
            stats["last_ts"] = ts

    sniffer = AsyncSniffer(iface=iface, prn=on_pkt, store=False)
    sniffer.start()
    try:
        while True:
            time.sleep(flush_s)
            now = stats["last_ts"]
            rows = table.extract_live(min_pkts=1, window=window)
            metas, results = cache.classify_rows(rows)
            for a in sorted(aggregate(metas, results), key=lambda x: -x["flows"]):
                if a["avg_conf"] < min_conf:
                    continue
                key = (a["src_ip"], a["kind"])
                if a["flows"] >= seen.get(key, 0) * 1.5 or key not in seen:
                    seen[key] = a["flows"]
                    print(fmt_incident(a)); alog.emit(a)
                    if responder is not None:
                        responder.handle(a["src_ip"], a["kind"], a["avg_conf"])
            if responder is not None:
                responder.expire()
            cache.forget(table.prune(older_than=idle_evict, now=now))
            print(DIM(f"  [{dt.datetime.now():%H:%M:%S}] pkts={stats['pkts']:,} "
                      f"flows={len(table):,} incidents={alog.n} "
                      f"| {cache.stats()}"), end="\r")
    except KeyboardInterrupt:
        print(CYA("\n[*] stopping..."))
    finally:
        sniffer.stop()
        flows = table.extract(min_pkts=1, window=None)
        results = det.classify([v for _, v in flows])
        n_benign = sum(1 for k, _ in results if k == "benign")
        incidents = aggregate([m for m, _ in flows], results)
        _summary(len(flows), n_benign, incidents, alog, stats["pkts"], 0)
        print(DIM(f"  [cache] {cache.stats()}"))


def main():
    ap = argparse.ArgumentParser(description="Real-time IoT IDS daemon")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pcap", help="offline: classify a capture file")
    g.add_argument("--replay", help="offline: replay a pcap, live-style alerts")
    g.add_argument("--iface", help="live: sniff an interface (needs root)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "alerts.jsonl"))
    ap.add_argument("--log-max-mb", type=int, default=32, dest="log_max_mb",
                    help="rotate the alert log past this size, 0 = never "
                         "(default 32)")
    ap.add_argument("--csv", default=None, help="offline: write per-flow CSV")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=1.0, help="replay/live flush interval")
    ap.add_argument("--speed", type=float, default=0.0, help="replay wall-clock factor")
    ap.add_argument("--min-conf", type=float, default=0.5, dest="min_conf",
                    help="live/replay: min confidence to raise an alert")
    # IPS (active response) — opt-in, dry-run unless --prevent
    ap.add_argument("--ips", action="store_true",
                    help="enable IPS layer in dry-run (logs would-block actions)")
    ap.add_argument("--prevent", action="store_true",
                    help="IPS enforce mode: actually block sources (needs root+nft/iptables)")
    ap.add_argument("--ips-min-conf", type=float, default=0.9, dest="ips_min_conf",
                    help="min confidence for the IPS to act (default 0.9)")
    ap.add_argument("--block-seconds", type=int, default=300, dest="block_seconds",
                    help="how long an IPS block lasts (default 300s)")
    ap.add_argument("--allow", action="append", default=[],
                    help="IP/CIDR the IPS must never block (repeatable)")
    ap.add_argument("--ips-scope", default="host", choices=("host", "network"),
                    dest="ips_scope",
                    help="host = INPUT only (passive sensor, protects this box); "
                         "network = INPUT+FORWARD (inline sensor, protects the "
                         "devices behind it)")
    ap.add_argument("--ips-strikes", type=int, default=3, dest="ips_strikes",
                    help="corroborating incidents required before blocking "
                         "(model confidence is uncalibrated; default 3)")
    ap.add_argument("--ips-strike-window", type=int, default=120,
                    dest="ips_strike_window",
                    help="seconds over which strikes are counted (default 120)")
    ap.add_argument("--ips-throttle-pps", type=int, default=20,
                    dest="ips_throttle_pps",
                    help="packets/s a throttled source is limited to (default 20)")
    args = ap.parse_args()

    det = Detector(args.model, args.meta)
    alog = AlertLog(args.log, max_bytes=args.log_max_mb * 1024 * 1024)

    responder = None
    if args.ips or args.prevent:
        from ips_response import Responder
        responder = Responder(mode="enforce" if args.prevent else "dry-run",
                              min_conf=args.ips_min_conf,
                              block_seconds=args.block_seconds,
                              allowlist=args.allow,
                              scope=args.ips_scope,
                              strikes=args.ips_strikes,
                              strike_window=args.ips_strike_window,
                              throttle_pps=args.ips_throttle_pps)
        print(CYA(f"[IPS] {json.dumps(responder.status())}"))

    try:
        if args.pcap:
            run_offline(args.pcap, det, alog, csv_out=args.csv)
        elif args.replay:
            run_replay(args.replay, det, alog, window=args.window,
                       step=args.step, speed=args.speed, min_conf=args.min_conf,
                       responder=responder)
        else:
            run_live(args.iface, det, alog, window=args.window,
                     flush_s=args.step, min_conf=args.min_conf,
                     responder=responder)
    finally:
        alog.close()


if __name__ == "__main__":
    main()
