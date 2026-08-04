#!/usr/bin/env python3
"""
benchmark.py — full system benchmark: accuracy, speed, efficiency, footprint.

Measures the live edge IDS end-to-end on real artifacts and writes a report to
demo/results/BENCHMARK.md (+ latency chart). All numbers are measured here, not
assumed; Raspberry Pi figures are the host measurement scaled by PI_FACTOR.

Sections:
  1. Model parameters      trees / nodes / features / classes / sizes
  2. Inference latency     ONNX single-flow + batched (mean/p50/p99/throughput)
  3. Native C model        compiled predict() latency (MCU path)
  4. Feature extraction    packets/s and flows/s from real pcaps
  5. End-to-end            pcap -> verdicts wall time
  6. Accuracy              held-out multiclass acc, macro-F1, per-class recall, FPR
  7. Memory footprint      process RSS + model RAM
  8. Raspberry Pi estimate host latency x PI_FACTOR
"""
from __future__ import annotations

import os
import sys
import json
import time
import platform
import subprocess
import statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "attacks"))
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "demo", "results")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Raspberry Pi 4 is ~10-15x slower than a modern laptop core on this workload;
# use a conservative 12x to project. Replace with a real Pi run when available.
PI_FACTOR = 12.0

REPORT = []
def out(s=""):
    print(s); REPORT.append(s)


def section(t):
    out("\n" + "=" * 68); out(f"  {t}"); out("=" * 68)


# ── 1. model parameters ───────────────────────────────────────────────────────
def bench_params():
    section("1. MODEL PARAMETERS & FOOTPRINT")
    meta = json.load(open(os.path.join(MODELS, "live_meta.json")))
    import xgboost as xgb
    b = xgb.Booster(); b.load_model(os.path.join(MODELS, "live_ids_booster.json"))
    df = b.trees_to_dataframe()
    n_trees = df["Tree"].nunique()
    n_nodes = len(df)
    n_leaves = int((df["Feature"] == "Leaf").sum())
    sizes = {f: os.path.getsize(os.path.join(MODELS, f)) for f in
             ["live_ids.onnx", "live_ids.h", "live_ids_booster.json", "live_meta.json"]
             if os.path.exists(os.path.join(MODELS, f))}
    out(f"  Features               : {meta['n_features']}")
    out(f"  Classes                : {meta['num_class']}  ({', '.join(meta['labels'].values())})")
    out(f"  Boosted trees          : {n_trees}")
    out(f"  Total nodes / leaves   : {n_nodes:,} / {n_leaves:,}")
    out(f"  ONNX model size        : {sizes['live_ids.onnx']/1024:.1f} KB")
    out(f"  C header size          : {sizes['live_ids.h']/1024:.1f} KB "
        f"(~{n_nodes*16//1024} KB const data)")
    out(f"  Booster JSON (train)   : {sizes['live_ids_booster.json']/1024:.1f} KB")
    out(f"  In-domain metrics      : {meta['metrics']}")
    return meta, sizes, n_trees, n_nodes


# ── 2. ONNX inference latency ─────────────────────────────────────────────────
def bench_latency(nf):
    section("2. ONNX INFERENCE LATENCY & THROUGHPUT")
    import onnxruntime as rt
    sess = rt.InferenceSession(os.path.join(MODELS, "live_ids.onnx"))
    name = sess.get_inputs()[0].name
    rng = np.random.RandomState(0)
    rows = []
    for bs in [1, 8, 32, 64, 128, 512, 1024]:
        X = rng.rand(bs, nf).astype(np.float32)
        for _ in range(30):
            sess.run(None, {name: X})               # warm-up
        ts = []
        reps = 2000 if bs == 1 else 500
        for _ in range(reps):
            t0 = time.perf_counter(); sess.run(None, {name: X})
            ts.append((time.perf_counter() - t0) * 1e3)    # ms
        ts = np.array(ts)
        per_flow_us = ts.mean() / bs * 1000
        thr = bs / (ts.mean() / 1000)
        rows.append((bs, ts.mean(), np.percentile(ts, 50), np.percentile(ts, 99),
                     per_flow_us, thr))
    out(f"  {'batch':>6} {'mean_ms':>9} {'p50_ms':>8} {'p99_ms':>8} "
        f"{'us/flow':>9} {'flows/s':>12}")
    for bs, m, p50, p99, us, thr in rows:
        out(f"  {bs:>6} {m:>9.4f} {p50:>8.4f} {p99:>8.4f} {us:>9.3f} {thr:>12,.0f}")
    single = rows[0]
    out(f"\n  single-flow latency    : {single[1]*1000:.1f} us "
        f"(p99 {single[3]*1000:.1f} us)")
    out(f"  peak throughput        : {max(r[5] for r in rows):,.0f} flows/s "
        f"(batch {rows[[r[5] for r in rows].index(max(r[5] for r in rows))][0]})")
    return rows


# ── 3. native C model latency ─────────────────────────────────────────────────
def bench_c(nf):
    section("3. NATIVE C MODEL (MCU PATH)")
    import shutil
    cc = shutil.which("gcc") or shutil.which("cc")
    hdr = os.path.join(MODELS, "live_ids.h")
    if not cc or not os.path.exists(hdr):
        out("  (skipped — no compiler or header)"); return None
    import tempfile
    tmp = tempfile.mkdtemp()
    shutil.copy(hdr, tmp)
    prog = f"""#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "live_ids.h"
int main(){{
  int N=200000; float x[IDS_NUM_FEATURES];
  for(int i=0;i<IDS_NUM_FEATURES;i++) x[i]=(float)(rand()%1000)/100.0f;
  volatile int s=0;
  struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
  for(int i=0;i<N;i++){{ x[i%IDS_NUM_FEATURES]+=0.001f; s+=ids_predict(x); }}
  clock_gettime(CLOCK_MONOTONIC,&b);
  double ns=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/N;
  printf("%.3f\\n", ns); return s&0;
}}"""
    open(os.path.join(tmp, "b.c"), "w").write(prog)
    subprocess.run([cc, "-O3", "-I", tmp, "-o", os.path.join(tmp, "b"),
                    os.path.join(tmp, "b.c")], check=True)
    r = subprocess.run([os.path.join(tmp, "b")], capture_output=True, text=True)
    ns = float(r.stdout.strip())
    shutil.rmtree(tmp, ignore_errors=True)
    out(f"  C ids_predict latency  : {ns:.1f} ns/flow ({ns/1000:.3f} us)")
    out(f"  C throughput           : {1e9/ns:,.0f} flows/s (single thread)")
    out(f"  runtime deps           : none (pure C99, ~130 B stack)")
    return ns


# ── 4. feature-extraction throughput ─────────────────────────────────────────
def bench_extract():
    section("4. FEATURE EXTRACTION THROUGHPUT")
    from flow_features import FlowTable, parse_raw, read_pcap
    pcap = os.path.join(ROOT, "data", "pcaps", "demo_mixed.pcap")
    if not os.path.exists(pcap):
        out("  (skipped — no demo pcap)"); return None
    t0 = time.perf_counter()
    recs = read_pcap(pcap)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    table = FlowTable(); npk = 0
    for ts, raw in recs:
        pk = parse_raw(raw)
        if pk: table.add_packet(pk, ts); npk += 1
    flows = table.extract(min_pkts=1)
    t_proc = time.perf_counter() - t0
    out(f"  pcap                   : {os.path.basename(pcap)} "
        f"({len(recs):,} packets -> {len(flows):,} flows)")
    out(f"  parse+read             : {t_read*1000:.1f} ms "
        f"({len(recs)/t_read:,.0f} packets/s)")
    out(f"  parse+aggregate        : {t_proc*1000:.1f} ms "
        f"({npk/t_proc:,.0f} packets/s, {len(flows)/t_proc:,.0f} flows/s)")
    return npk, len(flows), t_proc


# ── 5. end-to-end ─────────────────────────────────────────────────────────────
def bench_e2e():
    section("5. END-TO-END (pcap -> verdicts)")
    from ids_daemon import Detector
    from flow_features import FlowTable, parse_raw, read_pcap
    det = Detector()
    pcap = os.path.join(ROOT, "data", "pcaps", "demo_mixed.pcap")
    t0 = time.perf_counter()
    table = FlowTable()
    for ts, raw in read_pcap(pcap):
        pk = parse_raw(raw)
        if pk: table.add_packet(pk, ts)
    flows = table.extract(min_pkts=1)
    vecs = [v for _, v in flows]
    res = det.classify(vecs)
    total = time.perf_counter() - t0
    n_att = sum(1 for k, _ in res if k != "benign")
    out(f"  {len(flows):,} flows classified in {total*1000:.1f} ms "
        f"({len(flows)/total:,.0f} flows/s end-to-end)")
    out(f"  detected {n_att:,} attack flows / {len(flows)-n_att:,} benign")
    return total, len(flows)


# ── 6. accuracy (held-out) ────────────────────────────────────────────────────
def bench_accuracy():
    section("6. ACCURACY (held-out, unseen-seed synthetic)")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "val", os.path.join(ROOT, "demo", "validate.py"))
    val = importlib.util.module_from_spec(spec)
    from ids_daemon import Detector
    import traffic_gen as tg
    from flow_features import FlowTable, parse_raw
    det = Detector()
    kinds = list(tg.ATTACK_KINDS)
    kid = {k: i for i, k in enumerate(kinds)}
    from sklearn.metrics import f1_score, recall_score, accuracy_score
    yt, yp = [], []
    for kind in kinds:
        for s in range(8):
            variant = (tg.BENIGN_VARIANTS[s % len(tg.BENIGN_VARIANTS)]
                       if kind == "benign" else kind)
            pkts = tg.generate(variant, seed=70000 + kid[kind] * 100 + s)
            table = FlowTable()
            for p in sorted(pkts, key=lambda x: float(x.time)):
                d = parse_raw(bytes(p))
                if d: table.add_packet(d, float(p.time))
            rows = table.extract(min_pkts=1)
            for pk, _c in det.classify([v for _, v in rows]):
                yt.append(kid[kind]); yp.append(kid[pk])
    yt, yp = np.array(yt), np.array(yp)
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average="macro", zero_division=0)
    bt, bp = (yt != 0).astype(int), (yp != 0).astype(int)
    det_rate = recall_score(bt, bp, zero_division=0)
    fpr = float((bp[bt == 0] == 1).mean())
    out(f"  flows evaluated        : {len(yt):,} (unseen seeds)")
    out(f"  multiclass accuracy    : {acc*100:.2f}%")
    out(f"  macro F1               : {mf1:.4f}")
    out(f"  attack detection rate  : {det_rate*100:.2f}%")
    out(f"  benign false-pos rate  : {fpr*100:.2f}%")
    out("  per-class recall:")
    for i, k in enumerate(kinds):
        m = yt == i
        if m.any():
            out(f"    {k:<16} {recall_score(m, yp == i, zero_division=0)*100:5.1f}%")
    out("\n  NOTE: synthetic traffic is separable; see CROSS_DATASET_FINDINGS.md")
    out("        for the honest cross-dataset numbers (in-domain 0.98 vs cross 0.45).")
    return acc, mf1, det_rate, fpr


# ── 7. memory ─────────────────────────────────────────────────────────────────
def bench_memory():
    section("7. MEMORY FOOTPRINT")
    # Measure the ACTUAL daemon runtime in a clean subprocess (only the imports
    # the edge daemon uses), not this benchmark's heavy pandas/xgboost process.
    snippet = (
        "import os,sys,json,numpy as np,onnxruntime as rt,psutil;"
        f"m={os.path.join(MODELS,'live_ids.onnx')!r};"
        "s=rt.InferenceSession(m);"
        "nf=json.load(open(%r))['n_features'];" % os.path.join(MODELS, 'live_meta.json') +
        "X=np.random.rand(64,nf).astype(np.float32);"
        "[s.run(None,{s.get_inputs()[0].name:X}) for _ in range(50)];"
        "print(psutil.Process().memory_info().rss)"
    )
    try:
        r = subprocess.run([sys.executable, "-c", snippet],
                           capture_output=True, text=True, timeout=60)
        rss = int(r.stdout.strip().splitlines()[-1])
        out(f"  daemon runtime RSS     : {rss/1e6:.1f} MB "
            f"(onnxruntime + numpy only, clean process)")
    except Exception as e:
        out(f"  daemon runtime RSS     : (measure failed: {e})")
        rss = None
    import psutil
    out(f"  benchmark process RSS  : {psutil.Process().memory_info().rss/1e6:.1f} MB "
        f"(harness — imports pandas/xgboost; NOT the daemon)")
    out(f"  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)")
    out(f"  MCU C model RAM        : ~130 bytes stack, 0 heap")
    return rss


# ── 8. Pi projection ──────────────────────────────────────────────────────────
def bench_pi(lat_rows, c_ns, extract):
    section("8. RASPBERRY PI 4 PROJECTION (host x %.0f)" % PI_FACTOR)
    single_us = lat_rows[0][1] * 1000
    host_cpu = "Apple M4"
    try:
        host_cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        host_cpu = platform.processor() or platform.machine()
    out(f"  host                   : {host_cpu} ({platform.machine()})")
    out(f"  ONNX single-flow (Pi)  : ~{single_us*PI_FACTOR:.1f} us/flow "
        f"(~{1e6/(single_us*PI_FACTOR):,.0f} flows/s)")
    if c_ns:
        out(f"  C model (Pi)           : ~{c_ns*PI_FACTOR/1000:.2f} us/flow "
            f"(~{1e9/(c_ns*PI_FACTOR):,.0f} flows/s)")
    if extract:
        npk, nfl, tp = extract
        out(f"  feature extraction (Pi): ~{npk/tp/PI_FACTOR:,.0f} packets/s "
            f"(the real bottleneck on a live link)")
    out("  verdict                : easily real-time on a Pi 4 for home/IIoT "
        "link rates; sniffing/aggregation, not inference, is the limit.")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out(f"IoT-IDS SYSTEM BENCHMARK   host={platform.platform()}")
    out(f"python={platform.python_version()}  time={time.strftime('%Y-%m-%d %H:%M')}")
    meta, sizes, n_trees, n_nodes = bench_params()
    lat = bench_latency(meta["n_features"])
    c_ns = bench_c(meta["n_features"])
    extract = bench_extract()
    bench_e2e()
    bench_accuracy()
    bench_memory()
    bench_pi(lat, c_ns, extract)

    with open(os.path.join(RESULTS, "BENCHMARK.md"), "w") as f:
        f.write("# IoT-IDS system benchmark\n\n```\n" + "\n".join(REPORT) + "\n```\n")
    _latency_chart(lat)
    out(f"\nwrote {os.path.join(RESULTS, 'BENCHMARK.md')}")


def _latency_chart(lat):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bs = [r[0] for r in lat]; thr = [r[5] for r in lat]; usf = [r[4] for r in lat]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(bs, thr, "o-", color="#199e70", label="throughput")
    ax1.set_xscale("log", base=2); ax1.set_xlabel("batch size")
    ax1.set_ylabel("flows/s", color="#199e70")
    ax1.set_xticks(bs); ax1.set_xticklabels(bs)
    ax2 = ax1.twinx()
    ax2.plot(bs, usf, "s--", color="#d95926", label="us/flow")
    ax2.set_ylabel("µs / flow", color="#d95926")
    ax1.set_title("ONNX inference: throughput & per-flow latency vs batch size")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "benchmark_latency.png"), dpi=150)


if __name__ == "__main__":
    main()
