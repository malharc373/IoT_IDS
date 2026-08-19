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
import glob
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

PASS, FAIL = [], []
TMP = tempfile.mkdtemp(prefix="iotids_test_")


def check(name, fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        PASS.append(name)
        print(f"  \033[32mPASS\033[0m  {name}")
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
    import numpy, pandas, sklearn, xgboost, lightgbm  # noqa
    import onnx, onnxruntime, skl2onnx, onnxmltools, joblib, scapy  # noqa


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
    assert len(recs) == 2
    # FlowTable -> 21-dim vector
    table = ff.FlowTable()
    for i, (ts, raw) in enumerate(recs):
        table.add_packet(ff.parse_raw(raw), ts + i)
    rows = table.extract(min_pkts=1)
    assert rows and len(rows[0][1]) == ff.N_FEATURES


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


def t_ips_responder():
    from ips_response import Responder
    r = Responder(mode="dry-run", min_conf=0.9, block_seconds=60,
                  allowlist=["10.0.0.0/8"],
                  state_path=os.path.join(TMP, "ips_state.json"))
    assert r.handle("203.0.113.9", "synflood", 0.99)["action"] == "would-block"
    assert r.handle("203.0.113.9", "synflood", 0.99)["action"] == "already-blocked"
    assert r.handle("10.1.2.3", "portscan", 0.99)["action"] == "skip"
    assert r.handle("198.51.100.4", "portscan", 0.5)["action"] == "monitor"


def t_c_export():
    import importlib.util, subprocess, shutil
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


def t_dashboard():
    import importlib.util, urllib.request, threading, time
    from http.server import ThreadingHTTPServer
    spec = importlib.util.spec_from_file_location("dash", os.path.join(ROOT, "src", "dashboard.py"))
    dash = importlib.util.module_from_spec(spec); spec.loader.exec_module(dash)
    # write a couple of alert records and aggregate them
    logp = os.path.join(TMP, "alerts.jsonl")
    with open(logp, "w") as f:
        f.write(json.dumps({"ts": "2026-08-03T10:00:00", "src_ip": "203.0.113.1",
                            "kind": "portscan", "flows": 500, "confidence": 1.0}) + "\n")
        f.write(json.dumps({"ts": "2026-08-03T10:00:01", "src_ip": "203.0.113.2",
                            "kind": "synflood", "flows": 900, "confidence": 1.0}) + "\n")
    st = dash.build_state(logp)
    assert st["totals"]["incidents"] == 2 and st["totals"]["sources"] == 2
    assert {d["type"] for d in st["by_type"]} == {"portscan", "synflood"}
    # serve it and fetch both endpoints
    dash.Handler.log_path = logp
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3).read()
        api = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3).read()
        assert b"IoT-IDS" in page and json.loads(api)["totals"]["incidents"] == 2
    finally:
        srv.shutdown()


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
    assert len(mds.UNIFIED_FEATURES) == 12
    # if the datasets happen to be present, one loader must align to 12 features
    present = mds.available()
    if present:
        df = mds.load(present[0])
        assert list(df.columns[:12]) == mds.UNIFIED_FEATURES
        assert set(df["y"].unique()) <= {0, 1}


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


def t_daemon_help():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "ids_daemon.py"), "--help"],
                       capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0 and "--replay" in r.stdout


# ── 7. SFAF reproduction pipeline (no datasets) ───────────────────────────────
def t_sfaf_trainer_guard():
    # When datasets are absent, load_datasets() must fail cleanly (SystemExit).
    # When present (e.g. via the Datasets symlink), it must load without error.
    import importlib.util, glob
    spec = importlib.util.spec_from_file_location(
        "m02", os.path.join(ROOT, "code", "02_train_sfaf.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    has_data = bool(glob.glob(os.path.join(ROOT, "Datasets", "MachineLearningCVE", "*.csv")))
    try:
        m.load_datasets()
        if not has_data:
            raise AssertionError("expected SystemExit without datasets")
    except SystemExit:
        if has_data:
            raise AssertionError("unexpected SystemExit though datasets present")


def t_sfaf_onnx_export():
    import importlib.util, numpy as np
    spec = importlib.util.spec_from_file_location(
        "m02b", os.path.join(ROOT, "code", "02_train_sfaf.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    import onnxruntime as rt
    rng = np.random.RandomState(0)
    X = rng.rand(1500, 12).astype(np.float32)
    y = (X[:, 5] + X[:, 2] > 1.0).astype(int)
    sc = StandardScaler().fit(X)
    edge = XGBClassifier(n_estimators=20, max_depth=4, tree_method="hist",
                         eval_metric="logloss", random_state=42).fit(sc.transform(X), y)
    path = os.path.join(TMP, "edge.onnx")
    m.export_edge_onnx(sc, edge, path)
    sess = rt.InferenceSession(path)
    onx = sess.run(None, {"input": X[:300]})[0].ravel()
    ref = Pipeline([("sc", sc), ("clf", edge)]).predict(X[:300])
    assert (onx == ref).all()


def t_download_datasets_runs():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "code", "download_datasets.py")],
                       capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0 and "layout" in r.stdout.lower()


# ── 8. shell scripts syntax ───────────────────────────────────────────────────
def t_shell_syntax():
    for sh in ["demo/run_demo.sh", "deploy/setup_pi.sh"]:
        r = subprocess.run(["bash", "-n", os.path.join(ROOT, sh)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{sh}: {r.stderr}"


def t_benchmark_params():
    # the benchmark's param/footprint section must run (fast, no full sweep)
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", os.path.join(ROOT, "demo", "benchmark.py"))
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    meta, sizes, n_trees, n_nodes = b.bench_params()
    assert n_trees > 0 and n_nodes > 0
    assert "live_ids.onnx" in sizes and sizes["live_ids.onnx"] > 0


def t_systemd_unit_sane():
    txt = open(os.path.join(ROOT, "deploy", "iot-ids.service")).read()
    assert "[Service]" in txt and "ids_daemon.py" in txt and "--iface" in txt
    dash = open(os.path.join(ROOT, "deploy", "iot-ids-dashboard.service")).read()
    assert "[Service]" in dash and "dashboard.py" in dash
    # setup script installs and enables both services
    setup = open(os.path.join(ROOT, "deploy", "setup_pi.sh")).read()
    assert "iot-ids-dashboard.service" in setup and "enable iot-ids.service iot-ids-dashboard.service" in setup


def main():
    print(f"Running smoke tests (tmp={TMP})\n")
    tests = [
        ("imports", t_imports),
        ("requirements importable", t_requirements_importable),
        ("flow_features core", t_flow_features_core),
        ("flow_features CLI", t_flow_features_cli),
        ("flowtable window + dirty", t_flowtable_window_and_dirty),
        ("flowtable thread safety", t_flowtable_thread_safety),
        ("verdict cache == full scoring", t_verdict_cache_matches_full_scoring),
        ("traffic_gen all kinds", t_traffic_gen_all_kinds),
        ("traffic_gen CLI", t_traffic_gen_cli),
        ("build_corpus helper", t_build_corpus_helper),
        ("model artifacts", t_model_artifacts),
        ("ips responder", t_ips_responder),
        ("c export + compile", t_c_export),
        ("SFAF alignment contract", t_alignment_contract),
        ("multidataset taxonomy", t_multidataset_taxonomy),
        ("trivial-baseline guard", t_trivial_baseline),
        ("web dashboard", t_dashboard),
        ("detector classify known", t_detector_classify_known),
        ("daemon offline mode", t_daemon_offline),
        ("daemon replay mode", t_daemon_replay),
        ("daemon live path (scapy)", t_live_path),
        ("daemon --help", t_daemon_help),
        ("SFAF trainer guard", t_sfaf_trainer_guard),
        ("SFAF onnx export", t_sfaf_onnx_export),
        ("download_datasets runs", t_download_datasets_runs),
        ("shell script syntax", t_shell_syntax),
        ("benchmark params", t_benchmark_params),
        ("systemd unit sane", t_systemd_unit_sane),
    ]
    for name, fn in tests:
        check(name, fn)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'='*50}\n  {len(PASS)} passed, {len(FAIL)} failed\n{'='*50}")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
