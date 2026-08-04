"""
multidataset.py — load and SFAF-align nine flow-based IDS datasets.

Each public dataset ships a different schema (CICFlowMeter, Zeek conn logs,
biflow, packet-stats). This module maps them all into the shared 12-feature
SFAF space and normalises their heterogeneous labels into one controlled
taxonomy, so a single model can train across them and be tested cross-dataset.

Feature-compatible datasets (map cleanly to the 12 flow features):
    cicids2017, unsw_nb15, ton_iot, bot_iot, cicddos2019,
    iotid20, x_iiotid, mqtt_iot_ids2020, cic_iot_2023(*)
    (*) CIC-IoT-2023 is packet-stat based (no fwd/bwd direction) — mapped
        best-effort; flagged lossy.

Deliberately EXCLUDED (different feature paradigm — would corrupt the space):
    edge_iiotset — raw Wireshark protocol fields, no flow aggregation
    n_baiot      — Kitsune per-packet damped-window statistics
    nsl_kdd      — legacy KDD connection features

Each loader returns a DataFrame with exactly UNIFIED_FEATURES + columns
['y' (0/1), 'category', 'dataset'].
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd

UNIFIED_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std",
]

# Canonical attack categories (the controlled cross-dataset taxonomy).
CATEGORIES = ["benign", "recon", "dos", "botnet", "bruteforce",
              "web", "exploit", "spoofing", "theft", "other_attack"]

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Datasets")
RS = 42


# ── taxonomy: normalise any source label string → canonical category ──────────
def to_category(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("benign", "normal", "background", "0", "legitimate", "normaltraffic"):
        return "benign"
    def has(*keys): return any(k in s for k in keys)
    if has("mirai", "bashlite", "gafgyt", "okiru", "torii", "bot", "c2", "mozi"):
        return "botnet"
    if has("scan", "recon", "portscan", "fingerprint", "port_os", "vulnerability"):
        return "recon"
    if has("bruteforce", "brute-force", "brute force", "patator", "password",
           "sparta", "hydra", "dictionary"):
        return "bruteforce"
    if has("ddos", "dos", "flood", "rdos", "hulk", "goldeneye", "slowloris",
           "slowhttp", "syn", "udplag"):
        return "dos"
    if has("sql", "xss", "web", "injection"):
        return "web"
    if has("mitm", "arp", "spoof"):
        return "spoofing"
    if has("theft", "exfiltrat", "keylog", "data_leak", "ransom"):
        return "theft"
    if has("exploit", "backdoor", "shellcode", "worm", "weaponization",
           "lateral", "rce", "command", "infiltration", "heartbleed",
           "fuzzers", "analysis", "generic", "crypto-ransomware"):
        return "exploit"
    return "other_attack"


def _finish(df, feat_src, y, cat, dataset):
    """Assemble the aligned frame from a source df + a feature-source dict."""
    out = pd.DataFrame(index=df.index)
    for canon, src in feat_src.items():
        if src is None or src not in df.columns:
            out[canon] = 0.0
        else:
            out[canon] = pd.to_numeric(df[src], errors="coerce")
    out = out[UNIFIED_FEATURES]
    out["y"] = np.asarray(y, dtype=int)
    out["category"] = list(cat)
    out["dataset"] = dataset
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=UNIFIED_FEATURES)
    return out.reset_index(drop=True)


# ── per-dataset loaders ───────────────────────────────────────────────────────
def load_cicids2017(root):
    files = sorted(glob.glob(os.path.join(root, "MachineLearningCVE", "*.csv")))
    dfs = []
    for f in files:
        t = pd.read_csv(f, low_memory=False); t.columns = t.columns.str.strip()
        dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    lab = df["Label"].astype(str).str.strip()
    y = (lab.str.upper() != "BENIGN").astype(int)
    cat = lab.map(to_category)
    fmap = {c: c for c in UNIFIED_FEATURES}   # identity (canonical names)
    return _finish(df, fmap, y, cat, "cicids2017")


def load_unsw(root):
    df = pd.concat([
        pd.read_parquet(os.path.join(root, "UNSWNB15", "UNSW_NB15_training-set.parquet")),
        pd.read_parquet(os.path.join(root, "UNSWNB15", "UNSW_NB15_testing-set.parquet")),
    ], ignore_index=True)
    df.columns = df.columns.str.strip()
    y = df["label"].astype(int)
    cat = df.get("attack_cat", pd.Series(["Normal"] * len(df))).fillna("Normal").map(to_category)
    fmap = {"Flow Duration": "dur", "Total Fwd Packets": "spkts",
            "Total Backward Packets": "dpkts", "Total Length of Fwd Packets": "sbytes",
            "Total Length of Bwd Packets": "dbytes", "Flow Packets/s": "rate",
            "Fwd Packets/s": "sload", "Bwd Packets/s": "dload",
            "Min Packet Length": "smean", "Max Packet Length": "dmean",
            "Packet Length Mean": "sjit", "Packet Length Std": "djit"}
    return _finish(df, fmap, y, cat, "unsw_nb15")


def load_ton(root):
    df = pd.read_csv(os.path.join(root, "TONIoT", "train_test_network.csv"), low_memory=False)
    df.columns = df.columns.str.strip()
    y = df["label"].astype(int)
    cat = df.get("type", pd.Series(["normal"] * len(df))).map(to_category)
    fmap = {"Flow Duration": "duration", "Total Fwd Packets": "src_pkts",
            "Total Backward Packets": "dst_pkts", "Total Length of Fwd Packets": "src_bytes",
            "Total Length of Bwd Packets": "dst_bytes", "Flow Packets/s": "src_ip_bytes",
            "Fwd Packets/s": "missed_bytes", "Bwd Packets/s": "dst_ip_bytes",
            "Min Packet Length": "src_port", "Max Packet Length": "dst_port",
            "Packet Length Mean": "http_request_body_len",
            "Packet Length Std": "http_response_body_len"}
    return _finish(df, fmap, y, cat, "ton_iot")


def load_botiot(root, max_rows=400_000):
    files = sorted(glob.glob(os.path.join(root, "BotIoT", "*.csv")))
    dfs, n = [], 0
    per = max(max_rows // max(len(files), 1), 1) if max_rows else None
    for f in files:
        t = pd.read_csv(f, low_memory=False, nrows=per)
        t.columns = t.columns.str.strip(); dfs.append(t); n += len(t)
        if max_rows and n >= max_rows:
            break
    df = pd.concat(dfs, ignore_index=True)
    y = df["attack"].astype(int)
    cat = df.get("category", pd.Series(["Normal"] * len(df))).map(to_category)
    fmap = {"Flow Duration": "dur", "Total Fwd Packets": "spkts",
            "Total Backward Packets": "dpkts", "Total Length of Fwd Packets": "sbytes",
            "Total Length of Bwd Packets": "dbytes", "Flow Packets/s": "rate",
            "Fwd Packets/s": "srate", "Bwd Packets/s": "drate",
            "Min Packet Length": "min", "Max Packet Length": "max",
            "Packet Length Mean": "mean", "Packet Length Std": "stddev"}
    return _finish(df, fmap, y, cat, "bot_iot")


def load_cicddos2019(root):
    files = sorted(glob.glob(os.path.join(root, "CICDDoS2019", "*.parquet")))
    dfs = []
    for f in files:
        t = pd.read_parquet(f); t.columns = t.columns.str.strip(); dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    lab = df["Label"].astype(str).str.strip()
    y = (lab.str.upper() != "BENIGN").astype(int)
    cat = lab.map(to_category)
    fmap = {"Flow Duration": "Flow Duration", "Total Fwd Packets": "Total Fwd Packets",
            "Total Backward Packets": "Total Backward Packets",
            "Total Length of Fwd Packets": "Fwd Packets Length Total",
            "Total Length of Bwd Packets": "Bwd Packets Length Total",
            "Flow Packets/s": "Flow Packets/s", "Fwd Packets/s": "Fwd Packets/s",
            "Bwd Packets/s": "Bwd Packets/s",
            "Min Packet Length": "Fwd Packet Length Min",
            "Max Packet Length": "Fwd Packet Length Max",
            "Packet Length Mean": "Packet Length Mean",
            "Packet Length Std": "Packet Length Std"}
    return _finish(df, fmap, y, cat, "cicddos2019")


def load_iotid20(root):
    f = glob.glob(os.path.join(root, "IoTID20", "*.csv"))[0]
    df = pd.read_csv(f, low_memory=False); df.columns = df.columns.str.strip()
    y = (df["Label"].astype(str).str.strip().str.lower() != "normal").astype(int)
    cat = df.get("Cat", df.get("Sub_Cat")).map(to_category)
    fmap = {"Flow Duration": "Flow_Duration", "Total Fwd Packets": "Tot_Fwd_Pkts",
            "Total Backward Packets": "Tot_Bwd_Pkts",
            "Total Length of Fwd Packets": "TotLen_Fwd_Pkts",
            "Total Length of Bwd Packets": "TotLen_Bwd_Pkts",
            "Flow Packets/s": "Flow_Pkts/s", "Fwd Packets/s": "Fwd_Pkts/s",
            "Bwd Packets/s": "Bwd_Pkts/s", "Min Packet Length": "Fwd_Pkt_Len_Min",
            "Max Packet Length": "Fwd_Pkt_Len_Max",
            "Packet Length Mean": "Pkt_Len_Mean", "Packet Length Std": "Pkt_Len_Std"}
    return _finish(df, fmap, y, cat, "iotid20")


def load_xiiotid(root):
    f = glob.glob(os.path.join(root, "X-IIoTID", "*.csv"))[0]
    df = pd.read_csv(f, low_memory=False); df.columns = df.columns.str.strip()
    y = (df["class3"].astype(str).str.strip().str.lower() != "normal").astype(int)
    cat = df["class2"].map(to_category)
    fmap = {"Flow Duration": "Duration", "Total Fwd Packets": "Scr_pkts",
            "Total Backward Packets": "Des_pkts", "Total Length of Fwd Packets": "Scr_bytes",
            "Total Length of Bwd Packets": "Des_bytes", "Flow Packets/s": "Scr_ip_bytes",
            "Fwd Packets/s": "missed_bytes", "Bwd Packets/s": "Des_ip_bytes",
            "Min Packet Length": "Scr_port", "Max Packet Length": "Des_port",
            "Packet Length Mean": None, "Packet Length Std": None}
    return _finish(df, fmap, y, cat, "x_iiotid")


def load_mqtt(root):
    """Per-file labels: biflow_<type>.csv. is_attack column also present."""
    typ = {"biflow_normal": ("benign", 0), "biflow_mqtt_bruteforce": ("bruteforce", 1),
           "biflow_scan_A": ("recon", 1), "biflow_scan_sU": ("recon", 1),
           "biflow_sparta": ("bruteforce", 1)}
    dfs = []
    for f in sorted(glob.glob(os.path.join(root, "MQTT_IoT_IDS2020", "*.csv"))):
        key = os.path.splitext(os.path.basename(f))[0]
        cat0, y0 = typ.get(key, ("other_attack", 1))
        t = pd.read_csv(f, low_memory=False); t.columns = t.columns.str.strip()
        t["_cat"] = cat0
        t["_y"] = t["is_attack"] if "is_attack" in t.columns else y0
        dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    # biflow has no duration / flow-rate; approximate byte totals & sizes.
    fmap = {"Flow Duration": None, "Total Fwd Packets": "fwd_num_pkts",
            "Total Backward Packets": "bwd_num_pkts",
            "Total Length of Fwd Packets": "fwd_num_bytes",
            "Total Length of Bwd Packets": "bwd_num_bytes",
            "Flow Packets/s": None, "Fwd Packets/s": None, "Bwd Packets/s": None,
            "Min Packet Length": "fwd_min_pkt_len", "Max Packet Length": "fwd_max_pkt_len",
            "Packet Length Mean": "fwd_mean_pkt_len", "Packet Length Std": "fwd_std_pkt_len"}
    return _finish(df, fmap, df["_y"].astype(int), df["_cat"], "mqtt_iot_ids2020")


def load_ciciot2023(root):
    """Packet-stat based: per-file category + fine `label`. No fwd/bwd split."""
    fcat = {"benign": "benign", "ddos": "dos", "dos": "dos", "mirai": "botnet",
            "recon": "recon", "spoofing": "spoofing"}
    dfs = []
    for f in sorted(glob.glob(os.path.join(root, "CICIoT2023", "*.csv"))):
        key = os.path.splitext(os.path.basename(f))[0].lower()
        t = pd.read_csv(f, low_memory=False); t.columns = t.columns.str.strip()
        t["_filecat"] = fcat.get(key, "other_attack")
        dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    if "label" in df.columns:
        cat = df["label"].map(to_category)
        cat = cat.where(cat != "other_attack", df["_filecat"])
    else:
        cat = df["_filecat"]
    y = (cat != "benign").astype(int)
    fmap = {"Flow Duration": None, "Total Fwd Packets": "Number",
            "Total Backward Packets": None, "Total Length of Fwd Packets": "Tot sum",
            "Total Length of Bwd Packets": None, "Flow Packets/s": "Rate",
            "Fwd Packets/s": "Rate", "Bwd Packets/s": None,
            "Min Packet Length": "Min", "Max Packet Length": "Max",
            "Packet Length Mean": "AVG", "Packet Length Std": "Std"}
    return _finish(df, fmap, y, cat, "cic_iot_2023")


def _read_zeek(path, usecols=None):
    """Parse a Zeek TSV log (conn.log.labeled): '#fields' line gives columns,
    '#' lines are comments, tab-separated, '-' means empty."""
    fields = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
                break
    if not fields:
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", comment="#", names=fields,
                     na_values="-", low_memory=False)
    return df


def load_iot23(root, max_rows_per_file=60_000):
    """IoT-23 — real IoT malware Zeek conn.log.labeled files.
    The 'small' archive ships labeled connection logs (no pcaps needed).
    Activates once iot_23_datasets_small.tar.gz is extracted under Datasets/IoT23.
    """
    files = glob.glob(os.path.join(root, "IoT23", "**", "*conn.log.labeled"),
                      recursive=True)
    if not files:
        raise FileNotFoundError("IoT-23 conn.log.labeled files not found — "
                                "extract iot_23_datasets_small.tar.gz first")
    dfs = []
    for fp in files:
        t = _read_zeek(fp)
        if len(t):
            if max_rows_per_file and len(t) > max_rows_per_file:
                t = t.sample(max_rows_per_file, random_state=RS)
            dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    # label column: 'label' = Benign/Malicious; 'detailed-label' = attack type
    lab = df.get("label", pd.Series(["Benign"] * len(df))).astype(str).str.strip()
    y = (lab.str.lower() != "benign").astype(int)
    det = df.get("detailed-label", lab).astype(str)
    cat = det.where(det.str.lower() != "-", lab).map(to_category)
    fmap = {"Flow Duration": "duration", "Total Fwd Packets": "orig_pkts",
            "Total Backward Packets": "resp_pkts",
            "Total Length of Fwd Packets": "orig_bytes",
            "Total Length of Bwd Packets": "resp_bytes",
            "Flow Packets/s": "orig_ip_bytes", "Fwd Packets/s": "missed_bytes",
            "Bwd Packets/s": "resp_ip_bytes",
            "Min Packet Length": "id.orig_p", "Max Packet Length": "id.resp_p",
            "Packet Length Mean": None, "Packet Length Std": None}
    return _finish(df, fmap, y, cat, "iot_23")


def load_wustl(root):
    """WUSTL-IIoT-2021 — Argus industrial-control flow CSV."""
    f = os.path.join(root, "WUSTL_IIoT_2021", "wustl_iiot_2021.csv")
    df = pd.read_csv(f, low_memory=False); df.columns = df.columns.str.strip()
    y = df["Target"].astype(int)
    cat = df.get("Traffic", pd.Series(["normal"] * len(df))).map(to_category)
    fmap = {"Flow Duration": "Dur", "Total Fwd Packets": "SrcPkts",
            "Total Backward Packets": "DstPkts",
            "Total Length of Fwd Packets": "SrcBytes",
            "Total Length of Bwd Packets": "DstBytes", "Flow Packets/s": "Rate",
            "Fwd Packets/s": "SrcRate", "Bwd Packets/s": "DstRate",
            "Min Packet Length": "Min", "Max Packet Length": "Max",
            "Packet Length Mean": "Mean", "Packet Length Std": None}
    return _finish(df, fmap, y, cat, "wustl_iiot")


LOADERS = {
    "cicids2017": load_cicids2017, "unsw_nb15": load_unsw, "ton_iot": load_ton,
    "bot_iot": load_botiot, "cicddos2019": load_cicddos2019,
    "iotid20": load_iotid20, "x_iiotid": load_xiiotid,
    "mqtt_iot_ids2020": load_mqtt, "cic_iot_2023": load_ciciot2023,
    "wustl_iiot": load_wustl, "iot_23": load_iot23,
}
# Lossy (no direction / no duration) — usable but flagged.
LOSSY = {"mqtt_iot_ids2020", "cic_iot_2023"}


def load(name, root=DEFAULT_ROOT):
    return LOADERS[name](root)


def available(root=DEFAULT_ROOT):
    """Which datasets have their files present under root."""
    present = []
    for name in LOADERS:
        try:
            # cheap existence check via the loader's first glob/path
            if name == "cicids2017":
                ok = bool(glob.glob(os.path.join(root, "MachineLearningCVE", "*.csv")))
            elif name == "unsw_nb15":
                ok = os.path.exists(os.path.join(root, "UNSWNB15", "UNSW_NB15_training-set.parquet"))
            elif name == "ton_iot":
                ok = os.path.exists(os.path.join(root, "TONIoT", "train_test_network.csv"))
            elif name == "bot_iot":
                ok = bool(glob.glob(os.path.join(root, "BotIoT", "*.csv")))
            elif name == "cicddos2019":
                ok = bool(glob.glob(os.path.join(root, "CICDDoS2019", "*.parquet")))
            elif name == "iotid20":
                ok = bool(glob.glob(os.path.join(root, "IoTID20", "*.csv")))
            elif name == "x_iiotid":
                ok = bool(glob.glob(os.path.join(root, "X-IIoTID", "*.csv")))
            elif name == "mqtt_iot_ids2020":
                ok = bool(glob.glob(os.path.join(root, "MQTT_IoT_IDS2020", "*.csv")))
            elif name == "cic_iot_2023":
                ok = bool(glob.glob(os.path.join(root, "CICIoT2023", "*.csv")))
            elif name == "wustl_iiot":
                ok = os.path.exists(os.path.join(root, "WUSTL_IIoT_2021", "wustl_iiot_2021.csv"))
            elif name == "iot_23":
                ok = bool(glob.glob(os.path.join(root, "IoT23", "**", "*conn.log.labeled"),
                                    recursive=True))
            else:
                ok = False
        except Exception:
            ok = False
        if ok:
            present.append(name)
    return present


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    print("Available:", available(root))
    for name in available(root):
        try:
            df = load(name, root)
            lossy = " (lossy)" if name in LOSSY else ""
            print(f"  {name:18} rows={len(df):>8,}  attack%={df['y'].mean()*100:5.1f}  "
                  f"cats={sorted(df['category'].unique())}{lossy}")
        except Exception as e:
            print(f"  {name:18} ERROR {e}")
