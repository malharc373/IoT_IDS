"""
multidataset.py — load and SFAF-align eleven flow-based IDS datasets.

Each public dataset ships a different schema (CICFlowMeter, Zeek conn logs,
Argus biflow, packet-stats). This module maps them all into the shared
12-feature SFAF space and normalises their heterogeneous labels into one
controlled taxonomy, so a single model can train across them and be tested
cross-dataset.

ALIGNMENT CONTRACT (see vault/Reference/Feature Spaces.md)
----------------------------------------------------------
The unified space is defined by *semantics and units*, not by column name:

    Flow Duration                 SECONDS          (not microseconds)
    Total Fwd/Backward Packets    packet counts
    Total Length of Fwd/Bwd Pkts  bytes
    Flow/Fwd/Bwd Packets/s        packets per second (NOT bits/s, NOT byte counts)
    Min/Max/Mean/Std Packet Len   bytes

Three rules follow from that contract:

  1. **Convert units.** CICFlowMeter datasets report Flow Duration in
     microseconds; Zeek/Argus datasets report it in seconds. Everything is
     converted to seconds here.
  2. **Derive, don't substitute.** If a dataset lacks a per-direction packet
     rate but has packets and duration, the rate is *computed*. A semantically
     different column is never dropped into the slot.
  3. **Absent means NaN, never 0.** A feature a dataset genuinely cannot supply
     is left NaN. XGBoost handles missing values natively; zero-filling creates
     a constant column that acts as a perfect dataset fingerprint and leaks
     domain identity into any pooled model.

The pre-2026-08 version of this module violated all three (port numbers mapped
into packet-length slots, byte counts into rate slots, µs vs s unmixed,
zero-fill for absent features). See vault/Findings/F01 and F12.

Feature-compatible datasets:
    cicids2017, unsw_nb15, ton_iot, bot_iot, cicddos2019, iotid20,
    x_iiotid, mqtt_iot_ids2020, cic_iot_2023, wustl_iiot, iot_23

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

# The unit each canonical feature is defined in. Loaders MUST convert to these.
FEATURE_UNITS = {
    "Flow Duration": "s",
    "Total Fwd Packets": "packets",
    "Total Backward Packets": "packets",
    "Total Length of Fwd Packets": "bytes",
    "Total Length of Bwd Packets": "bytes",
    "Flow Packets/s": "packets/s",
    "Fwd Packets/s": "packets/s",
    "Bwd Packets/s": "packets/s",
    "Min Packet Length": "bytes",
    "Max Packet Length": "bytes",
    "Packet Length Mean": "bytes",
    "Packet Length Std": "bytes",
}

# Canonical attack categories (the controlled cross-dataset taxonomy).
CATEGORIES = ["benign", "recon", "dos", "botnet", "bruteforce",
              "web", "exploit", "spoofing", "theft", "other_attack"]

# Datasets root: env override (IOTIDS_DATASETS_ROOT) else the repo's ./Datasets
# (which may be a symlink to an external drive). See .env.example.
DEFAULT_ROOT = os.environ.get(
    "IOTIDS_DATASETS_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Datasets"))
RS = 42

US_PER_S = 1_000_000.0


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


# ── alignment helpers ─────────────────────────────────────────────────────────
def _num(df, col):
    """Numeric view of a source column, or an all-NaN series if absent."""
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _rate(count, duration):
    """packets/s from a count and a duration in seconds.

    Zero/absent duration yields NaN rather than inf — a single-packet flow of
    unknown length has an *unknown* rate, not an infinite one.
    """
    d = duration.where(duration > 0)
    return count / d


def _pooled_std(n_a, mean_a, std_a, n_b, mean_b, std_b):
    """Combined std of two groups from their counts, means and stds.

    Var_total = (Σ n_i·(σ_i² + μ_i²)) / Σ n_i  −  μ_total²
    """
    n = n_a.fillna(0) + n_b.fillna(0)
    m = (n_a.fillna(0) * mean_a.fillna(0) + n_b.fillna(0) * mean_b.fillna(0))
    m = m / n.where(n > 0)
    ss = (n_a.fillna(0) * (std_a.fillna(0) ** 2 + mean_a.fillna(0) ** 2)
          + n_b.fillna(0) * (std_b.fillna(0) ** 2 + mean_b.fillna(0) ** 2))
    var = ss / n.where(n > 0) - m ** 2
    return np.sqrt(var.clip(lower=0))


def _finish(df, feat_src, y, cat, dataset):
    """Assemble the aligned frame.

    `feat_src` maps each canonical feature to one of:
        str            — a source column name (coerced to numeric)
        pandas.Series  — a derived value, already in canonical units
        None           — the dataset genuinely cannot supply this feature;
                         the column is emitted as NaN and rows are NOT dropped
                         because of it (see the alignment contract above).
    """
    out = pd.DataFrame(index=df.index)
    supplied = []
    for canon in UNIFIED_FEATURES:
        src = feat_src.get(canon)
        if src is None:
            out[canon] = np.nan
        elif isinstance(src, pd.Series):
            out[canon] = pd.to_numeric(src, errors="coerce").astype("float64")
            supplied.append(canon)
        else:
            out[canon] = _num(df, src)
            supplied.append(canon)
    out = out[UNIFIED_FEATURES]
    out["y"] = np.asarray(y, dtype=int)
    out["category"] = list(cat)
    out["dataset"] = dataset
    # inf is a parse artifact (a rate over zero duration); NaN is the honest value
    out = out.replace([np.inf, -np.inf], np.nan)
    # drop rows only for features this dataset claims to supply
    if supplied:
        out = out.dropna(subset=supplied)
    return out.reset_index(drop=True)


def coverage(name):
    """Which canonical features a dataset structurally supplies (True) or
    cannot (False). Cheap, data-free — used for reporting and sanity checks."""
    return _COVERAGE.get(name, {})


# ── per-dataset loaders ───────────────────────────────────────────────────────
def load_cicids2017(root):
    """CICIDS2017 — CICFlowMeter. Flow Duration is in MICROSECONDS."""
    files = sorted(glob.glob(os.path.join(root, "MachineLearningCVE", "*.csv")))
    dfs = []
    for f in files:
        t = pd.read_csv(f, low_memory=False); t.columns = t.columns.str.strip()
        dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    lab = df["Label"].astype(str).str.strip()
    y = (lab.str.upper() != "BENIGN").astype(int)
    cat = lab.map(to_category)
    fmap = {
        "Flow Duration": _num(df, "Flow Duration") / US_PER_S,     # µs → s
        "Total Fwd Packets": "Total Fwd Packets",
        "Total Backward Packets": "Total Backward Packets",
        "Total Length of Fwd Packets": "Total Length of Fwd Packets",
        "Total Length of Bwd Packets": "Total Length of Bwd Packets",
        "Flow Packets/s": "Flow Packets/s",
        "Fwd Packets/s": "Fwd Packets/s",
        "Bwd Packets/s": "Bwd Packets/s",
        "Min Packet Length": "Min Packet Length",
        "Max Packet Length": "Max Packet Length",
        "Packet Length Mean": "Packet Length Mean",
        "Packet Length Std": "Packet Length Std",
    }
    return _finish(df, fmap, y, cat, "cicids2017")


def load_unsw(root):
    """UNSW-NB15 — Argus/Bro derived. `dur` is in SECONDS.

    Notes on what is NOT used:
      * `sload`/`dload` are **bits per second**, not packets/s — the directional
        packet rates are derived from spkts/dpkts and dur instead.
      * `sjit`/`djit` are jitter (ms), not packet lengths.
      * `smean`/`dmean` are per-direction MEAN packet sizes, so a flow-level
        packet-length mean is derivable, but min/max/std are not.
    """
    df = pd.concat([
        pd.read_parquet(os.path.join(root, "UNSWNB15", "UNSW_NB15_training-set.parquet")),
        pd.read_parquet(os.path.join(root, "UNSWNB15", "UNSW_NB15_testing-set.parquet")),
    ], ignore_index=True)
    df.columns = df.columns.str.strip()
    y = df["label"].astype(int)
    cat = df.get("attack_cat", pd.Series(["Normal"] * len(df))).fillna("Normal").map(to_category)

    dur = _num(df, "dur")
    spkts, dpkts = _num(df, "spkts"), _num(df, "dpkts")
    sbytes, dbytes = _num(df, "sbytes"), _num(df, "dbytes")
    tot_pkts = spkts.fillna(0) + dpkts.fillna(0)
    fmap = {
        "Flow Duration": dur,
        "Total Fwd Packets": "spkts",
        "Total Backward Packets": "dpkts",
        "Total Length of Fwd Packets": "sbytes",
        "Total Length of Bwd Packets": "dbytes",
        "Flow Packets/s": "rate",                       # already packets/s
        "Fwd Packets/s": _rate(spkts, dur),
        "Bwd Packets/s": _rate(dpkts, dur),
        "Min Packet Length": None,                      # not recoverable
        "Max Packet Length": None,                      # not recoverable
        "Packet Length Mean": (sbytes.fillna(0) + dbytes.fillna(0))
                              / tot_pkts.where(tot_pkts > 0),
        "Packet Length Std": None,                      # not recoverable
    }
    return _finish(df, fmap, y, cat, "unsw_nb15")


def load_ton(root):
    """TON-IoT — Zeek conn log. `duration` is in SECONDS.

    Zeek reports no packet-length statistics. A flow-level mean is derivable
    from the IP-byte totals and packet counts; min/max/std are not.
    """
    df = pd.read_csv(os.path.join(root, "TONIoT", "train_test_network.csv"), low_memory=False)
    df.columns = df.columns.str.strip()
    y = df["label"].astype(int)
    cat = df.get("type", pd.Series(["normal"] * len(df))).map(to_category)

    dur = _num(df, "duration")
    spkts, dpkts = _num(df, "src_pkts"), _num(df, "dst_pkts")
    sip_b, dip_b = _num(df, "src_ip_bytes"), _num(df, "dst_ip_bytes")
    tot_pkts = spkts.fillna(0) + dpkts.fillna(0)
    fmap = {
        "Flow Duration": dur,
        "Total Fwd Packets": "src_pkts",
        "Total Backward Packets": "dst_pkts",
        "Total Length of Fwd Packets": "src_bytes",
        "Total Length of Bwd Packets": "dst_bytes",
        "Flow Packets/s": _rate(tot_pkts, dur),
        "Fwd Packets/s": _rate(spkts, dur),
        "Bwd Packets/s": _rate(dpkts, dur),
        "Min Packet Length": None,
        "Max Packet Length": None,
        "Packet Length Mean": (sip_b.fillna(0) + dip_b.fillna(0))
                              / tot_pkts.where(tot_pkts > 0),
        "Packet Length Std": None,
    }
    return _finish(df, fmap, y, cat, "ton_iot")


def load_botiot(root, max_rows=400_000):
    """Bot-IoT — Argus. `dur` in SECONDS; min/max/mean/stddev are packet sizes.

    Rows are sampled uniformly per file. Bot-IoT CSVs are ordered by attack, so
    a `nrows=` head-read (the pre-2026-08 behaviour) returns a biased slice.
    """
    files = sorted(glob.glob(os.path.join(root, "BotIoT", "*.csv")))
    per = max(max_rows // max(len(files), 1), 1) if max_rows else None
    dfs = []
    for f in files:
        t = pd.read_csv(f, low_memory=False)
        t.columns = t.columns.str.strip()
        if per and len(t) > per:
            t = t.sample(per, random_state=RS)
        dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    y = df["attack"].astype(int)
    cat = df.get("category", pd.Series(["Normal"] * len(df))).map(to_category)
    fmap = {
        "Flow Duration": "dur",
        "Total Fwd Packets": "spkts",
        "Total Backward Packets": "dpkts",
        "Total Length of Fwd Packets": "sbytes",
        "Total Length of Bwd Packets": "dbytes",
        "Flow Packets/s": "rate",
        "Fwd Packets/s": "srate",
        "Bwd Packets/s": "drate",
        "Min Packet Length": "min",
        "Max Packet Length": "max",
        "Packet Length Mean": "mean",
        "Packet Length Std": "stddev",
    }
    return _finish(df, fmap, y, cat, "bot_iot")


def load_cicddos2019(root):
    """CICDDoS2019 — CICFlowMeter v3. Flow Duration in MICROSECONDS.

    This schema has true flow-level `Packet Length Min/Max` columns; the
    pre-2026-08 map used the forward-only `Fwd Packet Length Min/Max`.
    """
    files = sorted(glob.glob(os.path.join(root, "CICDDoS2019", "*.parquet")))
    dfs = []
    for f in files:
        t = pd.read_parquet(f); t.columns = t.columns.str.strip(); dfs.append(t)
    df = pd.concat(dfs, ignore_index=True)
    lab = df["Label"].astype(str).str.strip()
    y = (lab.str.upper() != "BENIGN").astype(int)
    cat = lab.map(to_category)
    fmap = {
        "Flow Duration": _num(df, "Flow Duration") / US_PER_S,     # µs → s
        "Total Fwd Packets": "Total Fwd Packets",
        "Total Backward Packets": "Total Backward Packets",
        "Total Length of Fwd Packets": "Fwd Packets Length Total",
        "Total Length of Bwd Packets": "Bwd Packets Length Total",
        "Flow Packets/s": "Flow Packets/s",
        "Fwd Packets/s": "Fwd Packets/s",
        "Bwd Packets/s": "Bwd Packets/s",
        "Min Packet Length": "Packet Length Min",
        "Max Packet Length": "Packet Length Max",
        "Packet Length Mean": "Packet Length Mean",
        "Packet Length Std": "Packet Length Std",
    }
    return _finish(df, fmap, y, cat, "cicddos2019")


def load_iotid20(root):
    """IoTID20 — CICFlowMeter v3 naming. Flow_Duration in MICROSECONDS.

    Has true flow-level `Pkt_Len_Min/Max`; the pre-2026-08 map used the
    forward-only `Fwd_Pkt_Len_Min/Max`.
    """
    f = glob.glob(os.path.join(root, "IoTID20", "*.csv"))[0]
    df = pd.read_csv(f, low_memory=False); df.columns = df.columns.str.strip()
    y = (df["Label"].astype(str).str.strip().str.lower() != "normal").astype(int)
    cat = df.get("Cat", df.get("Sub_Cat")).map(to_category)
    fmap = {
        "Flow Duration": _num(df, "Flow_Duration") / US_PER_S,     # µs → s
        "Total Fwd Packets": "Tot_Fwd_Pkts",
        "Total Backward Packets": "Tot_Bwd_Pkts",
        "Total Length of Fwd Packets": "TotLen_Fwd_Pkts",
        "Total Length of Bwd Packets": "TotLen_Bwd_Pkts",
        "Flow Packets/s": "Flow_Pkts/s",
        "Fwd Packets/s": "Fwd_Pkts/s",
        "Bwd Packets/s": "Bwd_Pkts/s",
        "Min Packet Length": "Pkt_Len_Min",
        "Max Packet Length": "Pkt_Len_Max",
        "Packet Length Mean": "Pkt_Len_Mean",
        "Packet Length Std": "Pkt_Len_Std",
    }
    return _finish(df, fmap, y, cat, "iotid20")


def load_xiiotid(root):
    """X-IIoTID — Zeek-derived. `Duration` in SECONDS; `paket_rate` is pkts/s.

    Packet-length min/max/std are not reported; the mean is derivable from
    total_bytes / total_packet.
    """
    f = glob.glob(os.path.join(root, "X-IIoTID", "*.csv"))[0]
    df = pd.read_csv(f, low_memory=False); df.columns = df.columns.str.strip()
    y = (df["class3"].astype(str).str.strip().str.lower() != "normal").astype(int)
    cat = df["class2"].map(to_category)

    dur = _num(df, "Duration")
    spkts, dpkts = _num(df, "Scr_pkts"), _num(df, "Des_pkts")
    tot_b, tot_p = _num(df, "total_bytes"), _num(df, "total_packet")
    fmap = {
        "Flow Duration": dur,
        "Total Fwd Packets": "Scr_pkts",
        "Total Backward Packets": "Des_pkts",
        "Total Length of Fwd Packets": "Scr_bytes",
        "Total Length of Bwd Packets": "Des_bytes",
        "Flow Packets/s": "paket_rate",
        "Fwd Packets/s": _rate(spkts, dur),
        "Bwd Packets/s": _rate(dpkts, dur),
        "Min Packet Length": None,
        "Max Packet Length": None,
        "Packet Length Mean": tot_b / tot_p.where(tot_p > 0),
        "Packet Length Std": None,
    }
    return _finish(df, fmap, y, cat, "x_iiotid")


def load_mqtt(root):
    """MQTT-IoT-IDS2020 — biflow. Per-file labels: biflow_<type>.csv.

    No duration column, but per-direction mean IAT and packet counts are
    present, so flow duration is *approximated* as the longer of the two
    directional spans, (n-1)·mean_iat. Packet-length statistics are reported
    per direction and combined here (min of mins, max of maxes, count-weighted
    mean, pooled std) rather than taking the forward direction only.
    """
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

    fp, bp = _num(df, "fwd_num_pkts"), _num(df, "bwd_num_pkts")
    f_iat, b_iat = _num(df, "fwd_mean_iat"), _num(df, "bwd_mean_iat")
    # (n-1) inter-arrival gaps of mean length approximate the directional span
    f_span = (fp - 1).clip(lower=0) * f_iat
    b_span = (bp - 1).clip(lower=0) * b_iat
    dur = pd.concat([f_span, b_span], axis=1).max(axis=1)
    tot_p = fp.fillna(0) + bp.fillna(0)

    f_mean, b_mean = _num(df, "fwd_mean_pkt_len"), _num(df, "bwd_mean_pkt_len")
    f_std, b_std = _num(df, "fwd_std_pkt_len"), _num(df, "bwd_std_pkt_len")
    fmap = {
        "Flow Duration": dur,
        "Total Fwd Packets": "fwd_num_pkts",
        "Total Backward Packets": "bwd_num_pkts",
        "Total Length of Fwd Packets": "fwd_num_bytes",
        "Total Length of Bwd Packets": "bwd_num_bytes",
        "Flow Packets/s": _rate(tot_p, dur),
        "Fwd Packets/s": _rate(fp, dur),
        "Bwd Packets/s": _rate(bp, dur),
        "Min Packet Length": pd.concat(
            [_num(df, "fwd_min_pkt_len"), _num(df, "bwd_min_pkt_len")], axis=1).min(axis=1),
        "Max Packet Length": pd.concat(
            [_num(df, "fwd_max_pkt_len"), _num(df, "bwd_max_pkt_len")], axis=1).max(axis=1),
        "Packet Length Mean": (fp.fillna(0) * f_mean.fillna(0)
                               + bp.fillna(0) * b_mean.fillna(0)) / tot_p.where(tot_p > 0),
        "Packet Length Std": _pooled_std(fp, f_mean, f_std, bp, b_mean, b_std),
    }
    return _finish(df, fmap, df["_y"].astype(int), df["_cat"], "mqtt_iot_ids2020")


def load_ciciot2023(root):
    """CIC-IoT-2023 — packet-stat windows, NOT biflows.

    There is no forward/backward split and no duration column. `Number` is the
    packet count in the window and `Rate` its packet rate, so a window span is
    derivable as Number/Rate. Every directional feature is genuinely absent and
    is emitted as NaN — this dataset is structurally lossy against the flow
    space and is flagged in LOSSY.
    """
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

    number, rate = _num(df, "Number"), _num(df, "Rate")
    fmap = {
        "Flow Duration": number / rate.where(rate > 0),
        "Total Fwd Packets": "Number",
        "Total Backward Packets": None,          # no direction in this schema
        "Total Length of Fwd Packets": "Tot sum",
        "Total Length of Bwd Packets": None,
        "Flow Packets/s": "Rate",
        "Fwd Packets/s": None,
        "Bwd Packets/s": None,
        "Min Packet Length": "Min",
        "Max Packet Length": "Max",
        "Packet Length Mean": "AVG",
        "Packet Length Std": "Std",
    }
    return _finish(df, fmap, y, cat, "cic_iot_2023")


def _read_zeek(path, usecols=None):
    """Parse a Zeek TSV log (conn.log.labeled): '#fields' line gives columns,
    '#' lines are comments, tab-separated, '-' means empty.

    IoT-23 quirk: its labelled conn logs separate the final
    `tunnel_parents / label / detailed-label` triple with **spaces** rather
    than tabs, in both the #fields header and the data rows:

        #fields  ts  uid  ...  resp_ip_bytes  tunnel_parents   label   detailed-label
        ...      ...          0              (empty)   Malicious   PartOfAHorizontalPortScan

    Splitting on tabs alone therefore yields one column literally named
    "tunnel_parents   label   detailed-label", no `label` column at all, and a
    loader that silently reads every row as benign — which is how IoT-23, a
    malware-capture dataset, came out 0.0% attack. Any column name containing
    whitespace is treated as a composite and expanded.
    """
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
    for col in list(df.columns):
        parts = col.split()
        if len(parts) > 1:
            sub = df[col].astype(str).str.split(r"\s+", expand=True)
            for j, name in enumerate(parts):
                df[name] = sub[j] if j < sub.shape[1] else np.nan
            df = df.drop(columns=[col])
    return df


def load_iot23(root, max_rows_per_file=60_000):
    """IoT-23 — real IoT malware Zeek conn.log.labeled files. `duration` in SECONDS.

    Same Zeek limitation as TON-IoT: no packet-length statistics beyond a
    derivable mean.
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
    lab = df.get("label", pd.Series(["Benign"] * len(df))).astype(str).str.strip()
    y = (lab.str.lower() != "benign").astype(int)
    det = df.get("detailed-label", lab).astype(str)
    cat = det.where(det.str.lower() != "-", lab).map(to_category)

    dur = _num(df, "duration")
    opkts, rpkts = _num(df, "orig_pkts"), _num(df, "resp_pkts")
    oip_b, rip_b = _num(df, "orig_ip_bytes"), _num(df, "resp_ip_bytes")
    tot_p = opkts.fillna(0) + rpkts.fillna(0)
    fmap = {
        "Flow Duration": dur,
        "Total Fwd Packets": "orig_pkts",
        "Total Backward Packets": "resp_pkts",
        "Total Length of Fwd Packets": "orig_bytes",
        "Total Length of Bwd Packets": "resp_bytes",
        "Flow Packets/s": _rate(tot_p, dur),
        "Fwd Packets/s": _rate(opkts, dur),
        "Bwd Packets/s": _rate(rpkts, dur),
        "Min Packet Length": None,
        "Max Packet Length": None,
        "Packet Length Mean": (oip_b.fillna(0) + rip_b.fillna(0))
                              / tot_p.where(tot_p > 0),
        "Packet Length Std": None,
    }
    return _finish(df, fmap, y, cat, "iot_23")


def load_wustl(root):
    """WUSTL-IIoT-2021 — Argus industrial-control flow CSV. `Dur` in SECONDS.

    Argus supplies Min/Max/Mean packet sizes but no packet-length std.
    """
    f = os.path.join(root, "WUSTL_IIoT_2021", "wustl_iiot_2021.csv")
    df = pd.read_csv(f, low_memory=False); df.columns = df.columns.str.strip()
    y = df["Target"].astype(int)
    cat = df.get("Traffic", pd.Series(["normal"] * len(df))).map(to_category)
    fmap = {
        "Flow Duration": "Dur",
        "Total Fwd Packets": "SrcPkts",
        "Total Backward Packets": "DstPkts",
        "Total Length of Fwd Packets": "SrcBytes",
        "Total Length of Bwd Packets": "DstBytes",
        "Flow Packets/s": "Rate",
        "Fwd Packets/s": "SrcRate",
        "Bwd Packets/s": "DstRate",
        "Min Packet Length": "Min",
        "Max Packet Length": "Max",
        "Packet Length Mean": "Mean",
        "Packet Length Std": None,
    }
    return _finish(df, fmap, y, cat, "wustl_iiot")


LOADERS = {
    "cicids2017": load_cicids2017, "unsw_nb15": load_unsw, "ton_iot": load_ton,
    "bot_iot": load_botiot, "cicddos2019": load_cicddos2019,
    "iotid20": load_iotid20, "x_iiotid": load_xiiotid,
    "mqtt_iot_ids2020": load_mqtt, "cic_iot_2023": load_ciciot2023,
    "wustl_iiot": load_wustl, "iot_23": load_iot23,
}

# Which canonical features each dataset can structurally supply. Anything False
# is emitted as NaN by the loader, never zero-filled.
_ALL = {f: True for f in UNIFIED_FEATURES}
def _minus(*absent):
    d = dict(_ALL)
    for a in absent:
        d[a] = False
    return d

_COVERAGE = {
    "cicids2017": dict(_ALL),
    "cicddos2019": dict(_ALL),
    "iotid20": dict(_ALL),
    "bot_iot": dict(_ALL),
    "mqtt_iot_ids2020": dict(_ALL),
    "unsw_nb15": _minus("Min Packet Length", "Max Packet Length", "Packet Length Std"),
    "ton_iot": _minus("Min Packet Length", "Max Packet Length", "Packet Length Std"),
    "iot_23": _minus("Min Packet Length", "Max Packet Length", "Packet Length Std"),
    "x_iiotid": _minus("Min Packet Length", "Max Packet Length", "Packet Length Std"),
    "wustl_iiot": _minus("Packet Length Std"),
    "cic_iot_2023": _minus("Total Backward Packets", "Total Length of Bwd Packets",
                           "Fwd Packets/s", "Bwd Packets/s"),
}

# Structurally lossy against the biflow space (missing direction or duration).
LOSSY = {"mqtt_iot_ids2020", "cic_iot_2023"}


def load(name, root=DEFAULT_ROOT):
    return LOADERS[name](root)


def available(root=DEFAULT_ROOT):
    """Which datasets have their files present under root."""
    checks = {
        "cicids2017": lambda: bool(glob.glob(os.path.join(root, "MachineLearningCVE", "*.csv"))),
        "unsw_nb15": lambda: os.path.exists(os.path.join(root, "UNSWNB15", "UNSW_NB15_training-set.parquet")),
        "ton_iot": lambda: os.path.exists(os.path.join(root, "TONIoT", "train_test_network.csv")),
        "bot_iot": lambda: bool(glob.glob(os.path.join(root, "BotIoT", "*.csv"))),
        "cicddos2019": lambda: bool(glob.glob(os.path.join(root, "CICDDoS2019", "*.parquet"))),
        "iotid20": lambda: bool(glob.glob(os.path.join(root, "IoTID20", "*.csv"))),
        "x_iiotid": lambda: bool(glob.glob(os.path.join(root, "X-IIoTID", "*.csv"))),
        "mqtt_iot_ids2020": lambda: bool(glob.glob(os.path.join(root, "MQTT_IoT_IDS2020", "*.csv"))),
        "cic_iot_2023": lambda: bool(glob.glob(os.path.join(root, "CICIoT2023", "*.csv"))),
        "wustl_iiot": lambda: os.path.exists(os.path.join(root, "WUSTL_IIoT_2021", "wustl_iiot_2021.csv")),
        "iot_23": lambda: bool(glob.glob(os.path.join(root, "IoT23", "**", "*conn.log.labeled"),
                                         recursive=True)),
    }
    present = []
    for name in LOADERS:
        try:
            if checks[name]():
                present.append(name)
        except Exception:
            pass
    return present


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    print("Available:", available(root))
    print()
    for name in available(root):
        try:
            df = load(name, root)
            lossy = " (lossy)" if name in LOSSY else ""
            miss = [f for f in UNIFIED_FEATURES if df[f].isna().all()]
            print(f"  {name:18} rows={len(df):>8,}  attack%={df['y'].mean()*100:5.1f}  "
                  f"NaN-features={len(miss)}{lossy}")
            print(f"  {'':18} cats={sorted(df['category'].unique())}")
            if miss:
                print(f"  {'':18} absent: {miss}")
        except Exception as e:
            print(f"  {name:18} ERROR {e}")
