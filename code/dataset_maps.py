"""
dataset_maps.py — semantic feature alignment (SFAF) for multiple IDS datasets.

Each public dataset ships a different schema. SFAF maps them all into one
canonical 12-feature space so a single model can train across them. This module
is the single source of truth for those mappings and is unit-tested on synthetic
mini-frames (see tests/test_dataset_maps.py) so the alignment is verified even
without the multi-GB downloads.

Supported: CICIDS2017, UNSW-NB15, TON-IoT, CIC-IoT-2023, Bot-IoT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNIFIED_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std",
]

# CICIDS2017 already uses the canonical names (identity map after strip).
CICIDS_MAP = {f: f for f in UNIFIED_FEATURES}

UNSW_MAP = {
    "dur": "Flow Duration", "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets", "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets", "rate": "Flow Packets/s",
    "sload": "Fwd Packets/s", "dload": "Bwd Packets/s",
    "smean": "Min Packet Length", "dmean": "Max Packet Length",
    "sjit": "Packet Length Mean", "djit": "Packet Length Std",
}

TON_MAP = {
    "duration": "Flow Duration", "src_pkts": "Total Fwd Packets",
    "dst_pkts": "Total Backward Packets", "src_bytes": "Total Length of Fwd Packets",
    "dst_bytes": "Total Length of Bwd Packets", "src_ip_bytes": "Flow Packets/s",
    "dst_ip_bytes": "Bwd Packets/s", "missed_bytes": "Fwd Packets/s",
    "src_port": "Min Packet Length", "dst_port": "Max Packet Length",
    "http_request_body_len": "Packet Length Mean",
    "http_response_body_len": "Packet Length Std",
}

# Bot-IoT (UNSW) is flow-based and maps cleanly.
BOTIOT_MAP = {
    "dur": "Flow Duration", "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets", "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets", "rate": "Flow Packets/s",
    "srate": "Fwd Packets/s", "drate": "Bwd Packets/s",
    "min": "Min Packet Length", "max": "Max Packet Length",
    "mean": "Packet Length Mean", "stddev": "Packet Length Std",
}

# CIC-IoT-2023 is packet-window based (no clean fwd/bwd split); best-effort
# proxies keep the semantics (rates, sizes, counts) aligned.
CICIOT_MAP = {
    "flow_duration": "Flow Duration", "Number": "Total Fwd Packets",
    "Tot sum": "Total Length of Fwd Packets", "Tot size": "Total Length of Bwd Packets",
    "Rate": "Flow Packets/s", "Srate": "Fwd Packets/s", "Drate": "Bwd Packets/s",
    "Min": "Min Packet Length", "Max": "Max Packet Length",
    "AVG": "Packet Length Mean", "Std": "Packet Length Std",
    # no distinct backward-packet count; approximate with header count if present
    "Header_Length": "Total Backward Packets",
}

MAPS = {
    "cicids2017": CICIDS_MAP,
    "unsw_nb15": UNSW_MAP,
    "ton_iot": TON_MAP,
    "bot_iot": BOTIOT_MAP,
    "cic_iot_2023": CICIOT_MAP,
}


def align(df: pd.DataFrame, dataset: str, label_col: str = "label") -> pd.DataFrame:
    """Rename a dataset's columns into the canonical 12-feature space and return
    a frame with exactly UNIFIED_FEATURES + 'label'. Missing mapped columns are
    filled with 0.0 so partial schemas still align."""
    if dataset not in MAPS:
        raise ValueError(f"unknown dataset '{dataset}'. known: {list(MAPS)}")
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=MAPS[dataset])
    for f in UNIFIED_FEATURES:
        if f not in df.columns:
            df[f] = 0.0
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column missing from {dataset}")
    out = df[UNIFIED_FEATURES + [label_col]].copy()
    out[UNIFIED_FEATURES] = (
        out[UNIFIED_FEATURES].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return out.dropna(subset=UNIFIED_FEATURES).reset_index(drop=True)
