"""
build_corpus.py — turn synthesized traffic into a labeled flow dataset.

For every traffic class we generate many randomized *scenarios* (different
seeds → different IPs, ports, counts, timings), extract per-flow feature
vectors with the shared extractor, and label every flow with its class.

Outputs:
    data/processed/flows.parquet   full labeled flow table (features + labels)
    data/pcaps/<kind>_sample.pcap  one representative capture per class
    data/pcaps/demo_mixed.pcap     benign background + interleaved attacks
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scapy.all import wrpcap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from flow_features import FlowTable, parse_raw, FEATURE_NAMES, N_FEATURES  # noqa: E402
import traffic_gen as tg  # noqa: E402


def _flows_from_packets(pkts):
    """Feature-extract a scapy packet list via the shared pipeline."""
    table = FlowTable()
    # scapy packets already carry .time; convert to raw bytes + parse so the
    # exact same code path as live pcap reading is used.
    for p in sorted(pkts, key=lambda x: float(x.time)):
        raw = bytes(p)
        pk = parse_raw(raw)
        if pk is not None:
            table.add_packet(pk, float(p.time))
    return table.extract(min_pkts=1, window=None)


def build(scenarios: int, out_dir: str, pcap_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pcap_dir, exist_ok=True)

    rows = []
    for kind in tg.ATTACK_KINDS:
        label = tg.LABELS[kind]
        binary = 0 if kind == "benign" else 1
        n_flows_total = 0
        for s in range(scenarios):
            pkts = tg.generate(kind, seed=1000 * tg.LABELS[kind] + s)
            if s == 0:
                wrpcap(os.path.join(pcap_dir, f"{kind}_sample.pcap"), pkts)
            for meta, vec in _flows_from_packets(pkts):
                rows.append(vec + [label, binary, kind])
                n_flows_total += 1
        print(f"  {kind:16s} {scenarios:3d} scenarios -> {n_flows_total:6d} flows")

    cols = FEATURE_NAMES + ["label", "binary", "kind"]
    df = pd.DataFrame(rows, columns=cols)
    # Clean up any non-finite values defensively.
    df[FEATURE_NAMES] = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_NAMES).reset_index(drop=True)

    out_path = os.path.join(out_dir, "flows.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nTotal flows: {len(df):,}  |  features: {N_FEATURES}")
    print("Class distribution:")
    print(df["kind"].value_counts().to_string())
    print(f"\nWrote {out_path}")

    _build_demo_mixed(pcap_dir)
    return df


def _build_demo_mixed(pcap_dir: str):
    """A single capture that mixes benign traffic with each attack, for a
    realistic 'detect the needle in the haystack' demo."""
    pkts = []
    # Distinct seeds per stream so each attacker/host gets its own IPs — mixing
    # the same seed collides source IPs and corrupts host-context features.
    pkts += tg.generate("benign", seed=7007)
    pkts += tg.generate("benign", seed=8008)
    for i, kind in enumerate(["portscan", "synflood", "icmpflood", "udpflood",
                              "ssh_bruteforce", "slowloris"]):
        pkts += tg.generate(kind, seed=90001 + i * 137)
    pkts += tg.generate("benign", seed=9009)
    pkts.sort(key=lambda x: float(x.time))
    path = os.path.join(pcap_dir, "demo_mixed.pcap")
    wrpcap(path, pkts)
    print(f"Wrote {path}  ({len(pkts)} packets)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=25,
                    help="randomized scenarios per class")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--pcaps", default=os.path.join(ROOT, "data", "pcaps"))
    args = ap.parse_args()
    print(f"Building corpus: {args.scenarios} scenarios/class\n")
    build(args.scenarios, args.out, args.pcaps)


if __name__ == "__main__":
    main()
