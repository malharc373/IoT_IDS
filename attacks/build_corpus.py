"""
build_corpus.py — turn synthesized traffic into a labeled flow dataset.

For every traffic class we generate many randomized *scenarios* (different
seeds → different IPs, ports, counts, timings), extract per-flow feature
vectors with the shared extractor, and label every flow with its class.


TWO PROPERTIES THIS FILE HAS TO GET RIGHT
------------------------------------------

**1. Scenario provenance** (vault/Findings/F13). Every row carries the
`scenario` it came from. Flows from one scenario share an attacker, a victim
and — exactly — the same host-context feature values, so a random row split
puts near-identical siblings on both sides of the train/test boundary. The
trainer groups on this column instead.

**2. Realistic host context** (vault/Findings/F14). Attack scenarios are
generated *mixed with benign background traffic* by default. Three of the 22
features (`host_dst_ports`, `host_dst_ips`, `host_flow_count`) are computed
across all of a source's flows in the window — so generating an attack in an
empty capture measures those features against a background that never exists at
deployment time. `--no-mix-background` restores the old isolated behaviour for
comparison.

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

from flow_features import (  # noqa: E402
    FlowTable, parse_raw, _flow_key, FEATURE_NAMES, N_FEATURES,
)
import traffic_gen as tg  # noqa: E402

# Background traffic is drawn from a seed range well away from every foreground
# scenario, so a mixed capture never accidentally reuses the attacker's IPs.
BG_SEED_BASE = 600_000


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


def _keys_of(pkts):
    """The set of canonical flow keys a packet list produces."""
    keys = set()
    for p in pkts:
        pk = parse_raw(bytes(p))
        if pk is not None:
            keys.add(_flow_key(pk)[0])
    return keys


def _mixed_scenario(kind, seed, bg_seed):
    """Attack traffic overlaid on benign background, with per-flow labels.

    Returns (rows, n_ambiguous) where each row is (features, label_kind).

    Labelling is by flow key rather than by IP: a flow belongs to whichever
    packet set produced it. Keys appearing in BOTH sets are ambiguous (the
    generators drew the same 5-tuple by chance) and are dropped rather than
    guessed at — they are rare and a wrong label is worse than a missing row.
    """
    atk = tg.generate(kind, seed=seed)
    variant = tg.BENIGN_VARIANTS[bg_seed % len(tg.BENIGN_VARIANTS)]
    bg = tg.generate(variant, seed=bg_seed)

    atk_keys, bg_keys = _keys_of(atk), _keys_of(bg)
    ambiguous = atk_keys & bg_keys

    table = FlowTable()
    for p in sorted(atk + bg, key=lambda x: float(x.time)):
        pk = parse_raw(bytes(p))
        if pk is not None:
            table.add_packet(pk, float(p.time))

    rows = []
    # extract_live gives us the flow key alongside meta/vector so each flow can
    # be attributed to the packet set that produced it
    for key, _meta, vec, _ in table.extract_live(min_pkts=1, window=None):
        if key in ambiguous:
            continue
        if key in atk_keys:
            rows.append((vec, kind))
        elif key in bg_keys:
            rows.append((vec, "benign"))
    return rows, len(ambiguous)


def build(scenarios: int, out_dir: str, pcap_dir: str, mix_background: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pcap_dir, exist_ok=True)

    rows = []
    total_ambiguous = 0
    for kind in tg.ATTACK_KINDS:
        n_flows_total = 0
        # benign yields fewer flows/scenario, so oversample it for balance
        n_scen = scenarios * 5 if kind == "benign" else scenarios
        for s in range(n_scen):
            seed = 1000 * tg.LABELS[kind] + s
            scenario = f"{kind}:{seed}"
            if kind == "benign":
                # the benign label is a family: rotate through plain + hard-benign
                # variants so the model sees legitimate traffic that resembles
                # attacks (bursty transfers, multi-endpoint telemetry).
                variant = tg.BENIGN_VARIANTS[s % len(tg.BENIGN_VARIANTS)]
                pkts = tg.generate(variant, seed=seed)
                if s == 0:
                    wrpcap(os.path.join(pcap_dir, f"{kind}_sample.pcap"), pkts)
                scen_rows = [(vec, "benign") for _, vec in _flows_from_packets(pkts)]
            elif mix_background:
                scen_rows, amb = _mixed_scenario(kind, seed, BG_SEED_BASE + seed)
                total_ambiguous += amb
                if s == 0:
                    wrpcap(os.path.join(pcap_dir, f"{kind}_sample.pcap"),
                           tg.generate(kind, seed=seed))
            else:
                pkts = tg.generate(kind, seed=seed)
                if s == 0:
                    wrpcap(os.path.join(pcap_dir, f"{kind}_sample.pcap"), pkts)
                scen_rows = [(vec, kind) for _, vec in _flows_from_packets(pkts)]

            for vec, lab in scen_rows:
                rows.append(vec + [tg.LABELS[lab], 0 if lab == "benign" else 1,
                                   lab, scenario])
                n_flows_total += 1
        print(f"  {kind:16s} {n_scen:3d} scenarios -> {n_flows_total:6d} flows")

    cols = FEATURE_NAMES + ["label", "binary", "kind", "scenario"]
    df = pd.DataFrame(rows, columns=cols)
    # Clean up any non-finite values defensively.
    df[FEATURE_NAMES] = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_NAMES).reset_index(drop=True)

    out_path = os.path.join(out_dir, "flows.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nTotal flows: {len(df):,}  |  features: {N_FEATURES}"
          f"  |  scenarios: {df['scenario'].nunique():,}")
    if mix_background:
        print(f"Background mixing ON — attack scenarios carry benign traffic so "
              f"host-context features are measured against a realistic backdrop")
        if total_ambiguous:
            print(f"  ({total_ambiguous} ambiguous flow key(s) dropped)")
    else:
        print("Background mixing OFF — host-context features are measured in an "
              "empty capture (does not match deployment; see F14)")
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
    pkts += tg.generate("benign_burst", seed=8008)
    pkts += tg.generate("benign_multi", seed=8123)
    for i, kind in enumerate([k for k in tg.ATTACK_KINDS if k != "benign"]):
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
    ap.add_argument("--no-mix-background", action="store_true",
                    dest="no_mix_background",
                    help="generate attack scenarios in isolation (pre-2026-08 "
                         "behaviour; host context then has no realistic "
                         "background — see vault/Findings/F14)")
    args = ap.parse_args()
    print(f"Building corpus: {args.scenarios} scenarios/class\n")
    build(args.scenarios, args.out, args.pcaps,
          mix_background=not args.no_mix_background)


if __name__ == "__main__":
    main()
