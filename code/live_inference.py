#!/usr/bin/env python3
"""
live_inference.py â€” Real-time IoT Intrusion Detection
Reads a pcap file (or live interface via tcpdump pipe),
extracts the 12 SFAF features, and classifies each flow
using the compressed ONNX edge model.

Usage:
    python3 live_inference.py --pcap /path/to/capture.pcap
    python3 live_inference.py --pcap /tmp/test.pcap --verbose
    python3 live_inference.py --pcap /tmp/test.pcap --output results.csv
"""

import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path

import numpy as np

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_edge.onnx"
SCALER_PATH = BASE_DIR / "models" / "scaler_unified_4dataset.pkl"

# â”€â”€ SFAF unified feature order (must match training) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
UNIFIED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
]

# â”€â”€ Import local feature extractor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import read_pcap, build_flows, extract_features


def load_model():
    """Load ONNX runtime session."""
    try:
        import onnxruntime as rt
    except ImportError:
        print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] ONNX model not found at: {MODEL_PATH}")
        print("        Run 02_SFAF_Unified_Model.ipynb first to generate it.")
        sys.exit(1)

    sess = rt.InferenceSession(str(MODEL_PATH))
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[OK] ONNX model loaded ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    return sess, input_name, output_name


def load_scaler():
    """Load fitted StandardScaler."""
    try:
        import joblib
    except ImportError:
        print("[ERROR] joblib not installed. Run: pip install joblib")
        sys.exit(1)

    if not SCALER_PATH.exists():
        print(f"[ERROR] Scaler not found at: {SCALER_PATH}")
        print("        Run 02_SFAF_Unified_Model.ipynb first to generate it.")
        sys.exit(1)

    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Scaler loaded from {SCALER_PATH.name}")
    return scaler


def flow_to_sfaf(flow_features: dict) -> np.ndarray:
    """
    Map feature_extractor.py output dict â†’ SFAF 12-feature vector.
    feature_extractor keys â†’ SFAF unified names:
        duration        â†’ Flow Duration
        pkt_count       â†’ Total Fwd Packets  (bidirectional, used as proxy)
        pkt_count       â†’ Total Backward Packets (same â€” bidirectional flow)
        total_bytes     â†’ Total Length of Fwd Packets
        total_bytes     â†’ Total Length of Bwd Packets
        pkts_per_sec    â†’ Flow Packets/s
        bytes_per_sec   â†’ Fwd Packets/s  (proxy)
        bytes_per_sec   â†’ Bwd Packets/s  (proxy)
        mean_pkt_len    â†’ Min Packet Length  (proxy)
        mean_pkt_len    â†’ Max Packet Length  (proxy)
        mean_pkt_len    â†’ Packet Length Mean
        std_pkt_len     â†’ Packet Length Std
    """
    d  = flow_features["duration"]
    n  = flow_features["pkt_count"]
    tb = flow_features["total_bytes"]
    ml = flow_features["mean_pkt_len"]
    sl = flow_features["std_pkt_len"]
    ps = flow_features["pkts_per_sec"]
    bs = flow_features["bytes_per_sec"]

    vector = [
        d,      # Flow Duration
        n,      # Total Fwd Packets
        n,      # Total Backward Packets (bidirectional proxy)
        tb,     # Total Length of Fwd Packets
        tb,     # Total Length of Bwd Packets (bidirectional proxy)
        ps,     # Flow Packets/s
        bs,     # Fwd Packets/s (proxy)
        bs,     # Bwd Packets/s (proxy)
        max(0, ml - sl),  # Min Packet Length (mean - std proxy)
        ml + sl,          # Max Packet Length (mean + std proxy)
        ml,     # Packet Length Mean
        sl,     # Packet Length Std
    ]
    return np.array(vector, dtype=np.float32).reshape(1, -1)


def classify_pcap(pcap_path: str, verbose: bool = False, output_csv: str = None):
    """Main classification pipeline."""
    sess, input_name, output_name = load_model()
    scaler = load_scaler()

    print(f"\n[*] Reading pcap: {pcap_path}")
    t0 = time.perf_counter()
    packets = read_pcap(pcap_path)
    flows   = build_flows(packets)
    feats   = extract_features(flows)
    read_ms = (time.perf_counter() - t0) * 1000

    print(f"[*] Parsed {len(packets):,} packets â†’ {len(feats):,} flows in {read_ms:.1f}ms")

    if not feats:
        print("[WARN] No flows extracted. Check pcap file.")
        return

    results = []
    n_attack = 0
    t_infer  = 0.0

    for flow in feats:
        raw_vec   = flow_to_sfaf(flow)
        scaled_vec = scaler.transform(raw_vec).astype(np.float32)

        t_start = time.perf_counter()
        pred    = sess.run([output_name], {input_name: scaled_vec})[0][0]
        t_infer += (time.perf_counter() - t_start) * 1000

        label = int(pred)
        if label == 1:
            n_attack += 1

        result = {
            "flow_key"      : flow["flow_key"],
            "duration_s"    : flow["duration"],
            "pkt_count"     : flow["pkt_count"],
            "total_bytes"   : flow["total_bytes"],
            "pkts_per_sec"  : flow["pkts_per_sec"],
            "prediction"    : "ATTACK" if label == 1 else "BENIGN",
            "label"         : label,
        }
        results.append(result)

        if verbose:
            tag = "âš ï¸  ATTACK" if label == 1 else "âœ… BENIGN"
            print(f"  {tag}  {flow['flow_key'][:60]:<60}  "
                  f"{flow['pkt_count']:>4} pkts  {flow['total_bytes']:>8} bytes")

    # â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_total  = len(results)
    n_benign = n_total - n_attack
    avg_ms   = t_infer / n_total if n_total else 0

    print("\n" + "="*55)
    print("  CLASSIFICATION SUMMARY")
    print("="*55)
    print(f"  Total flows classified : {n_total:,}")
    print(f"  BENIGN                 : {n_benign:,}  ({n_benign/n_total*100:.1f}%)")
    print(f"  ATTACK                 : {n_attack:,}  ({n_attack/n_total*100:.1f}%)")
    print(f"  Avg inference latency  : {avg_ms:.4f} ms/flow")
    print(f"  Total inference time   : {t_infer:.2f} ms")
    print("="*55)

    if n_attack > 0:
        print(f"\nâš ï¸  {n_attack} ATTACK FLOW(S) DETECTED")
    else:
        print("\nâœ… All flows classified as BENIGN")

    # â”€â”€ Optional CSV output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[*] Results saved to: {output_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="IoT-IDS live inference pipeline (SFAF + ONNX edge model)"
    )
    parser.add_argument("--pcap",    required=True, help="Path to .pcap file")
    parser.add_argument("--verbose", action="store_true", help="Print per-flow results")
    parser.add_argument("--output",  default=None, help="Save results to CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.pcap):
        print(f"[ERROR] pcap file not found: {args.pcap}")
        sys.exit(1)

    classify_pcap(args.pcap, verbose=args.verbose, output_csv=args.output)


if __name__ == "__main__":
    main()