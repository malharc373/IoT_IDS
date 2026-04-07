import onnxruntime as rt
import numpy as np
import joblib
import os
import time
import argparse
import pandas as pd

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')

UNIFIED_FEATURES = [
    'Flow Duration','Total Fwd Packets','Total Backward Packets',
    'Total Length of Fwd Packets','Total Length of Bwd Packets',
    'Flow Packets/s','Fwd Packets/s','Bwd Packets/s',
    'Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std'
]

def load_model():
    sess   = rt.InferenceSession(os.path.join(MODELS, 'xgb_edge.onnx'))
    scaler = joblib.load(os.path.join(MODELS, 'scaler_unified_4dataset.pkl'))
    return sess, scaler

def predict_single(sess, scaler, feature_vector: list) -> dict:
    x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x).astype(np.float32)
    t0 = time.perf_counter()
    out = sess.run(None, {'input': x_scaled})
    latency_ms = (time.perf_counter() - t0) * 1000
    label = int(out[0][0])
    confidence = float(out[1][0][label])
    return {
        'label': 'ATTACK' if label == 1 else 'BENIGN',
        'confidence': round(confidence, 4),
        'latency_ms': round(latency_ms, 4)
    }

def predict_csv(sess, scaler, csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    missing = [f for f in UNIFIED_FEATURES if f not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return
    X = df[UNIFIED_FEATURES].replace([np.inf, -np.inf], np.nan).dropna().values.astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    t0 = time.perf_counter()
    preds = sess.run(None, {'input': X_scaled})[0]
    total_ms = (time.perf_counter() - t0) * 1000
    n_attack = int(preds.sum())
    print(f"Processed {len(preds)} flows in {total_ms:.2f}ms")
    print(f"  ATTACK: {n_attack} ({n_attack/len(preds)*100:.1f}%)")
    print(f"  BENIGN: {len(preds)-n_attack} ({(len(preds)-n_attack)/len(preds)*100:.1f}%)")
    print(f"  Avg latency per flow: {total_ms/len(preds)*1000:.3f} µs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IoT-IDS Live Inference')
    parser.add_argument('--csv', type=str, help='Path to CSV file for batch inference')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    args = parser.parse_args()

    sess, scaler = load_model()
    print("Model loaded. Ready for inference.")

    if args.demo:
        print("\n--- Demo: 5 synthetic flows ---")
        rng = np.random.RandomState(42)
        for i in range(5):
            fv = rng.normal(loc=scaler.mean_, scale=np.sqrt(scaler.var_)).tolist()
            result = predict_single(sess, scaler, fv)
            print(f"  Flow {i+1}: {result['label']} (conf={result['confidence']}, latency={result['latency_ms']}ms)")

    elif args.csv:
        predict_csv(sess, scaler, args.csv)

    else:
        print("Use --demo for synthetic test or --csv <path> for batch inference")
