import onnxruntime as rt
import numpy as np
import pandas as pd
import joblib, time, json, os

MODELS = "models/"
DATA   = "data/processed/"

# Load ONNX model + scaler
sess    = rt.InferenceSession(os.path.join(MODELS, "xgb_edge.onnx"))
scaler  = joblib.load(os.path.join(MODELS, "scaler_unified_4dataset.pkl"))
in_name = sess.get_inputs()[0].name
print("Input name:", in_name)
print("Input shape:", sess.get_inputs()[0].shape)

# Load test data
df = pd.read_parquet(os.path.join(DATA, "ton_iot_test.parquet"))
print("Columns:", df.columns.tolist())
