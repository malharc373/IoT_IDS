## Key Numbers for Report

### Generalization Gap (Section 4.1)
- Baseline (CICIDS-only, 78 feat): CICIDS acc=98.36%, UNSW acc=36.09%
- SFAF Unified (12 feat): CICIDS=98.33%, UNSW=92.68%, TON=99.41%
- Generalization gain: +56.59 pp on UNSW-NB15

### Model Comparison (Table in Section 4.2)
XGBoost: CIC=98.33% F1=0.9584 AUC=0.9985 | UNSW=92.68% F1=0.9432 AUC=0.9844 | TON=99.41% F1=0.9961 AUC=0.9997
RandomForest: CIC=97.58% F1=0.9384 | UNSW=89.34% F1=0.9223 | TON=96.86% F1=0.9798
LightGBM: CIC=98.19% F1=0.9549 | UNSW=90.88% F1=0.9306 | TON=99.13% F1=0.9943

### Edge Deployment (Section 5)
- ONNX model: 45KB (88.9% reduction from 404KB)
- Single-flow latency: 0.010ms mean, 0.011ms P99
- Throughput: 103,659 RPS at batch=1
