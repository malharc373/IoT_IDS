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

## Chapter 5 Updates (Today's actual numbers — replace old estimates)

### Section 5.4 (Unified Model Results) — REPLACE table with:
| Test Dataset  | Accuracy | F1     | AUC    | Features |
| CICIDS2017    | 0.9833   | 0.9584 | 0.9985 | 12       |
| UNSW-NB15     | 0.9268   | 0.9432 | 0.9844 | 12       |
| TON-IoT       | 0.9941   | 0.9961 | 0.9997 | 12       |

### Section 5.6 (Multi-Model Comparison) — REPLACE table with:
XGBoost  CIC: Acc=0.9833 F1=0.9584 AUC=0.9985 Time=2.53s
XGBoost  UNSW: Acc=0.9268 F1=0.9432 AUC=0.9844
XGBoost  TON: Acc=0.9941 F1=0.9961 AUC=0.9997
RF       CIC: Acc=0.9758 F1=0.9384 AUC=0.9964 Time=1.03s
RF       UNSW: Acc=0.8934 F1=0.9223 AUC=0.9733
RF       TON: Acc=0.9686 F1=0.9798 AUC=0.9986
LightGBM CIC: Acc=0.9819 F1=0.9549 AUC=0.9979 Time=1.95s
LightGBM UNSW: Acc=0.9088 F1=0.9306 AUC=0.9777
LightGBM TON: Acc=0.9913 F1=0.9943 AUC=0.9989

### Section 5.7 (Edge Deployment) — REPLACE table with:
ONNX model size: 45KB (88.9% reduction from 404KB, NOT 98% from 2291KB)
Mean latency: 0.010ms | P99: 0.011ms | Throughput: 103,659 RPS at batch=1
Batch-64 throughput: 2,555,137 RPS

### Generalisation gain (Section 5.3):
Baseline CICIDS-only on UNSW: 36.09% (NOT 35%)
Unified SFAF on UNSW: 92.68%
Gain: +56.59 percentage points
