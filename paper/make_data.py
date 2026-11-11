"""Derive the paper's plot data from committed raw evidence.

    python paper/make_data.py

Reads  reports/pi2-live-2026-09-30/raw/soak.csv  (5-minute Pi 2 soak samples)
Writes paper/data/soak.dat  (hours since start, sensor RSS MB, incidents since
start), which main.tex plots with pgfplots.

Reads  demo/results/cross_dataset_auc_matrix.csv  (corrected SFAF rerun)
Writes paper/data/auc_matrix.dat  (test index, train index, ROC-AUC, ROC-AUC x 100), which
main.tex draws as an 11x11 matrix plot. Standard library only.
"""
import csv
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "reports", "pi2-live-2026-09-30", "raw", "soak.csv")
OUT = os.path.join(HERE, "data", "soak.dat")

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
t0 = datetime.fromisoformat(rows[0]["ts"])
inc0 = int(rows[0]["incidents_emitted"])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("hours rss_mb incidents\n")
    for r in rows:
        h = (datetime.fromisoformat(r["ts"]) - t0).total_seconds() / 3600
        f.write(f"{h:.3f} {float(r['daemon_rss_mb']):.1f} "
                f"{int(r['incidents_emitted']) - inc0}\n")
last = rows[-1]
print(f"{len(rows)} samples, {h:.2f} h, "
      f"{int(last['incidents_emitted']) - inc0} incidents since start -> {OUT}")

# --- cross-dataset ROC-AUC matrix ------------------------------------------
AUC_SRC = os.path.join(HERE, "..", "demo", "results", "cross_dataset_auc_matrix.csv")
AUC_OUT = os.path.join(HERE, "data", "auc_matrix.dat")
ORDER = ["cicids2017", "unsw_nb15", "ton_iot", "bot_iot", "cicddos2019",
         "iotid20", "x_iiotid", "mqtt_iot_ids2020", "cic_iot_2023",
         "wustl_iiot", "iot_23"]          # must match the tick labels in main.tex
grid = {r["train"]: r for r in csv.DictReader(open(AUC_SRC, encoding="utf-8"))}
if sorted(grid) != sorted(ORDER):
    raise SystemExit(f"unexpected datasets in {AUC_SRC}: {sorted(grid)}")
with open(AUC_OUT, "w", encoding="utf-8") as f:
    f.write("test train auc pct\n")
    for i, train in enumerate(ORDER):
        for j, test in enumerate(ORDER):
            v = float(grid[train][test])
            f.write(f"{j} {i} {v:.4f} {round(100 * v)}\n")
        f.write("\n")
print(f"{len(ORDER)}x{len(ORDER)} ROC-AUC matrix -> {AUC_OUT}")
