"""Derive the paper's plot data from committed raw evidence.

    python paper/make_data.py

Reads  reports/pi2-live-2026-09-30/raw/soak.csv  (5-minute Pi 2 soak samples)
Writes paper/data/soak.dat  (hours since start, sensor RSS MB, incidents since
start), which main.tex plots with pgfplots. Standard library only.
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
