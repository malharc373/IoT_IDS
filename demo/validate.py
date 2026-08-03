"""
validate.py — held-out validation on UNSEEN traffic scenarios.

The training corpus used seeds 0-24 per class (0..6000+24).  Here we generate
fresh scenarios with far-away seeds (50000+) that the model has never seen,
run them through the exact live pipeline (parse -> flows -> ONNX), and score
predictions against ground truth.  This measures real generalization to new
IPs/ports/timings rather than in-distribution memorization.

Outputs:
    demo/results/heldout_confusion_matrix.png
    demo/results/heldout_report.csv
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "attacks"))

from flow_features import FlowTable, parse_raw  # noqa: E402
from ids_daemon import Detector  # noqa: E402
import traffic_gen as tg  # noqa: E402

RESULTS = os.path.join(ROOT, "demo", "results")
KINDS = ["benign", "portscan", "synflood", "icmpflood",
         "udpflood", "ssh_bruteforce", "slowloris"]
SCENARIOS_PER_CLASS = 10
BASE_SEED = 50_000   # far outside the training range


def flows_of(pkts):
    table = FlowTable()
    for p in sorted(pkts, key=lambda x: float(x.time)):
        pk = parse_raw(bytes(p))
        if pk is not None:
            table.add_packet(pk, float(p.time))
    return table.extract(min_pkts=1, window=None)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    det = Detector()
    label_to_id = {k: i for i, k in enumerate(KINDS)}

    y_true, y_pred = [], []
    for kind in KINDS:
        for s in range(SCENARIOS_PER_CLASS):
            pkts = tg.generate(kind, seed=BASE_SEED + label_to_id[kind] * 1000 + s)
            flows = flows_of(pkts)
            vecs = [v for _, v in flows]
            preds = det.classify(vecs)
            for (pkind, _conf) in preds:
                y_true.append(label_to_id[kind])
                y_pred.append(label_to_id[pkind])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    acc = (y_true == y_pred).mean()
    bin_true = (y_true != 0).astype(int)
    bin_pred = (y_pred != 0).astype(int)
    bin_acc = (bin_true == bin_pred).mean()

    # detection rate (recall on attack classes) and benign false-positive rate
    attack_mask = bin_true == 1
    detection_rate = (bin_pred[attack_mask] == 1).mean()
    benign_mask = bin_true == 0
    fpr = (bin_pred[benign_mask] == 1).mean()

    print("=" * 60)
    print(f"HELD-OUT VALIDATION  ({n:,} flows, unseen seeds {BASE_SEED}+)")
    print("=" * 60)
    print(f"  Multiclass accuracy   : {acc*100:.2f}%")
    print(f"  Binary accuracy       : {bin_acc*100:.2f}%")
    print(f"  Attack detection rate : {detection_rate*100:.2f}%  (recall)")
    print(f"  Benign false-pos rate : {fpr*100:.2f}%")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=list(range(len(KINDS))),
                                target_names=KINDS, digits=4, zero_division=0))

    rep = classification_report(y_true, y_pred, labels=list(range(len(KINDS))),
                                target_names=KINDS, output_dict=True,
                                zero_division=0)
    pd.DataFrame(rep).T.to_csv(os.path.join(RESULTS, "heldout_report.csv"))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(KINDS))))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(KINDS))); ax.set_yticks(range(len(KINDS)))
    ax.set_xticklabels(KINDS, rotation=45, ha="right"); ax.set_yticklabels(KINDS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (ground truth)")
    ax.set_title(f"Held-out Validation — Unseen Scenarios (acc={acc*100:.1f}%)")
    for i in range(len(KINDS)):
        for j in range(len(KINDS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(RESULTS, "heldout_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
