---
title: F15 — Attack classes separable by dst_port alone
tags: [finding, live-model, evaluation, hypothesis-disproved]
severity: significant
status: measured-and-refuted
files: ["src/train_live_model.py"]
date: 2026-08-19
---

# F15 — Attack classes separable by `dst_port` alone

> [!important] Hypothesis raised in review, tested, and **refuted**.
> The concern was legitimate and worth checking. The measurement says the model
> does not depend on the port at all.

## The concern

`dst_port` is feature 22, and the generators draw target ports from small fixed
sets:

| class | target port(s) |
|---|---|
| `mqtt_flood` | 1883 (always) |
| `ssh_bruteforce` | 22 (always) |
| `slowloris` | 80 (always) |
| `synflood` | {80, 443, 22, 8080} |
| `udpflood` | {53, 123, 1900, 5353, 19} |
| `mirai` | {23, 2323, 22} |

With 10 classes and a port that nearly identifies several of them, a tree could
score ~1.0 by memorising port constants and learning no behaviour at all. The
near-perfect scores made this a live worry.

## The measurement

`src/train_live_model.py` now trains a second, identical model with `dst_port`
removed on every run, and reports both plus a per-class F1 delta:

```python
port_idx = FEATURE_NAMES.index("dst_port")
keep = [i for i in range(N_FEATURES) if i != port_idx]
abl = fit_model(X_tr[:, keep], y_tr, len(kinds))
```

Result, on a scenario-level split with mixed benign background:

```
── dst_port ablation ────────────────────────────────────────
  with dst_port   : acc=0.9999  macro-F1=0.9999
  without dst_port: acc=0.9999  macro-F1=0.9999
  delta           : acc=+0.0000  macro-F1=+0.0000
  no class loses more than 0.02 F1 without the port
```

**Zero dependence.** Not "small" — zero, to four decimals, and no individual
class loses more than 0.02 F1. The flow-shape, rate, flag-ratio and
host-context features fully determine the class on their own; the port is
redundant information the model does not need.

## What this does and does not settle

It settles the concern as raised: the model is not a port lookup table. Every
class remains separable from behaviour alone.

It does **not** rescue the headline number. Combined with
[[F13 - Live model in-domain metrics are leaky]] and
[[F14 - Host-context features shift between train and deploy]], the picture is
that the synthetic corpus is separable along *many* redundant axes at once —
which is exactly why removing any one of them changes nothing. That is a
property of the generators, not evidence of a robust detector.

The ablation stays in the trainer as a permanent guard: if a future corpus
change makes the model port-dependent, the delta will say so on the next run.

## Related

[[F13 - Live model in-domain metrics are leaky]] ·
[[F14 - Host-context features shift between train and deploy]] ·
[[Feature Spaces]] · [[Future Work]]
