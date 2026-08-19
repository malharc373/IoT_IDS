---
title: F13 — Live model in-domain metrics are leaky
tags: [finding, significant, live-model, evaluation, hypothesis-disproved]
severity: significant
status: fixed
files: ["attacks/build_corpus.py", "src/train_live_model.py"]
date: 2026-08-19
---

# F13 — Live model in-domain metrics are leaky

> [!important] The fix landed; the hypothesis was wrong.
> The split *was* leaky and is now correct. But leakage was **not** why the
> score was 1.0 — the score did not move. See "What actually explains it".

## The problem

`models/live_meta.json` reported:

```json
"metrics": {"multiclass_accuracy": 1.0, "macro_f1": 1.0,
            "binary_accuracy": 1.0, "binary_f1_attack": 1.0}
```

Four metrics, all exactly 1.0000. That is a leakage signature, and the
mechanism was real: the corpus is generated as randomized *scenarios*, and

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, ..., stratify=y)
```

split on **rows**. Flows from one scenario share an attacker, a victim, and —
bit for bit — the same `host_dst_ports`, `host_dst_ips` and `host_flow_count`,
because those three features are computed across that scenario's own flows. So
each test flow had near-identical siblings in training.

`demo/results/BENCHMARK.md` printed these numbers under "In-domain metrics" with
no caveat.

## The fix

`attacks/build_corpus.py` now records a `scenario` column (`kind:seed`) on every
row, and `src/train_live_model.py` splits with `GroupShuffleSplit` on it, with
an assertion that nothing leaks:

```python
overlap = set(groups[tr_idx]) & set(groups[te_idx])
assert not overlap, f"scenario leaked across the split: {list(overlap)[:3]}"
```

```
Train: 49,444 flows / 336 scenarios   Test: 10,405 flows / 84 scenarios
Split is by SCENARIO — the test set contains unseen attackers.
```

## What actually explains it

With a clean scenario-level split — test attackers never seen in training, and
[[F14 - Host-context features shift between train and deploy]] mixing benign
background into every scenario — the result barely moves:

| | pre-fix (row split) | post-fix (scenario split) |
|---|---|---|
| Multiclass accuracy | 1.0000 | 0.9999 |
| Macro F1 | 1.0000 | 0.9999 |

And `demo/validate.py`, on freshly generated scenarios with seeds 50000+:

```
HELD-OUT VALIDATION  (47,484 flows, unseen seeds 50000+)
  Multiclass accuracy   : 100.00%
  Attack detection rate : 100.00%
  Benign false-pos rate :   0.00%
```

**The synthetic task is simply trivial.** The generators produce classes that
are perfectly separable regardless of how the data is split, whether benign
background is mixed in, or whether `dst_port` is available
([[F15 - Attack classes separable by dst_port alone]]). The near-1.0 score was
never mostly leakage — it is the corpus.

## Why the fix still matters

The number is unchanged but its **status** is not. Before, ~1.0 was
uninterpretable: it could have been leakage, port memorisation, or a genuinely
easy task, with no way to tell. Three measurements now rule out the first two,
which leaves exactly one honest conclusion — and it is the one that matters:

> The live model's ~99.99% is a statement about the difficulty of the
> **synthetic generators**, not about detection performance. It cannot be
> quoted as an accuracy figure for real traffic.

That makes validating on real labelled traffic (IoT-23 pcaps) the single
highest-value next step, not a nice-to-have. See [[Future Work]].

## Related

[[F14 - Host-context features shift between train and deploy]] ·
[[F15 - Attack classes separable by dst_port alone]] ·
[[F09 - IPS gate uses uncalibrated confidence]] (calibrating against a trivial
task produces confidently useless probabilities) · [[Future Work]]
