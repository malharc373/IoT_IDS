---
title: EXP03 — Threshold transfer
tags: [experiment, sfaf, domain-adaptation, results]
date: 2026-08-20
status: superseded
superseded_by: unconditional calibration rerun pending in Remediation 2026-08-22
---

# EXP03 — How many labelled target flows buy back the gap?

> [!danger] Superseded on 2026-08-22
> Single-class calibration samples were skipped, so small-budget means were
> conditional on drawing both classes. The corrected implementation evaluates
> every draw and falls back to threshold 0.5 when calibration is impossible.
> The tables below are preserved as historical evidence but must not be cited
> until the datasets are remounted and the unconditional study is rerun. See
> [[Remediation 2026-08-22]].

Follow-up to [[EXP02 - Corrected alignment rerun]], which surfaced a result the
old F1-only metrics had hidden: on some held-out datasets the model's *ranking*
transfers while only its *decision threshold* fails.

`code/threshold_transfer.py` · output `demo/results/threshold_transfer.csv`

## Question

If a detector reaches AUC 0.927 on a domain it has never seen but scores F1
0.561 at threshold 0.5, the representation transferred and the operating point
did not. **How many labelled flows from the target domain does it take to fix
just the operating point?**

If the answer is "a handful", then *calibrate on a small labelled sample* is a
deployable answer for that regime — far cheaper than domain adaptation.

## Method

Pool-train on CICIDS2017 + UNSW + TON-IoT + Bot-IoT + CIC-IoT-2023 (the target
is never seen). Score the held-out domain. Draw *n* labelled flows, fit **only a
scalar threshold** on them, evaluate on the remaining flows. 20 draws per
budget, budgets 10 → 1000.

Nothing but a single scalar is fitted, so no target information enters beyond
the operating point.

Three reference lines: **default** (0.5, what ships today), **oracle** (best
threshold fitted on the entire target set — the ceiling ranking quality
permits), and **trivial** (all-attack, 2p/(1+p)).

## The trap, and why the third reference line exists

The first run reported "recovered 99.7% of the calibration gap" for
`wustl_iiot` and `iotid20`. Both were false.

On a domain whose ranking carries no signal, maximising F1 over a threshold
simply **rediscovers the trivial all-attack classifier** — the optimum is a cut
below every score. Their oracle F1 was *exactly* 0.667, the trivial baseline.
The experiment was "recovering" its way to the degenerate solution.

That is [[F02 - Cross-domain F1 includes degenerate classifiers]] resurfacing
inside a brand-new experiment written by someone who had just fixed it
elsewhere. The guard is now structural: every row carries
`oracle_lift = oracle_f1 − trivial_f1`, and the summary refuses to claim a win
when it is ≤ 0.02.

## Results

| held out | AUC | default F1 | oracle F1 | trivial | oracle lift | labels for 80% |
|---|---|---|---|---|---|---|
| **cicddos2019** | 0.927 | 0.561 | 0.905 | 0.667 | **+0.238** | **10** |
| **mqtt_iot_ids2020** | 0.987 | 0.943 | 0.991 | 0.667 | **+0.324** | **10** |
| x_iiotid | 0.706 | 0.667 | 0.689 | 0.667 | +0.022 | 500 (marginal) |
| iotid20 | 0.606 | 0.135 | 0.667 | 0.667 | +0.000 | degenerate |
| wustl_iiot | 0.314 | 0.337 | 0.667 | 0.667 | +0.000 | degenerate |
| iot_23 | 0.195 | 0.678 | 0.681 | 0.667 | +0.014 | degenerate |

### The headline

**CICDDoS2019: ten labelled flows take F1 from 0.561 to 0.868** — 89% of the
way to the 0.905 oracle. Fifty labels reach 0.893 (96.5%); the curve is flat
after that.

```
 10 labels -> F1 0.868  (89.3% of the calibration gap)
 25 labels -> F1 0.880  (92.9%)
 50 labels -> F1 0.893  (96.5%)
250 labels -> F1 0.904  (99.7%)
```

MQTT behaves the same way from a higher base: 0.943 → 0.982 on ten labels.

### The rule

Calibration is a real fix on **3 of 6** held-out datasets, and the predictor of
which is simply **AUC**:

- **AUC ≥ 0.9** → ten labels recover almost everything. The representation
  already transferred; only the operating point was wrong.
- **AUC ≈ 0.7** → marginal. Hundreds of labels for a couple of F1 points.
- **AUC ≤ 0.6, or inverted** → nothing to calibrate toward. No operating point
  rescues a ranking that carries no signal; that needs domain adaptation.

Stated as guidance: **measure AUC on the target domain first.** It tells you
whether you have a cheap calibration problem or an expensive representation
problem — and the two look identical under F1.

## Why this matters for the thesis

[[EXP02 - Corrected alignment rerun]] establishes the negative result:
cross-dataset transfer is at chance. This adds the constructive half — the gap
is not uniform, and the high-AUC slice of it is fixable for the cost of
labelling ten flows. A negative result plus a cheap, measured remedy for the
tractable sub-case is a stronger contribution than the negative result alone.

## Open

- The two **inverted** domains (`wustl_iiot` AUC 0.314, `iot_23` AUC 0.195) are
  reliably *wrong*, not merely uninformative. Flipping the decision would give
  0.686 and 0.805. That is a systematic polarity effect worth tracing to a
  feature — see [[Future Work]].
- Calibrating a *probability* rather than a threshold (Platt/isotonic on the
  target sample) would also fix [[F09 - IPS gate uses uncalibrated confidence]].

## Related

[[EXP02 - Corrected alignment rerun]] ·
[[F02 - Cross-domain F1 includes degenerate classifiers]] ·
[[F09 - IPS gate uses uncalibrated confidence]] · [[Future Work]]
