---
title: F09 — IPS gate uses uncalibrated confidence
tags: [finding, significant, ips, ml-calibration]
severity: significant
status: mitigated
files: ["src/ips_response.py", "src/train_live_model.py"]
date: 2026-08-19
---

# F09 — The IPS gate uses uncalibrated confidence

## The problem

The decision to blackhole a host rested on a single number:

```python
if confidence < self.min_conf:          # default 0.9
    return {"ip": ip, "action": "monitor", ...}
```

That `confidence` is raw XGBoost softmax output
(`Detector.classify` → `probs[i][lab]`). Gradient-boosted trees trained to
minimise log-loss are systematically **overconfident**: a reported 0.9 does not
mean "90% likely correct", and on a 10-class model trained to ~1.0 accuracy on
its own test split ([[F13 - Live model in-domain metrics are leaky]]) the
softmax saturates near 1.0 for almost everything it sees, including inputs
unlike anything in training.

So `--ips-min-conf 0.9` looked like a safety guarantee and was closer to a
formality. In the `demo_mixed` replay essentially every incident reports
`conf=1.00`.

For a monitoring action that would be tolerable. For an action that severs a
host's connectivity, resting on one saturated score from an uncalibrated model
is not defensible.

## The mitigation

Confidence alone no longer authorises a block. Enforcement additionally requires
**corroboration over time**: `strikes` distinct incidents from the same source
within `strike_window` seconds (defaults 3 / 120 s), tracked in a sliding deque:

```python
def _record_sighting(self, ip, now):
    q = self._sightings[ip]
    q.append(now)
    cutoff = now - self.strike_window
    while q and q[0] < cutoff:
        q.popleft()
    return len(q)
```

Below the strike count the source is **throttled**, not blocked
([[F08 - Rate limiting is documented but not implemented]]) — so a genuine
attack is degraded immediately while evidence accumulates, and a false positive
costs a rate limit rather than a blackhole. Stale sightings are dropped in
`expire()` so strikes cannot accumulate indefinitely across a long session.

Exposed as `--ips-strikes` / `--ips-strike-window`. `strikes=1` restores the old
immediate-block behaviour for callers that explicitly want it.

## Why this is `mitigated`, not `fixed`

Corroboration reduces the *variance* of the decision; it does not make the
underlying probability meaningful. The real fix is to calibrate the model —
`CalibratedClassifierCV` (isotonic or Platt) fit on a held-out split, so that a
reported 0.9 is empirically 0.9 — and then to set `min_conf` from a target
false-positive rate rather than by intuition.

That requires retraining and re-exporting the ONNX and the C header, and it
interacts with [[F13 - Live model in-domain metrics are leaky]]: calibrating
against a leaky split would produce confidently wrong probabilities. Sequenced
in [[Future Work]] behind the scenario-level split fix.

The C export (`models/live_ids.h`) is a further constraint — it returns an
arg-max class id with no probability at all, so the MCU path cannot honour any
confidence gate. Noted in [[F16 - Moderate issues roundup]].

## Related

[[F08 - Rate limiting is documented but not implemented]] ·
[[F13 - Live model in-domain metrics are leaky]] · [[Future Work]]
