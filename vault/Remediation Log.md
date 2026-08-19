---
title: Remediation Log
tags: [log, iot-ids]
date: 2026-08-19
branch: fix/review-remediation
---

# Remediation Log — 2026-08-19

Work done against [[Review 2026-08-19]], in order. Branch
`fix/review-remediation` off `main`.

## Status

| # | Finding | Severity | Status |
|---|---|---|---|
| F01 | [[F01 - SFAF feature mappings are semantically wrong]] | critical | fixed |
| F02 | [[F02 - Cross-domain F1 includes degenerate classifiers]] | critical | fixed |
| F03 | [[F03 - xgb_edge.onnx exported with the wrong scaler]] | critical | fixed |
| F04 | [[F04 - Live mode reclassifies the entire flow table]] | critical | fixed |
| F05 | [[F05 - Data race between sniffer thread and flush loop]] | critical | fixed |
| F06 | [[F06 - IPS only protects the sensor not the network]] | significant | fixed |
| F07 | [[F07 - nftables setup is not idempotent]] | significant | fixed |
| F08 | [[F08 - Rate limiting is documented but not implemented]] | significant | fixed |
| F09 | [[F09 - IPS gate uses uncalibrated confidence]] | significant | mitigated |
| F10 | [[F10 - Dashboard binds 0.0.0.0 with no auth]] | significant | fixed |
| F11 | [[F11 - Bot-IoT is loaded non-randomly]] | significant | fixed |
| F12 | [[F12 - Missing features are zero-filled]] | significant | fixed |
| F13 | [[F13 - Live model in-domain metrics are leaky]] | significant | fixed (hypothesis refuted) |
| F14 | [[F14 - Host-context features shift between train and deploy]] | significant | fixed |
| F15 | [[F15 - Attack classes separable by dst_port alone]] | significant | measured, refuted |
| F16 | [[F16 - Moderate issues roundup]] | moderate | fixed |
| F17 | [[F17 - Documentation inconsistencies]] | moderate | fixed |
| F18 | [[F18 - Pipeline ONNX export silently ships a broken model]] | critical | fixed — *found during remediation* |

Test suite: **23 passed / 1 failed** before → **37 passed / 0 failed** after.

## Commits

| commit | scope |
|---|---|
| `d7d7087` | vault skeleton |
| `9052b0f` | F01 — SFAF alignment: semantics, units, coverage |
| `f874c52` | F02 — threshold-free, prevalence-robust metrics |
| `f963a82` | F03 — one scaler for SFAF; delete duplicate maps |
| `f1d9c42` | F04, F05 — thread-safe flow table, incremental scoring |
| `bf86af4` | F06–F09 — IPS ladder, scope, idempotence |
| `2f0d847` | F10 — dashboard auth, binding, incremental reads, rotation |
| `1a9e543` | secret hygiene (see below) |
| `443a0be` | F13–F15, F18 — scenario split, background mixing, ablation, export fix |
| `ed28743` | F16 — IPv6, snaplen, TCP teardown, dead code, pickles |

## Course corrections worth recording

### A GitGuardian alert, mid-work

The dashboard commit put the live token into the generated systemd unit as
`Environment=IOTIDS_DASHBOARD_TOKEN=...` and hardcoded a fake token literal in a
test. No real credential was exposed — the installer's token is generated on the
Pi at run time in a gitignored directory — but the unit-file placement was a
genuine weakness beyond the scanner: units under `/etc/systemd/system` are
world-readable and the value shows in `systemctl show`.

Fixed by adding `--token-file` (600, owned by the service user), minting tokens
at runtime in tests, and adding a **`no credential literals`** test that scans
tracked sources. It caught its own docstring example on the first run.

### Two hypotheses from the review were wrong

Worth stating plainly, because the value is in the measurement either way:

- **F13** predicted the live model's exactly-1.0 metrics were mostly leakage.
  The split *was* leaky and is now correct — and the score did not move
  (1.0000 → 0.9999, held-out 100%). The real explanation is that the synthetic
  corpus is trivially separable.
- **F15** predicted the model leaned on `dst_port` constants. The ablation says
  **zero** dependence, to four decimals.

Both are now permanent measurements in the trainer rather than open questions.

### One new critical finding surfaced during the work

[[F18 - Pipeline ONNX export silently ships a broken model]] — the retrain
exposed an ONNX export that produced an internally-consistent 16.6%-accurate
stand-in for a 99.99% model. Found only because the parity check printed a
number; it now aborts and deletes the artifact instead.

## The through-line

Three of the five critical findings (F03, F18, and F05's timestamp half) are the
same failure mode: **a step that could silently produce a wrong artifact, with a
check that either did not exist or could not fail the build.** Every export path
now aborts on mismatch:

```python
if agree < 0.999:
    os.remove(onnx_path)
    sys.exit("[ERROR] exported ONNX disagrees with the trained pipeline ...")
```

The second theme is **metrics that cannot distinguish success from degeneracy**
(F02, F13, F15). The fix in each case was not a better number but a number
reported *next to what it must beat* — the trivial baseline, the ablated model,
the chance line.

## Related

[[Home]] · [[Review 2026-08-19]] · [[EXP02 - Corrected alignment rerun]] ·
[[Future Work]]
