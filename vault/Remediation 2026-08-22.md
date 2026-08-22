---
title: Remediation Program — 2026-08-22
tags: [moc, remediation, iot-ids, active]
date: 2026-08-22
branch: fix/review-remediation
status: active
---

# Remediation Program — 2026-08-22

Durable execution record for the independent review performed after
[[Review 2026-08-21]]. This note is the current source of truth while the work
is in progress. Each item moves through **open → in progress → verified** and
records the test evidence and commit that closed it.

## Operating rules

1. Fix one coherent failure mode at a time.
2. Add a regression test that fails for the original bug whenever practical.
3. Run the narrow test first, then the complete suite at phase boundaries.
4. Do not silently replace research artifacts: record when retraining or a new
   experiment is required and keep superseded claims quarantined.
5. Record external actions (GitHub settings, hardware runs, dataset downloads)
   separately from changes that are reproducible in a clone.

## Baseline

- Branch: `fix/review-remediation`
- Starting commit: `35e7605`
- Working tree: clean
- Local branch state: 8 commits ahead of `origin/fix/review-remediation`
- Previously verified: `ruff check .`, 45 pytest checks, full demo and benchmark
- Current merge blockers: GitGuardian false positive; branch protections and
  repository governance are not configured

## Active backlog

| ID | Priority | Work item | Status | Verification |
|---|---:|---|---|---|
| R01 | P0 | Fresh Pi setup reads `RUN_USER` before assignment | verified | strict preflight + full suite |
| R02 | P0 | Authenticated dashboard page does not authenticate API polls | verified | authenticated HTTP regression + full suite |
| R03 | P0 | Remove secret-scanner-triggering CLI credential examples | verified | credential guard + source scan |
| R04 | P0 | Correct initiator/responder flow direction semantics | open | retrain required |
| R05 | P1 | Honor classic-pcap nanosecond timestamp resolution | open | pending |
| R06 | P1 | Respect capture link type; reject unsupported types safely | open | pending |
| R07 | P1 | Parse fragments and IPv6 extension headers safely | open | pending |
| R08 | P1 | Bound live incident suppression state (`seen`) | open | pending |
| R09 | P1 | Refresh IPS firewall timeout and persisted state | open | pending |
| R10 | P1 | Make nft throttling per-source rather than shared | open | pending |
| R11 | P0 | Remove in-domain resubstitution from cross-dataset evaluation | open | rerun required |
| R12 | P1 | Account for rows dropped by feature completeness policy | open | pending |
| R13 | P1 | Make Bot-IoT sampling memory-bounded and representative | open | pending |
| R14 | P1 | Version IoT-23 preprocessing caches | open | pending |
| R15 | P1 | Report threshold-transfer selection bias at small budgets | open | pending |
| R16 | P0 | Resolve 12-feature research vs 22-feature runtime disconnect | open | design/retrain |
| R17 | P1 | Secure dataset downloads (TLS, checksums, safe extraction) | open | pending |
| R18 | P1 | Declare complete/reproducible dependencies and lock strategy | open | pending |
| R19 | P1 | Replace stale report and presentation with editable sources | open | pending |
| R20 | P1 | Reconcile README, vault, daemon, model-card claims | open | pending |
| R21 | P2 | Add real labelled-traffic and group-split experiments | blocked externally | dataset/hardware |
| R22 | P2 | Run actual Raspberry Pi performance/soak measurements | blocked externally | Pi required |
| R23 | P1 | Add license, model/data cards, branch protection, security settings | open | GitHub access needed |
| R24 | P2 | Remove historical large binaries from Git history | decision required | coordinated rewrite |

## Execution log

### 2026-08-22 — program opened

The post-remediation audit was converted into the backlog above. The first
batch is R01–R03 because it is merge-blocking, does not require model retraining,
and can be fully verified without hardware.

### 2026-08-22 — R01–R03 verified

- **R01:** `REPO_DIR`, `RUN_USER`, `VENV` and `PY` are now assigned before the
  token-file ownership operation. `IOTIDS_SETUP_PREFLIGHT_ONLY=1` provides a
  safe executable preflight; the regression runs the real Bash script under its
  existing `set -euo pipefail`, not merely `bash -n`.
- **R02:** the page captures the token from the initial query, stores it only
  for the browser tab, sends it as `X-Auth-Token` on every API poll, and removes
  it from the visible URL/history. The HTTP regression checks that an
  authenticated page contains both propagation and URL-cleanup logic, while
  unauthenticated API requests still receive 401.
- **R03:** credential-shaped `--token <secret>` examples were replaced with
  the private `--token-file PATH` workflow. The repository guard reports no
  credential literals in tracked executable/configuration sources.

Evidence at this checkpoint:

```text
pytest tests/ -v  -> 46 passed in 12.98s
ruff check .      -> All checks passed
```

## External blockers and boundaries

- A real Pi benchmark cannot be fabricated on this Mac. The code and runbook
  can be prepared; the measurement remains open until Pi output is captured.
- Real IoT-23 packet validation depends on the external dataset mount, which was
  unavailable during the independent audit.
- Rewriting public Git history is destructive and affects collaborators. It is
  tracked, but will not be performed without a specific coordinated decision.
- GitHub branch protection/security settings are remote mutations. They will be
  applied only when credentials and repository permissions are available, and
  recorded with the resulting state.

## Related

[[Home]] · [[Remediation Log]] · [[Review 2026-08-21]] · [[Future Work]]
