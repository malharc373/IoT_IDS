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
| R04 | P0 | Correct initiator/responder flow direction semantics | verified | corpus/model regenerated; ONNX/C parity 100% |
| R05 | P1 | Honor classic-pcap nanosecond timestamp resolution | verified | 0.1 s ns-PCAP regression |
| R06 | P1 | Respect capture link type; reject unsupported types safely | verified | Ethernet validation + SLL rejection |
| R07 | P1 | Parse fragments and IPv6 extension headers safely | verified | IPv4/IPv6 fragment regressions |
| R08 | P1 | Bound live incident suppression state (`seen`) | verified | lifecycle + 10k-key regression |
| R09 | P1 | Refresh IPS firewall timeout and persisted state | verified | memory/disk/backend regression |
| R10 | P1 | Make nft throttling per-source rather than shared | verified | meter grammar + ruleset regression |
| R11 | P0 | Remove in-domain resubstitution from cross-dataset evaluation | implemented; rerun blocked | 80/20 split regression; datasets unmounted |
| R12 | P1 | Account for rows dropped by feature completeness policy | verified | retain-NaN + quality-report regression |
| R13 | P1 | Make Bot-IoT sampling memory-bounded and representative | verified | chunked reservoir regression |
| R14 | P1 | Version IoT-23 preprocessing caches | verified | loader-digest cache regression |
| R15 | P1 | Report threshold-transfer selection bias at small budgets | verified | unconditional-repeat regression |
| R16 | P0 | Resolve 12-feature research vs 22-feature runtime disconnect | verified | explicit separation + runtime contract guard |
| R17 | P1 | Secure dataset downloads (TLS, checksums, safe extraction) | verified | archive security regression + full suite |
| R18 | P1 | Declare complete/reproducible dependencies and lock strategy | verified | fresh-venv suite + lock freshness |
| R19 | P1 | Replace stale report and presentation with editable sources | verified | PDF/PPTX render QA + editable sources |
| R20 | P1 | Reconcile README, vault, daemon, model-card claims | verified | claim guard + full suite |
| R21 | P2 | Add real labelled-traffic and group-split experiments | protocol ready; blocked externally | labelled pcap required |
| R22 | P2 | Run actual Raspberry Pi performance/soak measurements | protocol ready; blocked externally | Pi required |
| R23 | P1 | Add license, model/data cards, branch protection, security settings | partial; owner decision | controls verified; license choice pending |
| R24 | P2 | Remove historical large binaries from Git history | plan ready; decision required | coordinated rewrite |

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

### 2026-08-22 — R04–R07 verified

- **R04:** canonical endpoint sorting remains only the bidirectional dictionary
  key. Packet direction is now relative to the `Flow` initiator. The first
  observed sender is used for partial captures, except a first SYN+ACK reverses
  orientation because it proves the sender is the responder. The reproduced
  request/response case now records 100 forward bytes and 1,000 backward bytes,
  with target port 443. A `FEATURE_CONTRACT_VERSION` now travels in model
  metadata and the C header; the daemon refuses a semantically stale model even
  when feature names and order are unchanged.
- **R05:** classic pcap magic selects a 10^6 or 10^9 timestamp divisor. A
  hand-built nanosecond capture with a 100,000,000-tick interval decodes as
  0.1 seconds instead of 100 seconds.
- **R06:** classic pcap and pcapng readers validate their link type. Unsupported
  Linux cooked, loopback, radiotap and other capture formats now fail with an
  explicit error rather than being silently decoded as Ethernet. Supporting
  those link-layer headers remains future extensibility, but unsafe parsing is
  closed.
- **R07:** non-initial IPv4 and IPv6 fragments are ignored instead of treating
  payload bytes as ports; first fragments still parse normally. IPv6 AH uses
  its RFC length formula. The Scapy live path applies the same fragment rule.

The entire synthetic corpus was regenerated after the semantic fix before any
model was accepted. Results and artifacts were regenerated in dependency order:

```text
corpus                  125,835 flows / 350 scenarios
scenario-held-out test  12,191 flows / 70 unseen scenarios / macro-F1 1.0000
independent demo seeds  47,485 flows / attack recall 100% / benign FPR 0%
ONNX vs sklearn         100.00% agreement on 5,000 flows
C vs XGBoost            100.00% agreement on 525 flows
pytest tests/ -q        49 passed in 13.13s
ruff check .            All checks passed
```

New ONNX SHA-256:
`0a62b0e5c7ad98c8172a0032bd0e6b55e6fb2208defeee2cc773b21b1f2d1336`.

> [!caution] The perfect validation number is still evidence about the
> synthetic generator, not real-network generalisation. R21 remains open. Also,
> a capture that begins midstream without a handshake cannot prove connection
> initiator; the first observed sender is explicitly a fallback, not certainty.

### 2026-08-22 — R08–R10 verified

- **R08:** replay and live paths now use `IncidentWatermarks`. Its state is
  retained only for source/type pairs present in the current classified window.
  Once an incident disappears, a later two-flow recurrence alerts even if an
  old incident from that source had 1,000 flows. A 10,000-key regression prunes
  back to the ten active keys.
- **R09:** a repeated sighting of an active block now updates the reason and
  deadline, persists `ips_state.json`, and refreshes the nft set element with
  an atomic `nft -f` delete/add batch (the CLI has no `update element` command).
  Iptables continues to use the refreshed userspace
  deadline because its rule has no kernel timeout. The test fixes time at 1,000
  and 1,030 seconds and verifies the stored/kernel refresh deadline is 1,090.
- **R10:** the nft throttle rule now uses a named meter keyed by `ip saddr` or
  `ip6 saddr`, with unique meters per hook and address family. Each throttled
  source therefore owns a token bucket; one noisy source cannot consume a
  single global allowance for every other source. Grammar was checked against
  the nftables project documentation:
  <https://wiki.nftables.org/wiki-nftables/index.php/Meters>.

Evidence:

```text
pytest tests/ -q  -> 51 passed in 13.83s
ruff check .      -> All checks passed
git diff --check  -> clean
```

### 2026-08-22 — R11–R15 implemented; invalid results quarantined

- **R11:** each dataset now receives one deterministic, stratified 80/20 split.
  A row model is trained only on the 80%; its diagonal is evaluated on the
  untouched 20%, while off-diagonals use independent target datasets. The long
  output records evaluation type and train/evaluation sizes. The regression
  proves the diagonal partitions are disjoint.
- **R12:** `_finish()` no longer deletes a row because a claimed feature failed
  numeric parsing. The value remains NaN and `alignment_report` records input,
  output, dropped, row-missing and per-feature missing counts. Loaders print the
  quality count, turning possible selection bias into reviewable evidence.
- **R13:** Bot-IoT now uses random-priority reservoir sampling over CSV chunks.
  Memory is bounded by the requested sample plus one chunk and the result is a
  uniform sample without replacement over the entire ordered file.
- **R14:** IoT-23 cache names include a 12-hex SHA-256 digest of the loader
  module. Any parsing, taxonomy, units or alignment code change selects a new
  cache even when raw source mtimes are unchanged.
- **R15:** every requested threshold-calibration repeat now contributes to the
  mean. A single-class labelled draw uses the deployed 0.5 threshold instead of
  being silently skipped. Output includes repeats, calibrated draws and success
  rate, so small-budget results are unconditional rather than lucky-draw-only.

The dataset symlink resolves to `/Volumes/GOAT/...`, but `/Volumes/GOAT` is not
mounted and `md.available()` returns `[]`. Therefore R11's code is implemented
and tested, but its scientific result is not closed. Every affected result was
moved, without deletion, to `legacy/resubstitution-results/`. The live
`demo/results/CROSS_DATASET_FINDINGS.md`, README, vault Home and EXP02 now state
that the exact rerun is pending. Specifically withdrawn:

- in-domain ROC-AUC 0.995;
- the 0.487 in-domain/cross-domain gap;
- exact transform rankings after the row/sampling policy change; and
- the “ten labels recover 89%” claim after unconditional repeat handling.

The historical off-diagonal mean ROC-AUC 0.509 remains evidence of severe
domain shift, but is not presented as the current full-protocol result.

Evidence before the artifact quarantine:

```text
pytest tests/ -q  -> 56 passed in 14.30s
ruff check .      -> All checks passed
```

**R22 preparation:** the benchmark no longer calls a host×12 estimate an
“easily real-time” verdict. Its section is explicitly `UNVALIDATED`, states that
`PI_FACTOR` is not a measurement, and prints the target-Pi acceptance gate.

### 2026-08-22 — R16 resolved as an explicit two-prototype architecture

The 12-feature SFAF binary model was not silently inserted into the live
sensor. Its exact research evidence is currently withdrawn and it does not
accept the 22-feature packet-derived vector; using it for enforcement would be
both technically incompatible and scientifically unjustified.

The boundary is now machine-enforced:

- `live_meta.json` declares `purpose: live_multiclass_ids`,
  `runtime_compatible: true`, feature contract v2, synthetic training data and
  synthetic-only evidence scope;
- future SFAF metadata declares `purpose: sfaf_cross_dataset_research` and
  `runtime_compatible: false`;
- `Detector` validates purpose, runtime compatibility, exact feature order and
  semantic contract **before** creating an ONNX session, using `ValueError`
  checks that cannot disappear under optimized Python;
- a regression proves the daemon rejects research metadata; and
- `models/README.md` is the artifact manifest and defines six gates for any
  future dual-model fusion.

The withdrawn `xgb_edge.onnx`, `edge_meta.json` and `xgb_unified.json` were
preserved under `legacy/resubstitution-results/models/`. The live model remains
in `models/`; no research artifact is presented as deployable.

Evidence:

```text
pytest tests/ -q  -> 57 passed in 14.42s
ruff check .      -> All checks passed
```

### 2026-08-22 — R17 dataset download path hardened

- Removed the WUSTL TLS-certificate bypass; direct downloads now use normal
  certificate verification.
- Kaggle no longer performs its own blind `--unzip`. Both Kaggle and direct
  archives pass through the repository's checked extractor.
- Tar and ZIP members are resolved against the destination before extraction.
  Absolute paths, traversal, links and special files are rejected.
- Every direct archive's SHA-256 is printed. Because the current publishers do
  not provide pinned digests in this repository, extraction defaults to off.
  It requires a digest obtained through a trusted channel via
  `--sha256 SUBDIR=HEX`, or explicit risk acceptance via `--allow-unverified`.
- Regressions exercise malicious tar/ZIP traversal, a valid nested archive,
  digest match/mismatch and the no-digest fail-closed behavior.

Evidence:

```text
pytest tests/ -q  -> 58 passed
ruff check .      -> All checks passed
git diff --check  -> clean
```

### 2026-08-22 — R18 dependency contracts and locks

Three environments now have reviewed direct inputs and exact transitive Python
3.10 locks:

| Environment | Reviewed inputs | Install lock |
|---|---|---|
| Research/training/demo | `requirements.in` | `requirements.txt` |
| Tests and lint | `requirements-dev.in` | `requirements-dev.txt` |
| Pi sensor runtime | `deploy/requirements-pi.in` | `deploy/requirements-pi.txt` |

The first GitHub run after the full remediation exposed a platform condition
that local macOS regeneration could not see: XGBoost depends on NCCL only on
Linux/x86_64. The project now also generates
`requirements-linux-x86_64.txt` and `requirements-dev-linux-x86_64.txt` from
reviewed inputs. Ubuntu installs and freshness-checks those locks; a separate
macOS job freshness-checks the default locks. This avoids both unpinned Linux
transitives and impossible CUDA installation on macOS.

The audit added `pyarrow` for Parquet datasets, `psutil` for benchmark memory
measurements, and `reportlab` for the editable report renderer. `lightgbm`, `seaborn`,
`skl2onnx` and `joblib` are no longer declared as direct dependencies because
project code does not use them; packages still needed transitively remain in
the generated lock. CI installs the development lock and recompiles all locks,
then fails if an input and generated file disagree.

The native Pi requirements were independently resolved against CPython 3.10
aarch64 wheel tags: NumPy 2.2.6 is published for manylinux2014 and ONNX Runtime
1.23.2 for manylinux 2.27/2.28. The deployment target now says glibc 2.28+ and
does not claim unmeasured Pi performance.

The locks pin versions but not wheel hashes. Full multi-platform hash generation
was attempted and stopped after the ML wheel hash scan did not complete in a
reasonable bounded run; this residual artifact-substitution risk is explicit
rather than hidden. R17's dataset checksum gate is independent of Python package
locking.

Evidence:

```text
fresh Python 3.10 venv install  -> requirements-dev.txt installed successfully
fresh-venv pytest tests/ -q     -> 58 passed in 38.58s
fresh-venv ruff check .         -> All checks passed
CPython 3.10 aarch64 wheels     -> NumPy and ONNX Runtime downloaded successfully
pip-compile lock refresh        -> no generated diff
```

### 2026-08-22 — R19 stale publications replaced

The April 2026 39-page report and ten-page image-only presentation repeated
withdrawn cross-dataset metrics, joined the incompatible 12-feature research
and 22-feature runtime stories, and converted a host slowdown estimate into a
Raspberry Pi feasibility conclusion. They were moved intact to
`legacy/stale-publication-artifacts/` with a supersession notice.

The current replacements are:

- `reports/PROJECT_REPORT.md`: editable source for a six-page corrected
  technical report;
- `reports/build_report.py`: deterministic ReportLab renderer;
- `output/pdf/IOT_IDS_Corrected_Technical_Report.pdf`: visually verified PDF;
- `output/presentation/IOT_IDS_Corrected_Project_Review.pptx`: editable
  eleven-slide deck with repository sources embedded in speaker notes.

Both artifacts separate implemented capability from evidence scope. They show
the two-prototype boundary, corrected evaluation protocol, synthetic-only live
model evidence, host-only benchmark, unavailable dataset/Pi blockers, and the
next three acceptance gates. No withdrawn exact SFAF metric was reintroduced.

Visual verification:

```text
PDF                       -> 6 A4 pages rendered and inspected; no clipping
PPTX                      -> 11 slides rendered and inspected at full size
slides_test.py            -> no overflow detected
manual corrections       -> title/rule collisions, label wraps, chart ticks fixed
fresh-venv suite         -> 58 passed in 13.74s; Ruff clean
PDF SHA-256              -> a43c2dfd323a41be74602a430b185752323dfaa9d12d3f4eb4885e9520e2ae39
PPTX SHA-256             -> cc49b2d9e99b7b89e4f7f90a8b914de22680dbbad230c4499e8ac2d238daf91e
```

### 2026-08-22 — R20 current claims reconciled

The README, deployment guides, daemon/trainer descriptions, project overview,
architecture note and roadmap now agree with the current artifacts and evidence:

- the live ONNX file is 91.8 KB and the generated C model contains about 43 KB
  of constant tree data;
- the audited Apple M4 result is 11.1 microseconds per single flow, with
  399,270 flows/s at batch 1024; these are explicitly host-only measurements;
- the unseen-seed synthetic result is 99.65% multiclass accuracy, 0.9961 macro
  F1 and 94.2% Mirai recall, not a real-traffic accuracy claim;
- cross-dataset exact conclusions remain withdrawn until the corrected rerun;
- the raw-feature ONNX and C paths have no scaler; and
- the currently supported IoT-23 source is Zeek flow logs, not packet captures
  capable of validating the live 22-feature extractor.

A regression now scans the authoritative current prose for the stale footprint,
throughput, scaler and withdrawn-polarity claims while requiring the README's
evidence-scope markers.

Evidence:

```text
pytest tests/test_suite.py -q  -> 59 passed in 13.32s
ruff check .                  -> All checks passed
git diff --check              -> clean
```

### 2026-08-22 — R23 governance controls applied; license pending

Local policy now includes a security policy, contribution rules, a complete
live model card, a two-domain data card, monthly Dependabot configuration, and
a governance regression. CI no longer runs duplicate branch-push and PR jobs;
it also runs weekly, has read-only contents permission, and pins GitHub-owned
actions to full commit SHAs.

Authenticated GitHub API access was available. The following remote controls
were enabled and read back: vulnerability alerts, automated security fixes,
private vulnerability reporting, GitHub-owned Actions only, mandatory SHA
pinning, and `main` protection. `main` now requires an up-to-date `test` check,
a pull request, linear history, resolved conversations, admin enforcement, and
disallows force pushes and deletion. The repository description was corrected
to remove an unvalidated Raspberry Pi deployment claim. Full state and rationale
are in [[Repository Governance]].

The remaining R23 item is intentionally not guessed: selecting an open-source
license grants legal rights. The owner must choose the license explicitly.

Evidence:

```text
pytest tests/test_suite.py -q    -> 60 passed in 13.34s
ruff check .                    -> All checks passed
workflow + Dependabot YAML      -> parsed successfully
GitHub protection/settings API  -> applied and read back
```

The first pushed full run initially failed because the macOS-generated lock did
not contain XGBoost's Linux/x86_64-only NCCL dependency. Commit `d07e4eb`
separated default/macOS and Ubuntu x86_64 locks and upgraded pinned Actions to
Node-24-compatible majors. GitHub run `32566402604` then passed both jobs:
`macOS lock freshness` in 27 seconds and the Ubuntu `test` job—including lock
regeneration, lint, 61 tests, artifact contracts, C compilation, demo and
benchmark—in 4 minutes 38 seconds. Both checks are now required on `main`.

GitGuardian remains red for incident `36364668`, which points only to the old
credential-shaped documentation placeholder in commit `aeeb62c`; current code
removed it and guards against recurrence. The incident needs an authenticated
false-positive disposition. Rewriting the branch merely to satisfy the scanner
would violate the separate R24 coordination boundary.

### 2026-08-22 — licensing and history decisions made actionable

The repository contained twelve third-party research PDFs without recorded
redistribution grants. They are removed from the current Git index and ignored,
while all twelve local files remain on this Mac. `Literature/README.md` preserves
their SHA-256 manifest and directs users to the maintained bibliography.

`docs/LICENSE_DECISION.md` scopes rightsholders and recommends Apache-2.0 for
original software/model artifacts plus CC BY 4.0 for original academic works,
subject to author consent. `docs/HISTORY_CLEANUP_PLAN.md` quantifies the 41.69
MiB pack and the historical 56.4 MB / 32.6 MB / 13.1 MB pickle leaders, defines
a narrow purge candidate, and requires a mirror backup, collaboration freeze,
disposable rehearsal, explicit ref leases, fresh-clone CI, and artifact-hash
verification. No history was rewritten.

### 2026-08-22 — R21/R22 external acceptance prepared

The unavailable evidence is no longer represented by a vague future-work line:

- `docs/REAL_TRAFFIC_ACCEPTANCE.md` specifies packet/label provenance, privacy,
  immutable hashes, group isolation, exact live feature extraction, fail-closed
  label joins, required metrics, and the boundary between evaluation and
  threshold calibration.
- `deploy/PI_ACCEPTANCE.md` specifies device identity, artifact hashes, raw
  benchmark retention, a minimum 24-hour passive soak, resource/thermal/drop
  observations, and the result fields required for publication.
- `demo/benchmark.py` no longer emits host-times-12 Pi estimates. On non-Pi
  hardware it prints only an acceptance gate. ARM Linux is not sufficient to
  claim a Pi run: the device tree must explicitly identify Raspberry Pi
  hardware, covered by a regression.

The Apple M4 benchmark was rerun after removing projection. Its current snapshot
is 8.1 microseconds mean single-flow ONNX latency (p99 11.2), 403,058 flows/s at
batch 512, 215,601 packets/s parse-plus-aggregate, 52,249 flows/s end to end,
and 55.1 MB daemon RSS. These remain host-only and run-to-run timings. The PDF
was rebuilt to describe its earlier audited 14:18 snapshot as a snapshot rather
than “latest,” and to reflect completed governance controls.

Evidence:

```text
pytest tests/test_suite.py -q -> 61 passed
benchmark section 8          -> NOT MEASURED; projection intentionally omitted
PDF visual QA                -> 6 rendered pages inspected; no clipping
PDF deterministic rebuild   -> identical SHA-256 on two consecutive builds
PDF SHA-256                  -> a43c2dfd323a41be74602a430b185752323dfaa9d12d3f4eb4885e9520e2ae39
```

## External blockers and boundaries

- A real Pi benchmark cannot be fabricated on this Mac. The code and runbook
  can be prepared; the measurement remains open until Pi output is captured.
- Live-model validation requires a labelled packet capture compatible with the
  22-feature extractor. The supported IoT-23 source is flow logs and the
  external research-dataset mount was unavailable during this audit.
- Rewriting public Git history is destructive and affects collaborators. It is
  tracked, but will not be performed without a specific coordinated decision.
- GitHub branch protection/security settings are remote mutations. They will be
  applied only when credentials and repository permissions are available, and
  recorded with the resulting state.

## Related

[[Home]] · [[Remediation Log]] · [[Review 2026-08-21]] · [[Future Work]]
