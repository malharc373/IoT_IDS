---
title: IoT-IDS Engineering Vault
tags: [moc, iot-ids]
created: 2026-08-19
---

# IoT-IDS — Engineering & Research Vault

Working knowledge base for the **IoT-IDS** B.Tech project: an edge intrusion
detection/prevention system for IoT networks, plus the **SFAF** (Semantic
Feature Alignment Framework) cross-dataset generalization study.

Repo: `/Users/malharfalke/IOT-IDS` · remote `github.com/malharc373/IoT_IDS`

> This vault lives inside the repo (`vault/`) so documentation is versioned
> alongside the code it describes. Open the `vault/` folder directly in Obsidian.

---

## Start here

- [[Project Overview]] — what the two halves of the project are and how they fit
- [[Review 2026-08-19]] — the full code/research review that started this work
- [[Remediation Log]] — running log of every fix, in order, with commits
- [[Review 2026-08-21]] — second pass: did the remediation hold? (yes, plus four new findings)
- [[Remediation 2026-08-22]] — active post-review implementation backlog and evidence log

## Findings (problems identified)

Severity-ordered. Each note states the problem, the evidence, and the fix.

### Critical — these change the project's conclusions
- [[F01 - SFAF feature mappings are semantically wrong]]
- [[F02 - Cross-domain F1 includes degenerate classifiers]]
- [[F03 - xgb_edge.onnx exported with the wrong scaler]]
- [[F04 - Live mode reclassifies the entire flow table]]
- [[F05 - Data race between sniffer thread and flush loop]]

### Significant
- [[F06 - IPS only protects the sensor not the network]]
- [[F07 - nftables setup is not idempotent]]
- [[F08 - Rate limiting is documented but not implemented]]
- [[F09 - IPS gate uses uncalibrated confidence]]
- [[F10 - Dashboard binds 0.0.0.0 with no auth]]
- [[F11 - Bot-IoT is loaded non-randomly]]
- [[F12 - Missing features are zero-filled]]
- [[F13 - Live model in-domain metrics are leaky]]
- [[F14 - Host-context features shift between train and deploy]]
- [[F15 - Attack classes separable by dst_port alone]]

### Moderate
- [[F16 - Moderate issues roundup]]
- [[F17 - Documentation inconsistencies]]

### Found during remediation
- [[F18 - Pipeline ONNX export silently ships a broken model]]
- [[F19 - IoT-23 labels parsed as all-benign]]

### Found verifying the remediation ([[Review 2026-08-21]])
- [[F20 - FlowTable generation map grows without bound]] — significant
- [[F22 - Benchmark published a report with a hole in it]] — significant
- [[F23 - Retracted thesis numbers still shipped as current]] — significant
- [[F21 - Dataset entry point dies on an unmounted drive]] — moderate

## Experiments

- [[EXP01 - Cross-dataset study baseline]] — the numbers as they stood pre-fix
- [[EXP02 - Corrected alignment rerun]] — protocol corrected; exact rerun pending
- [[EXP03 - Threshold transfer]] — historical conditional-sampling run; unconditional rerun pending

## Reference

- [[Feature Spaces]] — the 22-feature live space and the 12-feature SFAF space
- [[Dataset Notes]] — per-dataset schema, units, quirks, gotchas
- [[Architecture]] — module map and data flow
- [[Future Work]] — prioritized roadmap out of the review

## Current publication artifacts

- Editable source: `reports/PROJECT_REPORT.md`
- Corrected PDF: `output/pdf/IOT_IDS_Corrected_Technical_Report.pdf`
- Editable deck: `output/presentation/IOT_IDS_Corrected_Project_Review.pptx`
- Superseded April artifacts: `legacy/stale-publication-artifacts/`

## Current state

- 23 findings: 22 fixed, 1 mitigated ([[F09 - IPS gate uses uncalibrated confidence]])
- Test suite **58 passed / 0 failed** (was 23/1 before the first review), plus
  `ruff` in CI and a benchmark that exits non-zero on a failed section
- Prior off-domain run: ROC-AUC 0.509 over 110 dataset pairs, now historical.
  Its diagonal was resubstitution, so no current gap is quoted pending the
  protocol-correct rerun — see [[Remediation 2026-08-22]].

> [!warning] Independent review on 2026-08-22 reopened correctness, deployment,
> research-methodology and governance work. The active status is maintained in
> [[Remediation 2026-08-22]]; the 23-finding count above describes the earlier
> review cycle, not the current readiness verdict.
