---
tags: [dashboard, moc, iot-ids]
status: canonical
---

# IoT-IDS shared-agent dashboard

> [!important] Start here
> Read [[Agent-Protocol]], then [[Projects/IoT-IDS/IoT-IDS]], [[Repo-State]],
> [[Tasks]], and [[Open-Decisions]]. The vault—not chat—is durable memory.

## Active database

- [[Projects/IoT-IDS/IoT-IDS]] — project hub
- [[Projects/IoT-IDS/Repo-State]] — volatile repository and verification state
- [[Projects/IoT-IDS/Tasks]] — canonical task queue and checklist index
- [[Projects/IoT-IDS/Open-Decisions]] — owner choices and external gates
- [[Projects/IoT-IDS/Bugs]] — live traps and unresolved defects
- [[Projects/IoT-IDS/Handovers/Handovers]] — resumable session/agent handoffs

## Durable knowledge

- [[Project Overview]] · [[Architecture]] · [[Feature Spaces]]
- [[Dataset Notes]] · [[Future Work]] · [[Repository Governance]]
- [[Home]] — historical review/finding index
- [[Decision Log]] · [[Daily Logs]] · [[Templates]] · [[Prompt Library]]

## Current pointer

> [!note] 2026-08-31 — Codex `[tree]` `[gh]`
> Shared-agent protocol replication is tracked as [[TASK-0001-vault-replication]].
> PR #1 is mergeable; native CI and CodeQL pass. GitGuardian incident 36364668
> remains the only red check and needs explicit owner confirmation to disposition.

> [!success] Completed — 2026-08-31 10:56 IST, Codex `[tree]` `[test]` `[gh]`
> [[TASK-0001-vault-replication]] delivered the Klimaforge-style shared-agent
> database at `b58538b`; CI and CodeQL passed. New sessions now start from this
> dashboard and the canonical [[Agent-Protocol]].
