---
tags: [tasks, checklist, hub]
status: active
---

# Tasks — canonical shared queue

The primary orchestrator assigns IDs and owners. One task file is the source of
truth for its checklist. Agents must not start work already owned by another
active task.

| ID | Status | Owner | Task | Dependencies |
|---|---|---|---|---|
| [[TASK-0001-vault-replication|TASK-0001]] | active | Codex orchestrator | Replicate Klimaforge shared-agent vault protocol | none |
| TASK-0002 | blocked | owner | GitGuardian incident 36364668 false-positive disposition | explicit confirmation |
| TASK-0003 | blocked | owner | Choose and apply repository license | rightsholder choice |
| TASK-0004 | blocked | owner | Coordinated historical binary rewrite | explicit destructive approval |
| TASK-0005 | deferred | unassigned | Corrected SFAF dataset reruns | user excluded dataset work |
| TASK-0006 | blocked | hardware owner | Raspberry Pi and real-PCAP acceptance | hardware/evidence |

## Queue rules

New work gets a `TASK-####-slug.md` from [[Task Template]]. Append a dated row or
status correction; do not silently edit historical task outcomes.

> [!success] Status update — 2026-08-31 10:56 IST, Codex `[tree]` `[test]` `[gh]`
> TASK-0001 is complete at `b58538b`: 65 local/CI tests, both CI jobs, and
> CodeQL passed. The row above preserves its original active assignment.
