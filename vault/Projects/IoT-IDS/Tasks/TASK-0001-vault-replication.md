---
task_id: TASK-0001
status: active
priority: P1
owner: Codex orchestrator
created: 2026-08-31T10:20:00+05:30
branch: fix/review-remediation
---

# TASK-0001 — replicate Klimaforge shared-agent vault

## Objective

Make the existing IoT-IDS Obsidian vault function like Klimaforge's shared
database for Claude Code, Codex, and bounded subagents.

## Scope

- Canonical protocol, hubs, task records, handoffs, decisions, logs, templates,
  prompt library, token/delegation rules, resume behavior, and provenance.
- Auto-loaded `AGENTS.md` and `CLAUDE.md` bootstrap caches.
- Project-scoped Codex subagent definitions and concurrency configuration.

## Exclusions

- No datasets, experiments, model retraining, hardware runs, GitGuardian action,
  license choice, history rewrite, merge, or PR comment.
- Do not alter or delete historical vault notes.

## Checklist

- [x] `[complete]` Inspect Klimaforge vault and repository agent configuration.
- [x] `[complete]` Adapt protocol for a versioned in-repo vault.
- [x] `[complete]` Create hubs, registers, templates, prompts, and handoff structure.
- [x] `[complete]` Add synchronized bootstrap files and subagent profiles.
- [x] `[complete]` Validate links, parity, configuration, tests, and Git diff.
- [ ] `[active]` Commit, push, and verify CI.

## Acceptance criteria

- A new Claude/Codex session has an unambiguous read order.
- Tasks have ownership, status, exclusions, dependencies, and evidence.
- Token/model, delegation, collision, background, and heartbeat rules are explicit.
- Bootstrap hard rules agree and point only to canonical volatile state.
- Existing research/audit history remains intact and discoverable.

## Evidence

Reference: `/Users/malharfalke/Documents/Klimaforge-Vault` and
`/Users/malharfalke/Klimaforge/{AGENTS.md,CLAUDE.md,.codex/}`. `[tree]`

```text
AGENTS.md vs CLAUDE.md                -> byte-identical
9 Codex TOML files                   -> parsed
6 Obsidian JSON files                -> parsed
78 vault files / 72 Markdown notes   -> 0 unresolved wikilinks
pytest tests/ -q                     -> 65 passed
ruff + compileall + Bash + diff      -> passed
```

— Codex, 2026-08-31T10:20:00+05:30
