---
tags: [meta, protocol, claude-code, codex, shared-memory]
date: 2026-08-31
status: canonical
---

# Agent Protocol — shared memory for every IoT-IDS coding agent

This vault is the canonical long-term memory, project database, task ledger,
and handoff channel for Claude Code, Codex, and their project-scoped subagents.
Chat transcripts and model memory are caches. Durable facts belong here.

Canonical path: `/Users/malharfalke/IOT-IDS/vault`.

## 0. Facts that shape this protocol

Nothing loads an Obsidian vault automatically. Claude Code starts from
`CLAUDE.md`; Codex starts from `AGENTS.md`. Those files contain durable hard
rules and pointers, while volatile state exists only in [[Repo-State]],
[[Tasks]], and [[Open-Decisions]]. Never copy moving commit IDs, check results,
or task status into bootstrap files.

This vault is tracked in Git, unlike the Klimaforge reference vault. That gives
history and review, but agents sharing one working tree can still overwrite one
another before a commit. Agents in another worktree must read and write the
canonical path above, not a worktree-local copy. Collision avoidance remains
mandatory.

A vault note records work; it does not deliver it. Delivery is a code/doc
change, commit, push, PR, comment, incident disposition, or other external
action, each subject to its own authorization gate.

## 1. Start-of-task read order

1. `AGENTS.md` or `CLAUDE.md`.
2. This note.
3. [[Projects/IoT-IDS/IoT-IDS]].
4. [[Projects/IoT-IDS/Repo-State]].
5. [[Projects/IoT-IDS/Tasks]].
6. [[Projects/IoT-IDS/Open-Decisions]].
7. Only the findings, handoff, task file, or references in the assigned scope.

Read before investigating. Do not load the whole vault into a prompt. Minimal
context packs save tokens and reduce stale-claim propagation.

## 2. Note classes and write ownership

### Class A — shared hubs

[[Dashboard]], [[Projects/IoT-IDS/IoT-IDS]], [[Projects/IoT-IDS/Repo-State]],
[[Projects/IoT-IDS/Tasks]], [[Projects/IoT-IDS/Open-Decisions]], and
[[Projects/IoT-IDS/Bugs]].

- Re-read immediately before writing.
- Append a dated, signed block. Do not rewrite an earlier agent's block.
- Corrections are new dated callouts naming what they supersede.
- Never delete. Archive or mark superseded.
- The primary orchestrator owns hub edits during an active multi-agent task.
  Subagents return a proposed block or write only when explicitly assigned.

### Class B — event and handoff notes

Investigations, reviews, experiments, validations, and handoffs use a unique,
dated filename. They are immutable once written. Correct them only by appending
`## Correction — <timestamp>, <agent>`.

### Class C — registers

[[Decision Log]], [[Daily Logs]], and task index entries are append-only. A
reversed decision is a new note that links to and supersedes the earlier one.

### Class D — stable reference

`Reference/`, `Resources/`, `Templates/`, `Prompt Library/`, and this protocol.
Change deliberately and record protocol changes in the decision log.

### Class E — active task records

One file per task under `Projects/IoT-IDS/Tasks/`, named
`TASK-####-short-slug.md`. Only the named owner may edit its body while
`status: active`. Everyone else may append a signed observation or message the
owner. The orchestrator assigns ownership and updates the task hub.

Every task records: objective, owner/agent, status, branch/worktree, scope,
explicit exclusions, dependencies, acceptance criteria, evidence, changed
files, tests, blockers, and the next resumable action.

Allowed states: `pending`, `active`, `blocked`, `complete`, `cancelled`, and
`superseded`. Do not use `complete` until every acceptance criterion is met.

## 3. Provenance

Tag material claims:

| Tag | Meaning |
|---|---|
| `[tree]` | Verified against files, Git history, diff, or a local command. |
| `[test]` | Produced by a named test/check on a named commit. |
| `[gh]` | Verified against live GitHub state this session. |
| `[artifact]` | Verified from a named model, report, pcap, or result artifact. |
| `[external]` | Verified from a cited primary external source. |
| `[relayed]` | Supplied by the user or another person; not independently verified. |
| `[rec]` | Recommendation, not an approved decision. |
| `[unverified]` | Believed but not checked. |

Sign non-trivial blocks `— <agent>, <ISO timestamp>`. Prefer `as of <commit>`
over “currently.” Never silently upgrade provenance.

## 4. Checklist-first execution

For every substantive multi-step task:

1. Reconcile the canonical vault, Git/GitHub, tests, worktrees, running workers,
   hardware/dataset availability, and protected artifacts.
2. Create or refresh one canonical task checklist. Each item gets dependencies,
   acceptance criteria, and a state. Copy every user exclusion into it.
3. Execute dependency-ordered tranches. Parallelize only independent work with
   disjoint ownership.
4. Update the task record continuously, especially before long waits, usage
   limits, background execution, or likely interruption.
5. Reconcile the resulting tree, test evidence, external state, task record,
   and open decisions before declaring completion.

The checklist never overrides approval requirements for destructive actions,
credential use, incident disposition, comments, reviews, PRs, merges, releases,
or other external mutations.

## 5. Token and delegation policy

- Use the cheapest capable model and lowest sufficient reasoning effort.
- Low-cost/low-effort agents handle inventory, extraction, formatting,
  deterministic checks, and mechanical edits.
- Stronger models or high effort are reserved for architecture, ambiguous root
  cause, security/scientific validity, and final integration review.
- Give every subagent one bounded outcome, explicit files/ownership, exclusions,
  acceptance evidence, and a concise return format. Never delegate “everything.”
- Do not send a subagent the entire vault. Send the task file and only linked
  references it needs.
- Maximum three concurrent subagents per orchestrator session. Parallel tasks
  must not edit the same files or the same Class A hub.
- Subagents do not recursively delegate unless explicitly authorized.
- Subagents never commit, push, change branches/worktrees, post, merge, or make
  external writes. The primary orchestrator integrates and verifies.
- Stop after the bounded deliverable. Escalate model/effort only when evidence
  shows the lower tier is insufficient.
- After two serious failed fixes of the same cause, use one independent rescue
  attempt. The owner gets one integration attempt; then record the disagreement
  or blocker instead of creating an agent loop.
- Token efficiency comes from bounded scope and selective reads, not from
  skipping verification or inventing arbitrary output limits.

## 6. Background work, limits, and resume heartbeat

Claude Code owns safely separable long-running/background execution when it is
available, including dataset downloads/reruns and hardware or traffic captures.
Codex owns scope, orchestration, acceptance criteria, bounded review agents,
integration, evidence review, safety, and final verification. A task may name a
different owner explicitly.

Before a credit/token limit or long wait, record worker/session identity,
branch/worktree, commit, process/run ID, last output, tests, remaining criteria,
and the exact next action. Maintain only one resume heartbeat per active task.
It must inspect whether work is already running before relaunching it. Never
duplicate an active worker. If automation is unavailable, leave a precise
handoff and stop.

Never put credentials, tokens, private packet contents, or secrets in the vault.

## 7. Handoff protocol

1. Create a unique dated note from [[Session Handoff Template]].
2. Include completed, unfinished, blockers, explicit not-verified list, branch,
   worktree, HEAD, dirty/unpushed state, running processes, and first next task.
3. Append one dated pointer to the relevant hub/task record.
4. Put undecided questions in [[Open-Decisions]] and settled choices in
   [[Decision Log]].
5. State what was deliberately not done and why.

## 8. Link and evidence rules

Use bare unique stems, a path suffix, or a full vault-root path in wikilinks.
Never put a parent-directory prefix inside a wikilink. Verify links mechanically.

Do not duplicate repository prose. Record rationale, evidence, dead ends,
corrections, decisions, traps, and unverified boundaries—the information that
cannot be cheaply reconstructed from the tree.

## Related

[[Dashboard]] · [[Projects/IoT-IDS/IoT-IDS]] · [[Projects/IoT-IDS/Repo-State]] ·
[[Projects/IoT-IDS/Tasks]] · [[Projects/IoT-IDS/Open-Decisions]]
