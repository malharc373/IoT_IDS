# IoT-IDS agent bootstrap

This repository contains two separate prototypes:

1. `src/` + `models/live_*`: 22-feature, 10-class live edge IDS/IPS.
2. `code/`: 12-feature binary SFAF cross-dataset research.

Never interchange their models, scalers, metadata, features, or claims.

## Canonical shared database

The Obsidian vault at `/Users/malharfalke/IOT-IDS/vault` is the shared
long-term memory, knowledge base, task ledger, and handoff channel for Claude
Code, Codex, and project-scoped subagents. This file is a bootstrap cache.

At the start of substantive work, read in order:

1. `vault/Agent-Protocol.md`
2. `vault/Projects/IoT-IDS/IoT-IDS.md`
3. `vault/Projects/IoT-IDS/Repo-State.md`
4. `vault/Projects/IoT-IDS/Tasks.md`
5. `vault/Projects/IoT-IDS/Open-Decisions.md`
6. the assigned TASK file and only its linked references

Volatile facts—HEAD, checks, PR state, active workers, and task status—live only
in the vault. Do not restate them here. Agents in another worktree still use
the canonical absolute vault path above.

## Hard rules

- Preserve unrelated dirty state and other agents' edits. Check status and task
  ownership before changing files.
- The primary orchestrator owns shared hubs, integration, commits, pushes,
  external actions, and final verification. Subagents do not commit, push,
  change branches/worktrees, post, merge, or perform external writes.
- Use a canonical TASK record for substantive multi-step work. It must include
  owner, scope, exclusions, dependencies, acceptance criteria, evidence, and
  exact next action. Do not duplicate an active task.
- Document material work continuously, especially before long waits, credit or
  token boundaries, background execution, or interruption.
- Maintain at most one non-duplicating resume heartbeat per active task. Inspect
  existing workers/runs before relaunching anything.
- Choose the cheapest capable model and lowest sufficient reasoning effort.
  Mechanical inventory/checks use low-cost agents; architecture, ambiguous
  debugging, security/scientific validity, and final review justify stronger
  models. Escalate only after evidence shows the lower tier is insufficient.
- Delegations are bounded, have disjoint file ownership, and receive only the
  task file plus necessary references—not the entire vault. Maximum three
  concurrent subagents. No recursive delegation without explicit permission.
- After two serious failed fixes for the same cause, allow one independent
  rescue attempt and one integration attempt; then record the blocker instead
  of creating an agent loop.
- Claude Code owns safely separable long-running/background execution when
  available, especially dataset, hardware, and traffic-capture jobs. Codex owns
  orchestration, acceptance criteria, integration, evidence review, safety, and
  final verification unless a task explicitly assigns otherwise.
- The vault never contains credentials, access tokens, private packet payloads,
  or secret values.
- A vault note is not delivery and never bypasses authorization for destructive
  actions, incident disposition, comments, reviews, PRs, merges, releases, or
  other external mutations.
- No `Co-Authored-By:` or AI-attribution trailers in commits.
- Avoid repetitive comments. Add new information or point to the canonical note.

## Project validity rules

- Synthetic held-out accuracy describes the generators, not real-network
  validity. Do not imply real-traffic performance without labelled raw-packet
  acceptance evidence.
- SFAF exact headline results remain withdrawn until the corrected protocol is
  rerun. Historical results under `legacy/` are evidence of what ran, not current
  scientific conclusions.
- The supported IoT-23 loader consumes labelled Zeek flow logs; that does not
  by itself supply a raw pcap for the 22-feature live extractor.
- Dataset presence or a mounted GOAT volume does not authorize downloads,
  reruns, publication, or a scope change.
- Raspberry Pi performance claims require identified Pi hardware and the
  acceptance procedure in `deploy/PI_ACCEPTANCE.md`.
- Deployable artifacts are ONNX or C headers. Do not introduce pickle loading
  into deployment paths.
- Scientific and enforcement gates use explicit runtime errors, not assertions
  removable by `python -O`.
- Current-tree secret cleanup does not clear historical-scanner incidents.

## Execution and verification

- Reconcile vault, Git/GitHub, worktrees, workers, tests, datasets/hardware, and
  protected artifacts before deciding what remains.
- Execute dependency-ordered checklist tranches. Parallelize only independent
  read/review/verification or disjoint implementation work.
- Run focused tests first and the broader hermetic suite in proportion to risk.
  Record exact commands, commit, passed/failed/skipped counts, and residual risk.
- Reconcile the completed tree, checks, external state, TASK record, and open
  decisions before declaring completion.

When changing a durable rule, update `AGENTS.md`, `CLAUDE.md`, and
`vault/Agent-Protocol.md` together. Keep `AGENTS.md` and `CLAUDE.md`
byte-identical to prevent bootstrap drift.
