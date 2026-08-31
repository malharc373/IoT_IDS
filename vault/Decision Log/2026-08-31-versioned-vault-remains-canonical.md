---
tags: [decision, vault, agents]
date: 2026-08-31
status: decided
decision_owner: Codex within user-requested implementation scope
---

# Decision — keep the versioned in-repo vault canonical

**Context:** Klimaforge uses an external, unversioned vault. IoT-IDS already has
a Git-tracked vault containing the full audit history.

**Decision:** Keep `/Users/malharfalke/IOT-IDS/vault` canonical and require
agents in other worktrees to use that absolute path.

**Why:** This preserves existing links and adds history, review, blame, backup,
and CI validation. Copying to a second vault would create drift; replacing the
tracked directory with an absolute symlink would break clones.

**Trade-off:** Agents sharing a working tree can still collide before commit,
so hub ownership, append-only blocks, unique task files, and re-read-before-write
remain mandatory.

— Codex, 2026-08-31T10:30:00+05:30 `[tree]` `[rec]`

## Addendum — 2026-08-31 11:07 IST, Codex

Stable Obsidian configuration remains versioned, but `workspace*.json` and
`graph.json` are local runtime state. Obsidian rewrote graph zoom/open state
immediately after each verified commit, leaving a permanently dirty shared
worktree. Those files are therefore preserved locally but ignored by Git. This
deliberately improves on the Klimaforge reference vault's collision behavior.

— Codex, 2026-08-31T11:07:00+05:30 `[tree]` `[rec]`
