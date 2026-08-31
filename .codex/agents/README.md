# IoT-IDS Codex subagents

All agents follow `AGENTS.md`, `vault/Agent-Protocol.md`, and the parent TASK
record. Invoke them only for bounded, independently useful work.

- `code_mapper`: read-only execution/contract mapping
- `reviewer`: read-only correctness/security review
- `python_pro`: scoped Python implementation
- `test_automator`: scoped hermetic regression tests
- `ml_researcher`: read-only scientific-method and leakage audit
- `security_auditor`: read-only threat/security review
- `edge_deployment`: scoped Pi/MCU/deployment implementation
- `docs_researcher`: read-only primary-source verification

Read-only agents never change local or external state. Write-capable agents edit
only assigned files, preserve other work, and never commit, push, post, merge,
change worktrees, or perform external writes. No agent recursively delegates
without explicit permission.
