---
title: Independent security hardening follow-up
tags: [finding, security, reliability, verified]
date: 2026-08-22
status: fixed
---

# F24 — Independent security hardening follow-up

## Finding

A fresh dataset-independent audit after the main remediation found three
defence-in-depth gaps not covered by the prior 61 checks:

1. `src/train_live_model.py` used an `assert` for the scenario-disjoint split
   gate. `python -O` removes assertions, even though this gate protects the
   validity of published evaluation evidence.
2. `src/ips_response.py` overwrote `ips_state.json` directly and swallowed all
   load/save failures. A crash could truncate the file, and an enforcement
   state that was not durable could disagree with the firewall after restart.
3. `code/download_datasets.py` used broad archive `extractall` calls after
   validation and invoked PATH-resolved tools by bare name. The validation was
   already strong, but member-by-member atomic writes and stable executable
   resolution reduce the remaining extraction and PATH-race surface.

## Resolution

- The split overlap is now an explicit `ValueError`, active under optimized
  Python.
- IPS state is written to a same-directory temporary file, flushed, fsynced,
  and atomically replaced. Failures return `False` and are logged; unreadable
  existing state is also reported instead of silently discarded.
- Tar and ZIP files are extracted member by member through atomic temporary
  files. Traversal, links, special files, and pre-existing escaping symlinks
  remain fail-closed.
- Kaggle and wget paths are resolved before invocation.

## Verification

```text
pytest tests/ -q                         -> 63 passed
ruff check .                             -> all checks passed
compileall + Bash syntax + diff check    -> passed
pip-audit requirements.txt               -> no known vulnerabilities
pip-audit deploy/requirements-pi.txt     -> no known vulnerabilities
Bandit                                   -> 0 high-severity findings;
                                             remaining medium reports are the
                                             authenticated all-interface option
                                             detected as string literals
GitHub CodeQL Python setup              -> run 32567799399 passed
```

Dataset loading and scientific reruns were not performed because `/Volumes/GOAT`
was not mounted. `download_datasets.py --check-only` failed with the intended,
actionable mount message rather than a traceback.

## Related

[[Remediation 2026-08-22]] · [[Repository Governance]]
