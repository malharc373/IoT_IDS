---
title: Tracked Jupyter autosave checkpoint
tags: [finding, repository-hygiene, notebooks, fixed]
date: 2026-08-31
status: fixed
---

# F25 — Tracked Jupyter autosave checkpoint

## Finding

The final non-dataset repository sweep found
`code/.ipynb_checkpoints/01_Dataset_EDA -checkpoint.ipynb` in Git. The file was
only a 72-byte empty notebook with zero cells and zero outputs. Its directory is
already excluded by `.gitignore`, but ignore rules do not remove files committed
before the rule was added.

The filename also contained a non-breaking space, making it awkward to see and
address from ordinary shell output. It was workspace debris, not a historical
research artifact; the actual pre-remediation notebooks remain intentionally
preserved under `legacy/notebooks/`.

## Resolution

- Removed the empty autosave checkpoint from the current tree.
- Added a hermetic regression over `git ls-files -z` that rejects tracked
  Jupyter checkpoints, Python/test/linter caches, `.DS_Store`, bytecode, swap
  files, and editor backup files.

## Verification

The regression operates only on the Git index and does not read any dataset.
Full verification is recorded in [[Remediation 2026-08-22]].

## Related

[[F23 - Retracted thesis numbers still shipped as current]] · [[Repository Governance]]
