---
title: F23 — Retracted thesis numbers still shipped as current
tags: [finding, significant, integrity, documentation, found-during-review]
severity: significant
status: fixed
files: ["legacy/", "README.md", "tests/smoke_test.py"]
date: 2026-08-21
---

# F23 — Retracted numbers still shipped as the numbers to put in the report

## The problem

The 2026-08-19 remediation corrected the SFAF study's conclusions. It did not
remove the artifacts the *old* conclusions were written into. Twenty-five
tracked files dated 2026-08-03 remained in `models/` and `results/`, unmarked
and indistinguishable from current output.

The worst of them is `models/report_numbers.md`. It is titled **"Key Numbers
for Report"**, it is written as instructions to a future self —

```
### Section 5.4 (Unified Model Results) — REPLACE table with:
| CICIDS2017 | 0.9833 | ... |
| UNSW-NB15  | 0.9268 | ... |

### Generalisation gain (Section 5.3):
Baseline CICIDS-only on UNSW: 36.09%
Unified SFAF on UNSW: 92.68%
Gain: +56.59 percentage points
```

— and every figure in it is invalidated by [[F01 - SFAF feature mappings are semantically wrong]]
and [[F02 - Cross-domain F1 includes degenerate classifiers]]. The corrected eleven-dataset study puts cross-domain transfer at
**ROC-AUC 0.509 against a chance baseline of 0.500**, MCC −0.007. The
"+56.59 pp generalisation gain" was, in large part, a measurement of the
misalignment itself.

Nothing in the repo referenced these files — grep found zero inbound links from
any `.md`, `.py`, or `.sh`. That is what made them dangerous rather than
harmless: nothing pointed at them, so nothing contradicted them either, and the
one file explicitly addressed to whoever writes the thesis was the one still
holding the retracted numbers.

The same applied to the three notebooks in `code/`, described in the README as
a "historical record" but sitting alongside the maintained scripts.
`02_SFAF_Unified_Model.ipynb` still contains the retracted mapping verbatim:

```python
'src_port':'Min Packet Length','dst_port':'Max Packet Length',
```

## Fix

A `legacy/` directory with a `README.md` that states plainly that nothing in it
is current, tabulates which finding invalidates which artifact, and points at
the live sources.

- `legacy/pre-remediation-results/` — the 2026-08-03 artifacts, including
  `report_numbers.md`
- `legacy/notebooks/` — the three original notebooks

The notebooks are kept **unmodified**. A corrected notebook that was never
re-run would be a third artifact claiming to be authoritative; the maintained
equivalents are `code/multidataset.py`, `code/02_train_sfaf.py`, and
`code/cross_dataset_eval.py`. `legacy/` is excluded from lint for the same
reason — the value of a frozen record is that it is frozen.

`models/references.md` was moved back out: a bibliography is not a retracted
result.

## Guard

`no retracted numbers in live docs` walks every `.md` outside `legacy/`,
`vault/`, and `Literature/` and fails if the strings `56.59` or `92.68` appear.
Copying a retracted headline back into live prose is now a test failure rather
than an editing accident.

## Why this counts as a finding

The project's credibility rests on having corrected itself in public. Leaving
the superseded numbers in place, unmarked and undated, means an examiner opening
`report_numbers.md` reads the retracted claim as the finding — and the honest
correction becomes indistinguishable from having reported two different results.

Retracting a number is not finished when the new number is computed. It is
finished when the old one can no longer be mistaken for the current one.

## Related

[[F01 - SFAF feature mappings are semantically wrong]] ·
[[F02 - Cross-domain F1 includes degenerate classifiers]] ·
[[F17 - Documentation inconsistencies]] ·
[[EXP01 - Cross-dataset study baseline]] · [[Review 2026-08-21]]
