---
title: Repository Governance
tags: [reference, governance, security, github]
date: 2026-08-22
status: active
---

# Repository Governance

Durable record of governance controls and decisions for
`github.com/malharc373/IoT_IDS`.

## Current repository policy

- `SECURITY.md` directs sensitive reports to GitHub private vulnerability
  reporting and states the prototype's operational limits.
- `CONTRIBUTING.md` requires evidence-scoped claims, hermetic checks, contract
  versioning, and coordinated generated artifacts.
- `models/README.md` is the live model card and separation manifest.
- `docs/DATA_CARD.md` records both data domains, provenance, privacy risk,
  leakage controls, and missing acceptance data.
- Dependabot checks Python and GitHub Actions monthly.
- CI runs once per pull request, on direct `main` pushes, manually, and weekly.
  Workflow permissions are read-only and every action is pinned to a full
  commit SHA. macOS and Ubuntu x86_64 locks are regenerated on matching runners.

## GitHub settings applied 2026-08-22

Read back through the GitHub API after mutation:

| Control | State |
|---|---|
| Dependabot vulnerability alerts | enabled |
| Dependabot automated security fixes | enabled |
| Private vulnerability reporting | enabled |
| Allowed Actions | GitHub-owned only |
| Full-SHA pinning | required |
| `main` required checks | `test` and `macOS lock freshness`, strict/up-to-date |
| Pull request required | yes; zero approvals for the single-owner workflow |
| Admin enforcement | enabled |
| Linear history | required |
| Conversation resolution | required |
| Force pushes / branch deletion | disabled / disabled |

The repository description was changed from an unvalidated Raspberry Pi
deployment claim to an evidence-scoped prototype description.

## Owner decisions still required

### License

No license is declared. That means copyright's default restrictions apply; it
is not legitimate to add MIT, Apache-2.0, GPL, or another grant without the
owner's explicit choice. Before accepting outside contributions or inviting
reuse, choose a license and confirm that the report, generated model, third-party
datasets, and any incorporated assets are compatible with it.

The completed compatibility review and recommended split are in
`docs/LICENSE_DECISION.md`: Apache-2.0 for original software/model artifacts and
CC BY 4.0 for original academic prose are recommended, but only after every
relevant rightsholder consents. Twelve third-party research PDFs were removed
from the current Git index without deleting the researcher's local files;
`Literature/README.md` preserves their integrity manifest and points to the
bibliography. Earlier commits still contain the blobs.

### Historical binary rewrite

R24 would change public commit IDs and require collaborator coordination. The
legacy artifacts are quarantined in the current tree, but removing their old
blobs from Git storage remains a separate destructive decision.
`docs/HISTORY_CLEANUP_PLAN.md` now records the measured 41.69 MiB pack, largest
historical blobs, precise candidate scope, backup/freeze prerequisites, and
post-rewrite acceptance evidence.

### GitGuardian pull-request check

PR #1's current code and native CI are green. GitGuardian still reports incident
`36364668`, a Generic CLI Secret in historical commit `aeeb62c` at
`deploy/README_PI.md`. The current file removed the credential-shaped example
and a regression prevents recurrence. Because the app scans every commit in the
PR, the check remains red until the authenticated incident owner marks this
specific historical placeholder as false positive or the branch is rewritten.
The latter is intentionally not used as a workaround for R24.

## Related

[[Remediation 2026-08-22]] · [[Project Overview]] · [[Future Work]]
