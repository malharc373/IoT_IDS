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
  commit SHA.

## GitHub settings applied 2026-08-22

Read back through the GitHub API after mutation:

| Control | State |
|---|---|
| Dependabot vulnerability alerts | enabled |
| Dependabot automated security fixes | enabled |
| Private vulnerability reporting | enabled |
| Allowed Actions | GitHub-owned only |
| Full-SHA pinning | required |
| `main` required check | `test`, strict/up-to-date |
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

### Historical binary rewrite

R24 would change public commit IDs and require collaborator coordination. The
legacy artifacts are quarantined in the current tree, but removing their old
blobs from Git storage remains a separate destructive decision.

## Related

[[Remediation 2026-08-22]] · [[Project Overview]] · [[Future Work]]
