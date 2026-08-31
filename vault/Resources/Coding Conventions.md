# Coding and agent conventions

- Preserve unrelated dirty state; never revert another agent's edits.
- Use `rg`/`rg --files` for search and `apply_patch` for intentional edits.
- Make the smallest coherent change; add behavior-focused regression coverage.
- Verify generated model/data/report artifacts against their contracts.
- Keep live and SFAF feature/model contracts separate.
- State commands and exact outcomes; distinguish skipped from passed.
- No AI/co-author trailers in commits.
- No repetitive comments; point to the canonical rationale.
