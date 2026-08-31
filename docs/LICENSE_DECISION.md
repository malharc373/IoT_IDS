# License decision record

Status: **owner consent required; no license currently granted**.

## Scope that can be licensed

The Git history currently identifies one committer/author, while the corrected
academic report names four student authors. Before licensing, confirm who owns
the original source code, generated model, report, presentation, and artwork,
and obtain consent from every relevant rightsholder.

Third-party datasets retain their publishers' terms and are not redistributed.
Third-party research PDFs have been removed from the current index and are
explicitly excluded. Dependency licenses are not replaced by the repository's
license.

## Recommended structure

1. **Apache License 2.0** for original source code, tests, deployment scripts,
   and the generated model artifacts, subject to author consent. It is
   permissive like MIT but adds an explicit patent grant and patent-termination
   terms, useful for an ML/security project.
2. **CC BY 4.0** for the original report, presentation, diagrams, and prose if
   all four named student authors agree. This makes academic reuse/attribution
   explicit without pretending software terms are ideal for a report.
3. A `NOTICE`/third-party inventory that states datasets, papers, package code,
   and publisher assets are not relicensed.

If the owner prioritizes the simplest single permissive software license, MIT
is a reasonable alternative for original code/model artifacts, but it lacks
Apache-2.0's explicit patent terms. GPL-3.0 should be chosen only if reciprocal
licensing of downstream derivative software is intentional.

## Decision checklist

- [ ] Confirm code/model copyright owner(s).
- [ ] Obtain consent from all report/deck authors or exclude those works.
- [ ] Choose a deliberate software license.
- [ ] Choose documentation terms separately.
- [ ] Add license files and per-directory scope statements.
- [ ] Review every non-original asset and dependency notice.
- [ ] Update `README.md`, `CONTRIBUTING.md`, model card, and release metadata.
