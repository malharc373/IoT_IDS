---
title: F21 — Dataset entry point dies on an unmounted drive
tags: [finding, moderate, datasets, tests, hermeticity, found-during-review]
severity: moderate
status: fixed
files: ["code/download_datasets.py", "tests/smoke_test.py"]
date: 2026-08-21
---

# F21 — The documented entry point dies when the drive is unplugged

## Discovery

Running the suite with the GOAT external drive unmounted:

```
FAILED tests/test_suite.py::test_check[download_datasets runs]
FileExistsError: [Errno 17] File exists: '/Users/malharfalke/IOT-IDS/Datasets'
```

With the drive mounted, 40/40 pass. The same test, the same commit, two
different answers.

## Two problems, one symptom

**1. `exist_ok=True` does not cover a dangling symlink.** `Datasets/` is a
symlink to the external volume (see [[Dataset Notes]]). When that volume is not
mounted the link dangles, and:

```python
os.makedirs(DS, exist_ok=True)
```

still raises — `exist_ok` suppresses the error only when the path is an
existing **directory**, and a broken symlink is not one. So the first command
in the README's reproduction section died with a bare traceback, at the exact
moment when the useful message would have been *"plug in the drive"*.

**2. The test was not hermetic.** It inherited the ambient path, so its result
depended on whether a USB volume happened to be mounted. The original review
flagged this class of problem and it was recorded as fixed for the SFAF trainer
guard — but the dataset-fetch check was left reading the real tree.

## Fix

`ensure_dataset_root()` distinguishes the three cases and says which one it
hit:

```
[ERROR] .../Datasets is a symlink to /Volumes/GOAT/..., which is not mounted.
        Mount that volume, or point the pipeline elsewhere with
        IOTIDS_DATASETS_ROOT=/path/to/datasets.
```

`IOTIDS_DATASETS_ROOT` now overrides the location, which is what makes the test
hermetic: it points at a temp directory instead of the real one.

## A second bug, created by the first fix

Pointing the test at an empty root made it *pass* — and take **375 seconds**
instead of 14. An empty root fails the layout check, so the script proceeded to
its next step and started downloading datasets from Kaggle. The "hermetic" test
had become a network test.

Fixed properly with a `--check-only` flag that reports what is present and
exits without fetching anything. Back to 12 seconds.

> [!warning] Worth stating plainly
> The first fix made the test green while making it worse. Green was not the
> signal — the runtime was. A test that got 25× slower changed what it was
> testing.

## Regression tests

- `download datasets runs` — hermetic, `--check-only`, asserts it reports the
  missing datasets.
- `download datasets dangling symlink` — a dangling root must exit non-zero,
  must **not** print a traceback, and must name both the unmounted target and
  the `IOTIDS_DATASETS_ROOT` escape hatch.

CI additionally asserts that no tracked Python file contains a `/Volumes/` path
and that `Datasets/` does not exist in the checkout, so the suite cannot drift
back to depending on the author's hardware.

## Related

[[Dataset Notes]] · [[F17 - Documentation inconsistencies]] · [[Review 2026-08-21]]
