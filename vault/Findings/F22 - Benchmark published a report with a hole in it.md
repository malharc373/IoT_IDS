---
title: F22 — Benchmark published a report with a hole in it
tags: [finding, significant, benchmark, silent-failure, found-during-review]
severity: significant
status: fixed
files: ["demo/benchmark.py", "ruff.toml", ".github/workflows/ci.yml"]
date: 2026-08-21
---

# F22 — The benchmark published a report with a hole in it

> [!quote] From [[Remediation Log]], written the day before this was found
> "Three of the five critical findings are the same failure mode: **a step that
> could silently produce a wrong artifact, with a check that either did not
> exist or could not fail the build.**"

The benchmark was doing it too, and the evidence was sitting in a committed
file the whole time.

## Discovery

A linter found it, not a test:

```
demo/benchmark.py:187:29: F821 Undefined name `orig_len`
```

And the published `demo/results/BENCHMARK.md`, committed the previous day, read:

```
====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  (section skipped — ValueError: too many values to unpack (expected 2))
```

## Root cause

The snaplen fix in [[F16 - Moderate issues roundup]] gave `read_pcap` a third
return value:

```python
def read_pcap(filename) -> List[tuple]:
    """Return a list of (ts, raw_bytes, orig_len) from a capture file."""
```

`bench_extract` was never updated and still unpacked two:

```python
for ts, raw in recs:              # ValueError
    pk = parse_raw(raw, orig_len) # ...and orig_len was never defined
```

Both errors are in two adjacent lines, and neither was reachable, because the
`ValueError` fired first.

## Why nothing caught it

`_guard` treated every exception as a legitimate skip:

```python
except ImportError as e:
    out(f"  (section skipped — missing dependency: {e.name or e})")
except Exception as e:
    out(f"  (section skipped — {type(e).__name__}: {e})")
```

The intent is reasonable — a section that needs an optional dependency should
not abort the run. But it rendered a *programming error* in the same words as
an environmental one, exit code 0, and wrote the result to a published report.

The existing test called `bench_params()` only, so section 4 was never
exercised. And the consequence was not confined to section 4: **section 8's
Raspberry Pi projection depends on it**, so the benchmark's headline verdict —
*"sniffing/aggregation, not inference, is the limit"* — had lost the number it
rests on. The restored figure is ~18,246 packets/s projected on a Pi 4.

## A second, quieter instance in the same file

Two lines of the accuracy section were hardcoded:

```python
out("        for the honest cross-dataset numbers: in-domain ROC-AUC 0.996 vs")
out("        cross-domain 0.514 against a chance baseline of 0.500 (MCC -0.002).")
```

Those are the **ten**-dataset numbers. After the eleven-dataset rerun
([[EXP02 - Corrected alignment rerun]]) the correct figures are 0.995 / 0.509 /
−0.007. A generated report was quoting a superseded result — the failure the
benchmark exists to prevent, inside the benchmark.

Now derived from `cross_dataset_metrics_long.csv` at run time, including the
dataset and pair counts, so it cannot go stale again.

## Fix

- Unpack the third value; delete the undefined name.
- `_guard` keeps `ImportError` as a skip. Everything else is recorded in
  `FAILED_SECTIONS`, printed as `(section FAILED — ...)`, and `main()` exits
  non-zero. The partial report is still written — it is useful — but the run
  is unambiguously a failure.
- Cross-dataset figures read from the results CSV.

## Guards added

- `benchmark extraction section runs` — calls `bench_extract()` directly,
  asserts it returns real numbers and that `FAILED_SECTIONS` is empty.
- CI runs `python demo/benchmark.py` after the demo (which generates the pcap
  it needs) and greps the published report for `section FAILED`.
- `ruff check .` on every push, restricted to pyflakes (`F`) rules. This
  finding is the argument for it: `F821` is not a style preference, it is a
  guaranteed `NameError` on a reachable line.

## The lesson, restated

The remediation log already knew the rule — *make the check able to fail, not
merely able to print*. A catch-all `except` is that anti-pattern in its purest
form: it converts every unknown failure into a known-looking one. The word
"skipped" did the damage; "FAILED" costs the same and means something.

## Related

[[F16 - Moderate issues roundup]] · [[F18 - Pipeline ONNX export silently ships a broken model]] ·
[[EXP02 - Corrected alignment rerun]] · [[Review 2026-08-21]]
