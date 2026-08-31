# Cross-dataset study — rerun required

The current research harness is ready, but the eleven source datasets are not
mounted on this machine. No current headline number is published until the
corrected experiment can be rerun.

## Why the prior result was withdrawn

The prior off-diagonal measurement was a genuine cross-dataset test and found
mean ROC-AUC 0.509 over 110 ordered pairs. However, the diagonal trained and
evaluated on the same rows. Its reported 0.995 ROC-AUC was resubstitution, so
the claimed in-domain/cross-domain gap of 0.487 was invalid.

The 2026-08-22 audit also found four protocol issues that can change exact
numbers:

- rows with a parse failure in any claimed feature were silently deleted;
- Bot-IoT was fully loaded before sampling, so the cap did not bound memory;
- IoT-23 derived caches did not include loader/contract code in invalidation;
- small-budget threshold-transfer means skipped single-class samples and were
  conditional on lucky calibration draws.

All affected outputs are preserved under
[`legacy/resubstitution-results/`](../../legacy/resubstitution-results/) with a
retraction notice. They are historical evidence, not current results.

## Correct protocol now implemented

`code/cross_dataset_eval.py` creates one deterministic stratified 80/20 split
per dataset. Every model is fit only on its 80% partition. Its diagonal cell is
evaluated on the untouched 20%; off-diagonal cells remain evaluations on
independent target datasets. The long-form output records `evaluation`,
`n_train`, and `n_eval` for every cell.

The alignment layer retains parse failures as NaN and attaches a row-quality
report. Bot-IoT uses bounded-memory reservoir sampling. IoT-23 cache names carry
a digest of the loader source. Threshold calibration evaluates every repeat,
falling back to the deployed 0.5 threshold on single-class samples and reporting
the calibration success rate.

## Reproduction gate

Mount the dataset root, then run:

```bash
python code/cross_dataset_eval.py --cap 50000
python code/transfer_experiment.py --cap 25000
python code/threshold_transfer.py --cap 50000 --repeats 20
```

Accept the new results only if:

1. all eleven expected datasets are present and both classes are reported;
2. every diagonal row says `evaluation=held_out_20pct`;
3. `n_train` and `n_eval` are nonzero and disjoint by construction;
4. loader quality reports are retained with the run manifest;
5. threshold rows report all requested repeats and their calibration success
   rates; and
6. the README, vault experiment note, benchmark, report, and presentation are
   updated from the generated artifacts in the same commit.

Until then, the defensible conclusion is qualitative: the previous off-domain
run showed severe domain shift, but its exact value and the size of the
in-domain/cross-domain gap are pending a protocol-correct rerun.
