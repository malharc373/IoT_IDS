# Git history cleanup plan

Status: **not executed; coordinated destructive decision required**.

## Measured problem

The repository's current pack is 41.69 MiB. The largest historical blobs are
obsolete model pickles and JSON exports:

| Path | Largest blob |
|---|---:|
| `models/random_forest_v1.pkl` | 56,438,465 bytes |
| `models/rf_multiclass.pkl` | 32,597,177 bytes |
| `models/rf_baseline.pkl` | 13,145,897 bytes |
| `models/rf_reduced_top20.pkl` | 7,405,033 bytes |
| `models/xgb_unified.json` | 3,999,140 bytes |
| `models/xgboost_smote_v1.pkl` | 3,770,949 bytes |
| `models/xgboost_v1.pkl` | 3,395,769 bytes |

Twelve third-party papers account for additional historical/current blobs; the
largest is `Literature/XGBoost_chapter.pdf` at 3,473,019 bytes. They are removed
from the current index but remain reachable from earlier commits until a purge.

## Proposed purge scope

- every historical `*.pkl`;
- obsolete historical `models/xgb_unified.json` versions; and
- `Literature/*.pdf` after confirming the bibliography is sufficient.

Do **not** purge the current deployable ONNX/C artifacts, corrected report/deck,
vault evidence, or quarantined result metadata merely to reduce size.

## Preconditions

1. Merge/close active work and announce a freeze to every collaborator.
2. Create a full bare mirror backup and verify all refs/tags are present.
3. Record the old default-branch and tag commit IDs outside the repository.
4. Install and version-pin `git-filter-repo` from its official distribution.
5. Test the rewrite in a disposable mirror, then run tests and inspect retained
   artifacts before changing the public remote.
6. Obtain explicit owner approval for force-pushing every affected ref.

## Candidate command (disposable mirror only)

```bash
git filter-repo \
  --path-glob '*.pkl' \
  --path-glob 'Literature/*.pdf' \
  --path 'models/xgb_unified.json' \
  --invert-paths
```

After verification, force-push explicit branches and tags using
`--force-with-lease` against recorded remote tips. Never run a broad force push
from the working repository. All collaborators must re-clone; old clones can
reintroduce purged objects.

## Acceptance evidence

- before/after `git count-objects -vH` and largest-blob inventory;
- clean fresh clone from the rewritten remote;
- full CI green at the rewritten default branch;
- current ONNX/C/report/deck hashes unchanged;
- GitGuardian rescans the intended commit range; and
- collaborator acknowledgement that old clones will not be pushed.
