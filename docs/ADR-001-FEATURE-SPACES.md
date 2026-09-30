# ADR-001: Keep research and live feature spaces separate

- Status: accepted
- Date: 2026-09-30

## Context

The cross-dataset research pipeline uses twelve features that can be aligned
across heterogeneous public flow datasets. The live sensor uses twenty-two
features extracted from packets and rolling host context. Treating these as one
model contract would either discard useful live signals or fabricate fields
that public datasets do not contain.

## Decision

Keep the feature spaces separate and version both contracts explicitly.

- The research pipeline owns the twelve-feature SFAF alignment contract.
- The live pipeline owns `FEATURE_NAMES` and `FEATURE_CONTRACT_VERSION` in
  `src/flow_features.py`.
- Models must declare which contract they implement. The daemon rejects a
  research model, feature-count mismatch, feature-order mismatch, or contract
  version mismatch.
- Results may be compared as experiments, but weights, thresholds, and claimed
  performance do not transfer between the spaces without a separately
  validated adapter.

## Consequences

This preserves train/serve integrity and honest evidence boundaries at the
cost of maintaining two documented pipelines. Convergence remains possible
later, but requires a labelled packet-to-flow corpus containing both feature
representations, a versioned adapter, grouped holdout evaluation, and explicit
acceptance thresholds.
