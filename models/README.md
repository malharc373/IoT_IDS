# Model artifact manifest

Only the live 22-feature synthetic model is currently published here.

| artifact | purpose | input contract | evidence | runtime use |
|---|---|---|---|---|
| `live_ids.onnx` | 10-class live IDS | 22 raw features, contract v2 | scenario-held-out synthetic traffic | Pi/macOS daemon |
| `live_ids.h` | compiled form of `live_ids.onnx`'s XGBoost source | same 22 features | C/booster parity verified | MCU |
| `live_meta.json` | machine-readable contract/model card | names, version, labels | states synthetic scope | daemon guard |

The SFAF pipeline produces a different artifact: a **binary, 12-feature research
model** (`xgb_edge.onnx`). It is not an alternate build of `live_ids.onnx`, does
not accept `src/flow_features.py`'s vector, and has no valid role in enforcement.
The last copy and its metadata were moved to
`legacy/resubstitution-results/models/` because the evaluation protocol and
loader policy changed on 2026-08-22.

`code/02_train_sfaf.py` may regenerate the research files after the external
datasets are mounted. Its metadata marks `runtime_compatible: false`; the daemon
requires `purpose: live_multiclass_ids`, the exact 22-feature list, and matching
feature-contract version before it loads an ONNX session.

## Future integration gate

A dual-model sensor is appropriate only after all of these are true:

1. real packet captures can be converted into both feature contracts without
   labels or dataset-specific preprocessing at inference time;
2. the 12-feature model is rerun with held-out diagonals and independent domains;
3. fusion/precedence is specified (binary gate vs multiclass label vs abstain);
4. confidence is calibrated on deployment-like traffic;
5. false-positive impact is measured in a benign long-duration soak test; and
6. IPS enforcement remains disabled for research-only/unknown verdicts until
   the operating threshold is justified.
