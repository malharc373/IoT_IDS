# Model artifact manifest

Only the live 22-feature synthetic model is currently published here.

## Live model card

### Intended use

`live_ids.onnx` is an experimental 10-class flow classifier for replay,
integration testing, and authorized edge-sensor evaluation. `live_ids.h` is a
decision-equivalent C export for MCU feasibility work. Neither artifact is a
production security control or evidence of accuracy on real traffic.

### Training and evaluation

The model is XGBoost trained on raw 22-feature flow vectors produced from the
synthetic scenario generator. The split is grouped by scenario. Current
metadata reports 44,761 training rows and 12,191 held-out rows across 280 and
70 scenarios respectively. The separately generated unseen-seed benchmark
reports 99.65% multiclass accuracy and 0.9961 macro F1, including 94.2% Mirai
recall and 0% benign false-positive rate. These values describe generator
separability, not real-network effectiveness.

### Inputs and outputs

Input is one or more float32 vectors in the exact order and semantics declared
by `live_meta.json` and `src/flow_features.py` contract version 2. Output is a
closed-set class and uncalibrated confidence. A novel attack is forced into an
existing class; confidence must not be read as an empirical probability.

### Limitations and risks

- no labelled real-traffic evaluation;
- no calibrated operating threshold or unknown-class abstention;
- host-context features may shift with placement and observation window;
- Raspberry Pi throughput/soak and MCU on-device feature extraction are not
  measured; and
- enforcement can disrupt a network, so the daemon defaults to non-enforcing
  behavior and requires topology-aware configuration.

### Artifact integrity

Training verifies ONNX/sklearn agreement before publication.
`python src/export_c.py --verify` checks C/booster parity. The daemon validates purpose, runtime
compatibility, feature names, and contract version before loading the model.
The repository does not currently publish signed artifacts or a release
checksum manifest.

## Artifact manifest

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
