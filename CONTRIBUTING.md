# Contributing

Changes should preserve the evidence boundaries documented in the README and
the Obsidian vault. In particular, do not describe synthetic accuracy, host
benchmarks, or withdrawn cross-dataset results as deployment evidence.

## Development check

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt  # macOS/default
# Ubuntu x86_64: requirements-dev-linux-x86_64.txt
ruff check .
pytest tests/ -q
```

For changes to generated deployment artifacts, also run:

```bash
python src/train_live_model.py
python src/export_c.py --verify
bash demo/run_demo.sh
python demo/benchmark.py
```

Commit the ONNX model, C header, metadata, and current benchmark together when
their contract or training data changes. Do not commit datasets, packet
captures, credentials, pickles, or private logs.

## Pull requests

Use a focused branch and explain the failure mode, evidence, and verification.
CI must pass. Changes to feature semantics require a contract-version bump and
model regeneration. Changes to research evaluation must preserve held-out
diagonals and keep old results quarantined rather than silently overwriting
their provenance.

No contribution can be accepted under an open-source license until the owner
records a license decision. In the meantime, copyright remains with each
contributor and no license is implied.
