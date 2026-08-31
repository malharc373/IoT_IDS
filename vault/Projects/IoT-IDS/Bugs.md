---
tags: [bugs, traps, hub]
status: active
---

# Bugs and traps — IoT-IDS

## Durable traps

- The live model uses 22 packet-derived features; SFAF uses 12 aligned dataset
  features. Never load one artifact into the other path.
- Synthetic scenario accuracy is not evidence of real-traffic performance.
- Supported IoT-23 input is labelled Zeek flow logs, not automatically a raw
  pcap suitable for the live extractor.
- A mounted dataset drive does not authorize a rerun or publication claim.
- `python -O` removes assertions; scientific/deployment gates use explicit errors.
- A green current tree does not clear secret-scanner findings in historical commits.

Detailed resolved findings remain indexed in [[Home]]. Append new traps here
with provenance and a link to their event/finding note.
