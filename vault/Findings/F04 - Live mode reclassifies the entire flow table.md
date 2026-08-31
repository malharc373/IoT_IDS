---
title: F04 — Live mode reclassifies the entire flow table
tags: [finding, critical, live-daemon, performance]
severity: critical
status: fixed
files: ["src/flow_features.py", "src/ids_daemon.py"]
date: 2026-08-19
---

# F04 — Live mode reclassifies the entire flow table every flush

## The problem

`FlowTable.extract(min_pkts, window)` passed `window` **only** to
`_host_context()`. The loop that actually built feature vectors iterated
`self.flows` unconditionally:

```python
def extract(self, min_pkts=1, window=None):
    ctx = self._host_context(window=window)      # window applied HERE
    out = []
    for key, f in self.flows.items():            # ...and nowhere else
        if f.tot_pkts < min_pkts:
            continue
        ...
```

So on every flush (`--step`, default 2 s) the live loop rebuilt host context and
ran ONNX over **every flow in the table**, including flows far outside the
60-second analysis window and flows already alerted on, right up until the 120-second
idle eviction.

Per-flush cost therefore tracked **table size**, not new traffic. A port scan
across 600 ports produces ~600 flows that are then re-classified on each of
roughly 60 flushes before eviction — ~36,000 inferences for 600 flows of
traffic. On a busy segment holding 20k live flows that is 20k inferences every
2 seconds regardless of whether anything happened.

This is why "6 µs/flow" did not answer the question of whether the Pi keeps up:
the per-flow latency was never the constraint — the redundant multiplier was.

## The fix

Two changes, in `src/flow_features.py`:

**1. `window` now bounds the returned flow set, not just host context.**
Extraction moved into a shared `_rows()` core that applies the same
`latest - f.ts_end > window` cut to the flows it emits.

**2. Per-flow dirty tracking + a verdict cache.** `Flow` gained a `dirty` flag
(set on every `update()`) and a new `FlowTable.extract_live()` returns
`(key, meta, vector, needs_scoring)` and clears the flag. `src/ids_daemon.py`
gained a `VerdictCache` that runs the model only on flows needing scoring and
serves the rest from cache.

Aggregation still sees a verdict for *every* windowed flow, so incident flow
counts and the `flows >= prev * 1.5` growth heuristic behave exactly as before.

### The subtlety that made the first attempt wrong

Three of the 22 features are **host-context** (`host_dst_ports`,
`host_dst_ips`, `host_flow_count`). They are computed across a source's *other*
flows — so a flow's feature vector can change when its **peers** change, even
though the flow itself gained no packets. A naive dirty flag therefore serves
stale verdicts.

The regression test caught this immediately:

```
FAILED: verdict cache == full scoring: AssertionError('cached verdicts diverged')
```

Invalidation is now exact. Each `Flow` records the host-context triple it was
last scored with (`ctx_sig`), and re-scores if either the flow changed **or**
its context changed:

```python
sig = (host["dst_ports"], host["dst_ips"], host["flow_count"])
needs = f.dirty or f.ctx_sig != sig
```

A flow is re-scored **iff** its feature vector could have changed. No
approximation, no tunable threshold.

## Verification

`demo_mixed.pcap` replay, detection output byte-identical to before
(portscan 593 dst-ports, udpflood 1,480 flows, 36 incidents):

```
[cache] scored=16,546 reused=266,146 (94% cached)
```

**94% of model invocations eliminated** with exact-equal results.

Note this is close to a worst case: `demo_mixed.pcap` is almost entirely active
attack traffic, where sources continuously change their own host context. On a
real segment — mostly completed benign sessions sitting idle — the reuse rate
is higher still.

A single-source scan is the honest exception: every new flow changes that
source's `host_flow_count`, so all its sibling flows genuinely need re-scoring.
That is correct, and the test asserts it rather than papering over it.

Regression tests added:
- `flowtable window + dirty` — window bounds the flow set; dirty flags set,
  cleared, and untouched by the batch `extract()` path
- `verdict cache == full scoring` — incremental verdicts and aggregated
  incidents are identical to scoring everything from scratch, and a flush with
  no new packets scores nothing

## Related

[[F05 - Data race between sniffer thread and flush loop]] (fixed in the same
pass, same files) · [[Architecture]] · [[Future Work]]
