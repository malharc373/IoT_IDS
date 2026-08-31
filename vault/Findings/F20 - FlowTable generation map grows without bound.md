---
title: F20 — FlowTable generation map grows without bound
tags: [finding, significant, live-daemon, memory, found-during-review]
severity: significant
status: fixed
files: ["src/flow_features.py"]
date: 2026-08-21
---

# F20 — The flow table's generation map never shrinks

> [!note] Introduced by the fix for a different finding.
> The `(5-tuple, generation)` key was added in
> [[F16 - Moderate issues roundup]] so that a 5-tuple reused after a TCP
> teardown yields a second, distinct flow record. The bookkeeping map that
> makes those generation numbers monotonic was never added to `prune()`.

## The claim

`FlowTable.prune()` is the sensor's only memory bound. It evicts idle flows and
cleans `self.flows` and `self.active` — but not `self._gen`, a
`defaultdict(int)` keyed by the **bare** 5-tuple. Nothing ever removes an entry
from it.

On a busy segment every connection carries a fresh ephemeral source port, so
every connection leaves a permanent entry. The daemon is meant to run for weeks
on a Raspberry Pi with 1–4 GB of RAM.

## Evidence

20,000 short-lived connections from rotating source ports, pruning throughout:

```python
t = FlowTable()
for i in range(20000):
    pkt = {...,"src_port": 1024 + (i % 64000), "dst_port": 80, ...}
    t.add_packet(pkt, ts=float(i))
    if i % 100 == 0:
        t.prune(older_than=10.0, now=float(i))
t.prune(older_than=10.0, now=1e9)
```

```
flows  : 0
active : 0
_gen   : 20000     <-- never shrinks
```

Flow count returns to zero. The generation map keeps every key forever, so
memory tracks *connections ever seen* rather than *flows currently live* — the
exact distinction `prune()` exists to enforce. Rough cost is a few hundred
bytes per entry: order 200 MB per million connections, which a moderately busy
link reaches in hours.

This is invisible to every existing test, because the suite asserts on flow
counts and feature values, never on the size of the auxiliary maps.

## Fix

Prune all three maps together. Once a 5-tuple has no live flow *and* the caller
has dropped its cached verdicts — `run_live` already does this via
`cache.forget(table.prune(...))` — restarting its generation counter at 0
cannot collide with anything:

```python
live_bases = {base for base, _gen_no in live}
for base in list(self._gen):
    if base not in live_bases:
        del self._gen[base]
```

## Regression tests

Two, because the fix could plausibly break the behaviour that motivated the
generation counter in the first place:

- `flow table prune bounds memory` — after every flow is evicted, all three
  maps must be empty. Asserts on `_gen` specifically, with a message naming
  the failure mode.
- `flow reuse distinct records` — a SYN after a teardown must still open a
  second record, so the leak fix cannot be "solved" by dropping generations.

## The shape of it

The same shape as the through-line in [[Remediation Log]]: a correctness fix
introduced a second data structure, and only the first one was wired into the
lifecycle that bounds it. A fix's *new state* needs the same review as its new
logic — particularly the state that only misbehaves at a scale no test runs at.

## Related

[[F04 - Live mode reclassifies the entire flow table]] ·
[[F05 - Data race between sniffer thread and flush loop]] ·
[[F16 - Moderate issues roundup]] · [[Architecture]] · [[Review 2026-08-21]]
