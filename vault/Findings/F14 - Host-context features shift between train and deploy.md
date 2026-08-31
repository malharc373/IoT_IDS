---
title: F14 — Host-context features shift between train and deploy
tags: [finding, significant, live-model, train-serve-skew]
severity: significant
status: fixed
files: ["attacks/build_corpus.py"]
date: 2026-08-19
---

# F14 — Host-context features shift between train and deploy

## The problem

Three of the 22 features are **host context**, computed across all of a
source's flows in the window:

```
host_dst_ports    distinct dst ports this src hit
host_dst_ips      distinct dst IPs this src hit
host_flow_count   flows opened by this src
```

They are the features doing the heaviest discriminative work — the README says
so explicitly: *"Host-context is what separates a port scan (many ports, one
host) from a Mirai spread (one port, many hosts) from a flood."*

But `build_corpus.py` generated every scenario **in isolation**:

```python
pkts = tg.generate(kind, seed=1000 * tg.LABELS[kind] + s)
for meta, vec in _flows_from_packets(pkts):     # only this attack's packets
```

So those three features were measured against an **empty** background. At
deployment the same source's flows are interleaved with all the benign traffic
on the segment, and every one of the three counters shifts. The features
carrying the most weight were the ones whose distribution differed most between
training and serving.

## The fix

Attack scenarios are now generated **mixed with benign background traffic** by
default. Each scenario overlays a benign variant drawn from a far-away seed
range (`BG_SEED_BASE = 600_000`) so the background never reuses the attacker's
addresses, builds one combined `FlowTable`, and labels each flow by **which
packet set produced it**:

```python
atk_keys, bg_keys = _keys_of(atk), _keys_of(bg)
ambiguous = atk_keys & bg_keys
...
for key, _meta, vec, _ in table.extract_live(min_pkts=1, window=None):
    if key in ambiguous:
        continue                       # dropped, never guessed at
    rows.append((vec, kind if key in atk_keys else "benign"))
```

Labelling by flow key rather than by source IP is what makes this safe: a flow
belongs to whichever generator actually produced it. Keys drawn by both
generators by chance are **dropped rather than guessed** — they are rare, and a
wrong label is worse than a missing row.

`--no-mix-background` restores the old isolated behaviour for comparison.

The mixing also yields free benign examples in realistic proximity to attacks,
which is why the benign class grew without extra scenarios.

## Verification

```
Total flows: 153,355  |  features: 22  |  scenarios: 420
Background mixing ON — attack scenarios carry benign traffic so host-context
features are measured against a realistic backdrop
```

Detection quality is unchanged (multiclass accuracy 0.9999 on a scenario-level
split), which — combined with
[[F13 - Live model in-domain metrics are leaky]] — is itself the finding: the
synthetic task is easy enough that even this correction does not move the
number. The corpus is now *right* rather than *lucky*, and the remaining
limitation is squarely that it is synthetic.

## Related

[[F13 - Live model in-domain metrics are leaky]] ·
[[F15 - Attack classes separable by dst_port alone]] ·
[[F05 - Data race between sniffer thread and flush loop]] (the other
train/serve skew, in timestamps) · [[Feature Spaces]]
