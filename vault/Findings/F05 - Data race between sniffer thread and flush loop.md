---
title: F05 — Data race between sniffer thread and flush loop
tags: [finding, critical, live-daemon, concurrency, train-serve-skew]
severity: critical
status: fixed
files: ["src/flow_features.py", "src/ids_daemon.py"]
date: 2026-08-19
---

# F05 — Data race between the sniffer thread and the flush loop

## The problem

`run_live()` starts scapy's `AsyncSniffer`, which invokes the `on_pkt` callback
**on its own thread**:

```python
def on_pkt(p):
    pk = normalize_scapy(p)
    if pk is not None:
        table.add_packet(pk, time.time())     # sniffer thread mutates
        stats["pkts"] += 1

sniffer = AsyncSniffer(iface=iface, prn=on_pkt, store=False)
sniffer.start()
while True:
    time.sleep(flush_s)
    flows = table.extract(...)                 # main thread iterates
    ...
    table.prune(older_than=idle_evict, now=now)
```

`add_packet` inserts into `self.flows` (an `OrderedDict`) while
`extract` → `_host_context` and `prune` iterate that same dict. On CPython this
raises:

```
RuntimeError: dictionary changed size during iteration
```

There was no lock anywhere in the file. The failure is intermittent and
load-dependent — it appears precisely when traffic is heavy enough that a
packet lands during the flush window, i.e. exactly the conditions the sensor
exists to handle. `deploy/setup_pi.sh` installs `Restart=on-failure`, so in
production this manifests as a sensor that silently restarts under load and
loses all flow state each time.

## Second defect in the same code path: timestamp source

`add_packet(pk, time.time())` stamps each packet with **the wall-clock time the
callback happened to run**, not the time the packet arrived. Under burst load
the callback lags the wire by a variable amount, so:

- `mean_iat` / `std_iat` reflect scheduler jitter rather than the traffic
- `duration`, and therefore `pkts_per_sec` and `bytes_per_sec`, are distorted

Training uses pcap capture timestamps (`float(p.time)`), so this is a genuine
**train/serve skew** — in a module whose docstring claims *"guarantees zero
train/serve skew."* That claim was true of the feature *code* and false of the
feature *inputs*.

## The fix

**Locking.** `FlowTable` now owns a public re-entrant lock, and every method
that touches `self.flows` — `add_packet`, `extract`, `extract_live`, `prune`,
`__len__` — takes it. `_host_context` and `_rows` document that the caller
holds it. The lock is public and re-entrant so a caller can hold it across a
compound operation without deadlocking on the individual methods.

`prune()` now also returns the evicted keys, so the daemon's verdict cache
(see [[F04 - Live mode reclassifies the entire flow table]]) can drop them
rather than leaking entries for flows that no longer exist.

**Timestamps.** `on_pkt` now uses the capture timestamp:

```python
ts = float(getattr(p, "time", 0.0)) or time.time()
table.add_packet(pk, ts)
if ts > stats["last_ts"]:
    stats["last_ts"] = ts
```

The flush loop and `prune()` now advance on `stats["last_ts"]` (the newest
packet time) rather than wall clock, so eviction is measured in traffic time —
consistent with how the replay path already worked.

## Verification

New regression test `flowtable thread safety` runs four writer threads
hammering `add_packet` while the main thread performs 60 rounds of
`extract` + `extract_live` + `prune`:

```python
threads = [threading.Thread(target=writer, args=(t,), daemon=True) for t in range(4)]
...
for _ in range(60):
    table.extract(min_pkts=1, window=30.0)
    table.extract_live(min_pkts=1, window=30.0)
    table.prune(older_than=5.0, now=2000.0)
assert not errors, f"concurrent access raised: {errors[:3]}"
```

Passes with the lock; this is the shape that reproduces the `RuntimeError`
without it. Full suite: 28 passed, 0 failed.

## Cost

An uncontended `threading.RLock` acquire is on the order of tens of
nanoseconds, against a per-packet parse cost measured in microseconds
(`~248k packets/s` in the benchmark). The overhead is not measurable at this
workload, and the alternative — a lock-free queue handoff — would add latency
and a second buffer for no benefit at these rates.

## Related

[[F04 - Live mode reclassifies the entire flow table]] · [[Architecture]]
