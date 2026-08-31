---
title: F07 — nftables setup is not idempotent
tags: [finding, significant, ips, reliability]
severity: significant
status: fixed
files: ["src/ips_response.py"]
date: 2026-08-19
---

# F07 — nftables setup is not idempotent

## The problem

`_ensure_backend()` was described as "idempotent setup of a dedicated
table/set/chain", but the last call was not:

```python
self._run(["nft", "add", "table", "inet", NFT_TABLE])          # idempotent
self._run(["nft", "add", "set",   "inet", NFT_TABLE, NFT_SET, ...])  # idempotent
self._run(["nft", "add", "chain", "inet", NFT_TABLE, "input", ...])  # idempotent
self._run(["nft", "add", "rule",  "inet", NFT_TABLE, "input",
           "ip", "saddr", f"@{NFT_SET}", "drop"])              # APPENDS
```

`nft add rule` appends unconditionally. Every daemon start added another
identical `drop` rule to the chain.

That matters because `deploy/setup_pi.sh` installs the service with
`Restart=on-failure` / `RestartSec=3`. A crash loop — which
[[F05 - Data race between sniffer thread and flush loop]] made a realistic
prospect — would add a rule every three seconds, growing the ruleset without
bound and degrading packet-path performance on the very device under load.

The iptables path had the mirror problem: `-I INPUT` inserted a fresh rule per
block with no dedicated chain, so the sensor's rules were interleaved with
whatever else the host had in INPUT, and cleanup depended on an exact-match
`-D`.

## The fix

**nftables — declarative, applied after a flush.** The full table is generated
as a ruleset and applied with `nft -f -` after `flush table`, so re-running
converges to exactly one copy regardless of how many times it happens:

```python
self._run(["nft", "add", "table", "inet", NFT_TABLE])
self._run(["nft", "flush", "table", "inet", NFT_TABLE])
self._run(["nft", "-f", "-"], stdin=self._nft_ruleset())
```

Because flushing also clears the `blocked`/`throttled` sets, `_restore_active()`
re-adds the persisted blocks with their **remaining** timeouts, so a restart no
longer silently releases every blocked host.

**iptables — a dedicated chain with guarded jumps.** Rules now live in a
private `IOT_IDS` chain that is created and flushed on start; the jump from
INPUT (and FORWARD, per [[F06 - IPS only protects the sensor not the network]])
is added only after `iptables -C` confirms it is absent:

```python
exists = subprocess.run(["iptables", "-C", chain, "-j", IPT_CHAIN],
                        capture_output=True).returncode == 0
if not exists:
    self._run(["iptables", "-I", chain, "-j", IPT_CHAIN])
```

## Verification

`ips scope + nft idempotence` asserts the ruleset is declarative and
single-copy: `assert n.count("hook forward") == 1`. The construction itself is
what guarantees idempotence — the ruleset is a fixed string applied to a
flushed table, so there is no path by which repetition accumulates.

## Related

[[F06 - IPS only protects the sensor not the network]] ·
[[F05 - Data race between sniffer thread and flush loop]]
