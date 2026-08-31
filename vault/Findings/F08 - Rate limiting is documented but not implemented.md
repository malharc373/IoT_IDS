---
title: F08 — Rate limiting is documented but not implemented
tags: [finding, significant, ips, documentation]
severity: significant
status: fixed
files: ["src/ips_response.py", "README.md"]
date: 2026-08-19
---

# F08 — Rate limiting was documented but not implemented

## The problem

Three places promised a graduated response:

- `src/ips_response.py` docstring: *"can actively block **or rate-limit** the
  offending source"*
- `README.md`, describing the IPS layer as block/rate-limit
- The module defined `NFT_SET = "blocked"` — the only set that existed

The implementation had exactly one action:

```python
def _apply_block(self, ip):
    if self.backend == "nftables":
        self._run(["nft", "add", "element", ..., NFT_SET, "{ %s timeout %ds }" % ...])
    elif self.backend == "iptables":
        self._run(["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"])
```

Blackhole or nothing. That is the worst possible response curve for a detector
whose confidence is not calibrated
([[F09 - IPS gate uses uncalibrated confidence]]): the only available action was
also the most destructive one, so every false positive cost the maximum.

## The fix

Implemented the missing tier, and made it the *default first* response.

**nftables** — a second timeout set consulted before the drop set, using nft's
native rate limiter:

```
set throttled { type ipv4_addr; flags timeout; }
chain input {
  type filter hook input priority -1; policy accept;
  ip saddr @throttled limit rate over 20/second drop
  ip saddr @blocked drop
}
```

`limit rate over N/second drop` drops only the *excess*, so a throttled source
keeps working at a reduced rate instead of vanishing.

**iptables** — the equivalent accept-under-limit / drop-the-rest pair, inside
the private `IOT_IDS` chain from
[[F07 - nftables setup is not idempotent]]:

```python
iptables -A IOT_IDS -s <ip> -m limit --limit 20/second --limit-burst 20 -j RETURN
iptables -A IOT_IDS -s <ip> -j DROP
```

Rate is configurable via `--ips-throttle-pps` (default 20). Throttles carry the
same auto-expiry and state persistence as blocks, and are lifted automatically
when a source escalates to a full block.

## Response ladder

```
conf < min_conf                   -> monitor    (nothing)
conf >= min_conf, strikes < N     -> throttle   (rate-limited to 20 pps)
conf >= min_conf, strikes >= N    -> block      (drop, auto-expiring)
allowlisted / max-active          -> skip
```

Observed on `demo_mixed.pcap`:

```
~ IPS would-throttle 192.168.1.4     (mqtt_flood, conf=1.00) to 20 pps [strike 1/3]
~ IPS would-throttle 192.168.1.221   (xmas_scan,  conf=1.00) to 20 pps [strike 1/3]
⛔ IPS would-block   192.168.1.4     (mqtt_flood, conf=1.00) for 300s [strikes=3/3]
```

## Related

[[F09 - IPS gate uses uncalibrated confidence]] ·
[[F07 - nftables setup is not idempotent]] ·
[[F17 - Documentation inconsistencies]]
