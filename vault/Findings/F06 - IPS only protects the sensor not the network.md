---
title: F06 — IPS only protects the sensor, not the network
tags: [finding, significant, ips, deployment]
severity: significant
status: fixed
files: ["src/ips_response.py", "src/ids_daemon.py"]
date: 2026-08-19
---

# F06 — The IPS only protected the sensor, not the network

## The problem

Both firewall backends installed rules on the **INPUT** path only:

```python
self._run(["nft", "add", "chain", "inet", NFT_TABLE, "input",
           "{ type filter hook input priority -1; }"])
...
self._run(["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"])
```

INPUT matches traffic **addressed to this host**. But the deployment the README
describes is a Raspberry Pi watching an IoT segment — either passively on a
SPAN/mirror port, or inline as a bridge/gateway the devices route through. In
both cases the attack traffic that matters is aimed at the *IoT devices*, not at
the Pi, so it never traverses INPUT and was never affected.

The sensor would print `⛔ IPS blocked 203.0.113.7` and change nothing about the
attack in progress. For a passive sensor there is no rule that *could* help —
the traffic does not pass through the box at all.

## The fix

Made the deployment topology an explicit, required choice rather than an
unstated assumption. `Responder(scope=...)`, exposed as `--ips-scope`:

| scope | chains / hooks | protects | correct when |
|---|---|---|---|
| `host` (default) | INPUT | this sensor only | passive sensor on a mirror port |
| `network` | INPUT + FORWARD | sensor + forwarded traffic | sensor is inline (bridge/gateway) |

`status()` now reports `scope` and a plain-English `protects` field, and
constructing a `host`-scope responder logs the limitation up front:

```
[IPS] scope=host: rules apply to traffic addressed to this sensor only.
      Attacks on other devices are NOT stopped — use scope='network'
      (--ips-scope network) when the sensor is inline.
```

Only `network` scope can actually stop an attack on a third-party device, and
now the operator is told which one is live instead of inferring it.

## Verification

Test `ips scope + nft idempotence` asserts the generated ruleset matches the
scope:

```python
assert "hook input" in h and "hook forward" not in h   # host
assert "hook input" in n and "hook forward" in n       # network
assert net.status()["protects"] == "this sensor + forwarded traffic"
```

## Still open

Bridge/gateway configuration on the Pi itself (`br0`, `net.ipv4.ip_forward`) is
a deployment step, not code — it belongs in `deploy/README_PI.md`. See
[[Future Work]].

## Related

[[F07 - nftables setup is not idempotent]] ·
[[F08 - Rate limiting is documented but not implemented]] ·
[[F09 - IPS gate uses uncalibrated confidence]]
