#!/usr/bin/env python3
"""
ips_response.py — turn the IDS into an IPS (intrusion *prevention* system).

On a corroborated, high-confidence incident the responder can actively
rate-limit or block the offending source. Safety is the priority:

  * dry-run by default — logs the action it *would* take, changes nothing
  * corroboration gate — N strikes inside a window before enforcing, because
                         the model's softmax score is NOT calibrated
  * graduated response  — throttle first, block on escalation
  * allowlist           — never touch loopback, the sensor's own IP, or
                          operator-supplied IPs/subnets
  * auto-expiry         — blocks lift themselves after a timeout
  * backend auto-detect — nftables (preferred) or iptables on Linux; on other
                          platforms it stays in dry-run (safe no-op)

Enforcement runs real firewall commands only with mode="enforce" AND a working
backend AND root — otherwise it degrades to dry-run rather than failing.


SCOPE: WHAT AN IPS ON THIS BOX CAN ACTUALLY STOP  (vault/Findings/F06)
-----------------------------------------------------------------------
The pre-2026-08 version installed rules on the INPUT path only
(`iptables -I INPUT`, nft `hook input`). That protects **the sensor itself** and
nothing else. If the Pi is a passive sensor on a SPAN/mirror port, or a bridge
that IoT devices route through, attack traffic aimed at those devices never
traverses INPUT and was never affected — while the summary still printed
"blocked".

`scope` now makes this explicit and is required to be a deliberate choice:

    scope="host"     INPUT only. Correct for a passive/monitor-port sensor.
                     Honest about protecting only this box.
    scope="network"  INPUT + FORWARD. Correct when the sensor is inline (a
                     bridge or gateway the protected devices route through).
                     Only this scope can actually stop an attack on a
                     third-party device.

`status()` reports the scope so the operator can see which one is live.


CONFIDENCE IS NOT CALIBRATED  (vault/Findings/F09)
---------------------------------------------------
The gate consumes raw XGBoost softmax output, which is systematically
overconfident — `conf >= 0.9` is not "90% likely correct". For an action as
destructive as blackholing a host, a single overconfident score is not enough
evidence, so enforcement additionally requires `strikes` separate incidents
within `strike_window` seconds. Until then the source is throttled, not blocked.
"""
from __future__ import annotations

import os
import json
import time
import socket
import shutil
import ipaddress
import subprocess
import collections

NFT_TABLE = "iot_ids"
NFT_BLOCK_SET = "blocked"
NFT_THROTTLE_SET = "throttled"
NFT_BLOCK_SET6 = "blocked6"
NFT_THROTTLE_SET6 = "throttled6"
IPT_CHAIN = "IOT_IDS"

SCOPES = ("host", "network")
# hook -> iptables built-in chain to jump from
_SCOPE_CHAINS = {"host": ["INPUT"], "network": ["INPUT", "FORWARD"]}
_SCOPE_HOOKS = {"host": ["input"], "network": ["input", "forward"]}


def _own_ips():
    ips = {"127.0.0.1", "::1"}
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips


def detect_backend():
    """Return 'nftables' | 'iptables' | 'none' for the current host."""
    if os.name != "posix":
        return "none"
    if shutil.which("nft"):
        return "nftables"
    if shutil.which("iptables"):
        return "iptables"
    return "none"


class Responder:
    def __init__(self, mode="dry-run", min_conf=0.9, block_seconds=300,
                 allowlist=None, backend="auto", state_path=None,
                 max_active=2000, logger=None, scope="host",
                 strikes=3, strike_window=120, throttle_pps=20):
        self.mode = mode                      # "dry-run" | "enforce"
        self.min_conf = min_conf
        self.block_seconds = block_seconds
        self.max_active = max_active
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        self.scope = scope
        self.strikes = max(int(strikes), 1)
        self.strike_window = strike_window
        self.throttle_pps = throttle_pps
        self.backend = detect_backend() if backend == "auto" else backend
        self.is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        self.log = logger or (lambda m: print(m))

        self.allow_nets = []
        self.allow_ips = set(_own_ips())
        for entry in (allowlist or []):
            self._add_allow(entry)

        self.state_path = state_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs", "ips_state.json")
        self.active = {}                      # ip -> {"until":ts,"kind":..}
        self.throttled = {}                   # ip -> {"until":ts,"kind":..}
        # ip -> deque of incident timestamps, for the corroboration gate
        self._sightings = collections.defaultdict(collections.deque)
        self._load_state()

        # if we intend to enforce, make sure we actually can; else fall back
        self.effective_enforce = (
            self.mode == "enforce" and self.backend in ("nftables", "iptables")
            and self.is_root)
        if self.mode == "enforce" and not self.effective_enforce:
            self.log(f"[IPS] enforce requested but not possible "
                     f"(backend={self.backend}, root={self.is_root}) -> dry-run")
        if self.effective_enforce:
            self._ensure_backend()
            self._restore_active()
        if self.scope == "host":
            self.log("[IPS] scope=host: rules apply to traffic addressed to this "
                     "sensor only. Attacks on other devices are NOT stopped — "
                     "use scope='network' (--ips-scope network) when the sensor "
                     "is inline.")

    # ── allowlist ─────────────────────────────────────────────────────────────
    def _add_allow(self, entry):
        entry = entry.strip()
        try:
            if "/" in entry:
                self.allow_nets.append(ipaddress.ip_network(entry, strict=False))
            else:
                self.allow_ips.add(entry)
        except ValueError:
            self.log(f"[IPS] ignoring bad allowlist entry: {entry}")

    def _allowed(self, ip):
        if ip in self.allow_ips:
            return True
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return True   # can't parse -> don't touch it
        return any(addr in net for net in self.allow_nets)

    # ── state persistence ─────────────────────────────────────────────────────
    def _load_state(self):
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            now = time.time()
            blocks = data.get("active", data)          # tolerate the old format
            self.active = {ip: v for ip, v in blocks.items() if v["until"] > now}
            self.throttled = {ip: v for ip, v in data.get("throttled", {}).items()
                              if v["until"] > now}
        except Exception:
            self.active = {}
            self.throttled = {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump({"active": self.active, "throttled": self.throttled}, f)
        except Exception:
            pass

    # ── corroboration ─────────────────────────────────────────────────────────
    def _record_sighting(self, ip, now):
        """Append this incident and return how many are inside the window.

        The model's confidence is uncalibrated, so a single high score is not
        sufficient evidence to blackhole a host (vault/Findings/F09).
        """
        q = self._sightings[ip]
        q.append(now)
        cutoff = now - self.strike_window
        while q and q[0] < cutoff:
            q.popleft()
        return len(q)

    # ── main entry point ──────────────────────────────────────────────────────
    def handle(self, ip, kind, confidence):
        """Decide + act on one incident. Returns an action dict.

        Ladder: monitor -> throttle (on first corroborated sighting) -> block
        (once `strikes` sightings land inside `strike_window`).
        """
        if confidence < self.min_conf:
            return {"ip": ip, "action": "monitor", "reason": "below-threshold"}
        if self._allowed(ip):
            return {"ip": ip, "action": "skip", "reason": "allowlisted"}
        now = time.time()
        if ip in self.active and self.active[ip]["until"] > now:
            self.active[ip] = {"until": now + self.block_seconds, "kind": kind}
            self._save_state()
            if self.effective_enforce:
                self._refresh_block(ip)
            return {"ip": ip, "action": "already-blocked", "kind": kind}
        if len(self.active) >= self.max_active:
            return {"ip": ip, "action": "skip", "reason": "max-active"}

        n = self._record_sighting(ip, now)
        if n < self.strikes:
            # graduated response: slow it down while evidence accumulates
            if ip not in self.throttled or self.throttled[ip]["until"] <= now:
                self.throttled[ip] = {"until": now + self.block_seconds,
                                      "kind": kind}
                self._save_state()
                verb = "throttled" if self.effective_enforce else "would-throttle"
                if self.effective_enforce:
                    self._apply_throttle(ip)
                self.log(f"  \033[1;33m~ IPS {verb}\033[0m {ip:<15} "
                         f"({kind}, conf={confidence:.2f}) to "
                         f"{self.throttle_pps} pps "
                         f"[strike {n}/{self.strikes}]")
            return {"ip": ip, "action": "throttle", "kind": kind,
                    "strikes": n, "needed": self.strikes}

        self.active[ip] = {"until": now + self.block_seconds, "kind": kind}
        self.throttled.pop(ip, None)
        self._save_state()
        verb = "blocked" if self.effective_enforce else "would-block"
        if self.effective_enforce:
            self._remove_throttle(ip)
            self._apply_block(ip)
        self.log(f"  \033[1;35m⛔ IPS {verb}\033[0m {ip:<15} "
                 f"({kind}, conf={confidence:.2f}) for {self.block_seconds}s "
                 f"[{'enforce' if self.effective_enforce else 'dry-run'}, "
                 f"scope={self.scope}, strikes={n}/{self.strikes}]")
        return {"ip": ip, "action": verb, "kind": kind,
                "expires_in": self.block_seconds, "strikes": n}

    def expire(self):
        """Lift blocks and throttles whose timeout has passed."""
        now = time.time()
        gone = [ip for ip, v in self.active.items() if v["until"] <= now]
        for ip in gone:
            if self.effective_enforce:
                self._remove_block(ip)
            del self.active[ip]
        thr_gone = [ip for ip, v in self.throttled.items() if v["until"] <= now]
        for ip in thr_gone:
            if self.effective_enforce:
                self._remove_throttle(ip)
            del self.throttled[ip]
        # forget stale corroboration history so strikes don't accumulate forever
        for ip in list(self._sightings):
            q = self._sightings[ip]
            while q and q[0] < now - self.strike_window:
                q.popleft()
            if not q:
                del self._sightings[ip]
        if gone or thr_gone:
            self._save_state()
        return gone

    # ── address family ────────────────────────────────────────────────────────
    @staticmethod
    def _is_v6(ip):
        try:
            return ipaddress.ip_address(ip).version == 6
        except ValueError:
            return False

    def _sets_for(self, ip):
        """(block_set, throttle_set, iptables_binary) for this address family.

        The flow extractor parses IPv6, so the responder has to be able to act
        on an IPv6 source. The pre-2026-08 version had an ipv4_addr-only nft set
        and an iptables-only path, so an IPv6 attacker was detected and then
        silently not blocked (vault/Findings/F16).
        """
        if self._is_v6(ip):
            return NFT_BLOCK_SET6, NFT_THROTTLE_SET6, "ip6tables"
        return NFT_BLOCK_SET, NFT_THROTTLE_SET, "iptables"

    # ── firewall backends ─────────────────────────────────────────────────────
    def _run(self, args, stdin=None):
        try:
            subprocess.run(args, check=True, capture_output=True, input=stdin,
                           text=stdin is not None)
            return True
        except Exception as e:
            self.log(f"[IPS] backend cmd failed: {' '.join(args)} ({e})")
            return False

    def _nft_ruleset(self):
        """The complete table, as a declarative ruleset.

        Applied after a flush so re-running is idempotent. `nft add rule` is
        NOT idempotent — the pre-2026-08 code called it on every start, so with
        systemd's Restart=on-failure a crash loop grew the ruleset without
        bound (vault/Findings/F07).
        """
        chains = []
        for hook in _SCOPE_HOOKS[self.scope]:
            chains.append(f"""  chain {hook} {{
    type filter hook {hook} priority -1; policy accept;
    ip saddr @{NFT_THROTTLE_SET} meter throttle_{hook}_v4 {{ ip saddr limit rate over {self.throttle_pps}/second }} drop
    ip6 saddr @{NFT_THROTTLE_SET6} meter throttle_{hook}_v6 {{ ip6 saddr limit rate over {self.throttle_pps}/second }} drop
    ip saddr @{NFT_BLOCK_SET} drop
    ip6 saddr @{NFT_BLOCK_SET6} drop
  }}""")
        return (f"table inet {NFT_TABLE} {{\n"
                f"  set {NFT_BLOCK_SET} {{ type ipv4_addr; flags timeout; }}\n"
                f"  set {NFT_THROTTLE_SET} {{ type ipv4_addr; flags timeout; }}\n"
                f"  set {NFT_BLOCK_SET6} {{ type ipv6_addr; flags timeout; }}\n"
                f"  set {NFT_THROTTLE_SET6} {{ type ipv6_addr; flags timeout; }}\n"
                + "\n".join(chains) + "\n}\n")

    def _ensure_backend(self):
        if self.backend == "nftables":
            # create-then-flush-then-declare == idempotent, no rule accumulation
            self._run(["nft", "add", "table", "inet", NFT_TABLE])
            self._run(["nft", "flush", "table", "inet", NFT_TABLE])
            self._run(["nft", "-f", "-"], stdin=self._nft_ruleset())
        elif self.backend == "iptables":
            # dedicated chain per family, flushed on start; jumps added if absent
            for ipt in ("iptables", "ip6tables"):
                if not shutil.which(ipt):
                    continue
                self._run([ipt, "-N", IPT_CHAIN])         # fails if exists: fine
                self._run([ipt, "-F", IPT_CHAIN])
                for chain in _SCOPE_CHAINS[self.scope]:
                    exists = subprocess.run(
                        [ipt, "-C", chain, "-j", IPT_CHAIN],
                        capture_output=True).returncode == 0
                    if not exists:
                        self._run([ipt, "-I", chain, "-j", IPT_CHAIN])

    def _restore_active(self):
        """Re-apply persisted blocks/throttles after the idempotent flush."""
        now = time.time()
        for ip, v in self.active.items():
            if v["until"] > now:
                self._apply_block(ip, int(v["until"] - now))
        for ip, v in self.throttled.items():
            if v["until"] > now:
                self._apply_throttle(ip, int(v["until"] - now))

    def _apply_block(self, ip, seconds=None):
        secs = seconds or self.block_seconds
        bset, _, ipt = self._sets_for(ip)
        if self.backend == "nftables":
            self._run(["nft", "add", "element", "inet", NFT_TABLE, bset,
                       "{ %s timeout %ds }" % (ip, secs)])
        elif self.backend == "iptables":
            self._run([ipt, "-A", IPT_CHAIN, "-s", ip, "-j", "DROP"])

    def _remove_block(self, ip):
        bset, _, ipt = self._sets_for(ip)
        if self.backend == "nftables":
            self._run(["nft", "delete", "element", "inet", NFT_TABLE,
                       bset, "{ %s }" % ip])
        elif self.backend == "iptables":
            self._run([ipt, "-D", IPT_CHAIN, "-s", ip, "-j", "DROP"])

    def _refresh_block(self, ip):
        """Align the backend timeout with the refreshed persisted deadline."""
        bset, _, _ipt = self._sets_for(ip)
        if self.backend == "nftables":
            # The CLI has no `update element` command. Delete + add in one
            # `nft -f` batch is atomic, so there is no unblocked gap between
            # replacing the old kernel timeout and installing the new one.
            batch = (f"delete element inet {NFT_TABLE} {bset} {{ {ip} }}\n"
                     f"add element inet {NFT_TABLE} {bset} "
                     f"{{ {ip} timeout {self.block_seconds}s }}\n")
            self._run(["nft", "-f", "-"], stdin=batch)
        # iptables rules have no kernel timeout: the userspace `expire()` call
        # removes them according to the refreshed `self.active` deadline.

    def _apply_throttle(self, ip, seconds=None):
        """Rate-limit rather than blackhole — the graduated response tier.

        This was documented in the module docstring and the README but never
        implemented; only DROP existed (vault/Findings/F08).
        """
        secs = seconds or self.block_seconds
        _, tset, ipt = self._sets_for(ip)
        if self.backend == "nftables":
            self._run(["nft", "add", "element", "inet", NFT_TABLE,
                       tset, "{ %s timeout %ds }" % (ip, secs)])
        elif self.backend == "iptables":
            # accept up to the limit, drop the excess from this source
            self._run([ipt, "-A", IPT_CHAIN, "-s", ip, "-m", "limit",
                       "--limit", f"{self.throttle_pps}/second",
                       "--limit-burst", str(self.throttle_pps), "-j", "RETURN"])
            self._run([ipt, "-A", IPT_CHAIN, "-s", ip, "-j", "DROP"])

    def _remove_throttle(self, ip):
        _, tset, ipt = self._sets_for(ip)
        if self.backend == "nftables":
            self._run(["nft", "delete", "element", "inet", NFT_TABLE,
                       tset, "{ %s }" % ip])
        elif self.backend == "iptables":
            self._run([ipt, "-D", IPT_CHAIN, "-s", ip, "-m", "limit",
                       "--limit", f"{self.throttle_pps}/second",
                       "--limit-burst", str(self.throttle_pps), "-j", "RETURN"])
            self._run([ipt, "-D", IPT_CHAIN, "-s", ip, "-j", "DROP"])

    def status(self):
        return {"mode": "enforce" if self.effective_enforce else "dry-run",
                "backend": self.backend, "scope": self.scope,
                "active_blocks": len(self.active),
                "active_throttles": len(self.throttled),
                "min_conf": self.min_conf, "strikes": self.strikes,
                "strike_window": self.strike_window,
                "throttle_pps": self.throttle_pps,
                "block_seconds": self.block_seconds,
                "protects": ("this sensor only" if self.scope == "host"
                             else "this sensor + forwarded traffic")}


def _demo():
    print("IPS responder demo (dry-run)\n")
    r = Responder(mode="dry-run", min_conf=0.9, block_seconds=60,
                  allowlist=["192.168.1.0/24"], strikes=3, strike_window=120)
    print("status:", json.dumps(r.status(), indent=2), "\n")
    tests = [
        ("203.0.113.7", "synflood", 0.99),    # strike 1 -> throttle
        ("203.0.113.7", "synflood", 0.99),    # strike 2 -> still throttled
        ("203.0.113.7", "synflood", 0.99),    # strike 3 -> block
        ("203.0.113.7", "synflood", 0.99),    # already blocked
        ("192.168.1.50", "portscan", 0.99),   # allowlisted subnet
        ("198.51.100.9", "portscan", 0.55),   # below threshold
    ]
    for ip, kind, conf in tests:
        print(" ", r.handle(ip, kind, conf))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IPS active-response")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--scope", default="host", choices=SCOPES)
    args = ap.parse_args()
    if args.demo:
        _demo()
    elif args.status:
        print(json.dumps(Responder(scope=args.scope).status(), indent=2))
    else:
        print("use --demo or --status")
