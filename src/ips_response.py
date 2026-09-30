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


SCOPE: WHAT AN IPS ON THIS BOX CAN ACTUALLY STOP  (review finding F06)
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


CONFIDENCE IS NOT CALIBRATED  (review finding F09)
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
import tempfile

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
_SYSTEM_COMMAND_DIRS = (
    "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin",
)


def find_command(name, search_dirs=_SYSTEM_COMMAND_DIRS):
    """Resolve a firewall executable even under systemd's restricted PATH."""
    found = shutil.which(name)
    if found:
        return os.path.abspath(found)
    for directory in search_dirs:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


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
    if find_command("nft"):
        return "nftables"
    if find_command("iptables"):
        return "iptables"
    return "none"


class Responder:
    def __init__(self, mode="dry-run", min_conf=0.9, block_seconds=300,
                 allowlist=None, backend="auto", state_path=None,
                 max_active=2000, logger=None, scope="host",
                 strikes=3, strike_window=120, throttle_pps=20,
                 class_min_conf=None):
        self.mode = mode                      # "dry-run" | "enforce"
        self.min_conf = min_conf
        if not 0.0 <= self.min_conf <= 1.0:
            raise ValueError("min_conf must be between 0 and 1")
        self.class_min_conf = dict(class_min_conf or {})
        for kind, value in self.class_min_conf.items():
            if not isinstance(kind, str) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "class_min_conf must map class names to values between 0 and 1")
        self.block_seconds = block_seconds
        self.max_active = max_active
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        self.scope = scope
        self.strikes = max(int(strikes), 1)
        self.strike_window = strike_window
        self.throttle_pps = throttle_pps
        self.commands = {
            name: find_command(name)
            for name in ("nft", "iptables", "ip6tables")
        }
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
        backend_ready = (
            bool(self.commands["nft"])
            if self.backend == "nftables"
            else bool(self.commands["iptables"])
        )
        self.effective_enforce = (
            self.mode == "enforce" and self.backend in ("nftables", "iptables")
            and backend_ready and self.is_root)
        if self.mode == "enforce" and not self.effective_enforce:
            self.log(f"[IPS] enforce requested but not possible "
                     f"(backend={self.backend}, command={self.backend_command}, "
                     f"root={self.is_root}) -> dry-run")
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

    @staticmethod
    def _unsafe_source_reason(ip):
        """Return why an address must never become a firewall target."""
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return "invalid-address"
        if addr.is_unspecified:
            return "unspecified"
        if addr.is_loopback:
            return "loopback"
        if addr.is_multicast:
            return "multicast"
        if addr.is_link_local:
            return "link-local"
        if str(addr) == "255.255.255.255":
            return "limited-broadcast"
        return None

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
        except FileNotFoundError:
            self.active = {}
            self.throttled = {}
        except Exception as e:
            self.active = {}
            self.throttled = {}
            self.log(f"[IPS] ignoring unreadable state file {self.state_path}: {e}")

    def _save_state(self):
        """Persist responder state atomically and report durability failures.

        Enforcement can outlive the daemon (notably nft set timeouts), so a
        half-written or silently missing state file makes restart behaviour
        disagree with the firewall.  Write beside the destination, fsync, and
        atomically replace it; keep enforcement available but make any failure
        visible to the operator.
        """
        tmp_path = None
        try:
            state_dir = os.path.dirname(os.path.abspath(self.state_path))
            os.makedirs(state_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                dir=state_dir,
                prefix=f".{os.path.basename(self.state_path)}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump({"active": self.active, "throttled": self.throttled}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_path)
            return True
        except Exception as e:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            self.log(f"[IPS] failed to persist state to {self.state_path}: {e}")
            return False

    # ── corroboration ─────────────────────────────────────────────────────────
    def _record_sighting(self, ip, now):
        """Append this incident and return how many are inside the window.

        The model's confidence is uncalibrated, so a single high score is not
        sufficient evidence to blackhole a host (review finding F09).
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
        if kind == "unknown" and kind not in self.class_min_conf:
            return {"ip": ip, "action": "monitor", "reason": "unknown-verdict"}
        threshold = self.class_min_conf.get(kind, self.min_conf)
        if confidence < threshold:
            return {"ip": ip, "action": "monitor", "reason": "below-threshold"}
        unsafe_reason = self._unsafe_source_reason(ip)
        if unsafe_reason:
            return {"ip": ip, "action": "skip",
                    "reason": f"non-enforceable-source:{unsafe_reason}"}
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
                self._remove_block(ip, missing_ok=True)
            del self.active[ip]
        thr_gone = [ip for ip, v in self.throttled.items() if v["until"] <= now]
        for ip in thr_gone:
            if self.effective_enforce:
                self._remove_throttle(ip, missing_ok=True)
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
        silently not blocked (review finding F16).
        """
        if self._is_v6(ip):
            return NFT_BLOCK_SET6, NFT_THROTTLE_SET6, "ip6tables"
        return NFT_BLOCK_SET, NFT_THROTTLE_SET, "iptables"

    # ── firewall backends ─────────────────────────────────────────────────────
    @property
    def backend_command(self):
        if self.backend == "nftables":
            return self.commands["nft"]
        if self.backend == "iptables":
            return self.commands["iptables"]
        return None

    def _command(self, name):
        """Return an absolute executable path when one is installed."""
        return self.commands.get(name) or name

    def _run(self, args, stdin=None, missing_ok=False):
        try:
            subprocess.run(args, check=True, capture_output=True, input=stdin,
                           text=stdin is not None)
            return True
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if missing_ok and "No such file or directory" in stderr:
                return False
            self.log(f"[IPS] backend cmd failed: {' '.join(args)} "
                     f"({e}; {stderr.strip()})")
            return False
        except Exception as e:
            self.log(f"[IPS] backend cmd failed: {' '.join(args)} ({e})")
            return False

    def _nft_ruleset(self):
        """The complete table, as a declarative ruleset.

        Applied after a flush so re-running is idempotent. `nft add rule` is
        NOT idempotent — the pre-2026-08 code called it on every start, so with
        systemd's Restart=on-failure a crash loop grew the ruleset without
        bound (review finding F07).
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
            nft = self._command("nft")
            self._run([nft, "add", "table", "inet", NFT_TABLE])
            self._run([nft, "flush", "table", "inet", NFT_TABLE])
            self._run([nft, "-f", "-"], stdin=self._nft_ruleset())
        elif self.backend == "iptables":
            # dedicated chain per family, flushed on start; jumps added if absent
            for name in ("iptables", "ip6tables"):
                ipt = self.commands[name]
                if not ipt:
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
            self._run([self._command("nft"), "add", "element", "inet", NFT_TABLE, bset,
                       "{ %s timeout %ds }" % (ip, secs)])
        elif self.backend == "iptables":
            self._run([self._command(ipt), "-A", IPT_CHAIN, "-s", ip, "-j", "DROP"])

    def _remove_block(self, ip, missing_ok=False):
        bset, _, ipt = self._sets_for(ip)
        if self.backend == "nftables":
            self._run([self._command("nft"), "delete", "element", "inet", NFT_TABLE,
                       bset, "{ %s }" % ip], missing_ok=missing_ok)
        elif self.backend == "iptables":
            self._run([self._command(ipt), "-D", IPT_CHAIN, "-s", ip, "-j", "DROP"])

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
            self._run([self._command("nft"), "-f", "-"], stdin=batch)
        # iptables rules have no kernel timeout: the userspace `expire()` call
        # removes them according to the refreshed `self.active` deadline.

    def _apply_throttle(self, ip, seconds=None):
        """Rate-limit rather than blackhole — the graduated response tier.

        This was documented in the module docstring and the README but never
        implemented; only DROP existed (review finding F08).
        """
        secs = seconds or self.block_seconds
        _, tset, ipt = self._sets_for(ip)
        if self.backend == "nftables":
            self._run([self._command("nft"), "add", "element", "inet", NFT_TABLE,
                       tset, "{ %s timeout %ds }" % (ip, secs)])
        elif self.backend == "iptables":
            # accept up to the limit, drop the excess from this source
            ipt = self._command(ipt)
            self._run([ipt, "-A", IPT_CHAIN, "-s", ip, "-m", "limit",
                       "--limit", f"{self.throttle_pps}/second",
                       "--limit-burst", str(self.throttle_pps), "-j", "RETURN"])
            self._run([ipt, "-A", IPT_CHAIN, "-s", ip, "-j", "DROP"])

    def _remove_throttle(self, ip, missing_ok=False):
        _, tset, ipt = self._sets_for(ip)
        if self.backend == "nftables":
            self._run([self._command("nft"), "delete", "element", "inet", NFT_TABLE,
                       tset, "{ %s }" % ip], missing_ok=missing_ok)
        elif self.backend == "iptables":
            ipt = self._command(ipt)
            self._run([ipt, "-D", IPT_CHAIN, "-s", ip, "-m", "limit",
                       "--limit", f"{self.throttle_pps}/second",
                       "--limit-burst", str(self.throttle_pps), "-j", "RETURN"])
            self._run([ipt, "-D", IPT_CHAIN, "-s", ip, "-j", "DROP"])

    def status(self):
        return {"mode": "enforce" if self.effective_enforce else "dry-run",
                "backend": self.backend, "backend_command": self.backend_command,
                "scope": self.scope,
                "active_blocks": len(self.active),
                "active_throttles": len(self.throttled),
                "min_conf": self.min_conf, "strikes": self.strikes,
                "class_min_conf": self.class_min_conf,
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
