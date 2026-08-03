#!/usr/bin/env python3
"""
ips_response.py — turn the IDS into an IPS (intrusion *prevention* system).

On a high-confidence incident the responder can actively block or rate-limit
the offending source. Safety is the priority:

  * dry-run by default — logs the action it *would* take, changes nothing
  * confidence gate     — only acts above a threshold
  * allowlist           — never touch loopback, the sensor's own IP, or
                          operator-supplied IPs/subnets
  * auto-expiry         — blocks lift themselves after a timeout
  * backend auto-detect — nftables (preferred) or iptables on Linux; on other
                          platforms it stays in dry-run (safe no-op)

Enforcement runs real firewall commands only with mode="enforce" AND a working
backend AND root — otherwise it degrades to dry-run rather than failing.
"""
from __future__ import annotations

import os
import json
import time
import socket
import shutil
import ipaddress
import subprocess
import datetime as dt

NFT_TABLE = "iot_ids"
NFT_SET = "blocked"


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
                 max_active=2000, logger=None):
        self.mode = mode                      # "dry-run" | "enforce"
        self.min_conf = min_conf
        self.block_seconds = block_seconds
        self.max_active = max_active
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
            self.active = {ip: v for ip, v in data.items() if v["until"] > now}
        except Exception:
            self.active = {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self.active, f)
        except Exception:
            pass

    # ── main entry point ──────────────────────────────────────────────────────
    def handle(self, ip, kind, confidence):
        """Decide + act on one incident. Returns an action dict."""
        if confidence < self.min_conf:
            return {"ip": ip, "action": "monitor", "reason": "below-threshold"}
        if self._allowed(ip):
            return {"ip": ip, "action": "skip", "reason": "allowlisted"}
        now = time.time()
        if ip in self.active and self.active[ip]["until"] > now:
            self.active[ip]["until"] = now + self.block_seconds   # refresh
            return {"ip": ip, "action": "already-blocked", "kind": kind}
        if len(self.active) >= self.max_active:
            return {"ip": ip, "action": "skip", "reason": "max-active"}

        self.active[ip] = {"until": now + self.block_seconds, "kind": kind}
        self._save_state()
        verb = "blocked" if self.effective_enforce else "would-block"
        if self.effective_enforce:
            self._apply_block(ip)
        self.log(f"  \033[1;35m⛔ IPS {verb}\033[0m {ip:<15} "
                 f"({kind}, conf={confidence:.2f}) for {self.block_seconds}s "
                 f"[{'enforce' if self.effective_enforce else 'dry-run'}]")
        return {"ip": ip, "action": verb, "kind": kind,
                "expires_in": self.block_seconds}

    def expire(self):
        """Lift blocks whose timeout has passed (iptables needs this; nft sets
        auto-expire, but we keep state consistent either way)."""
        now = time.time()
        gone = [ip for ip, v in self.active.items() if v["until"] <= now]
        for ip in gone:
            if self.effective_enforce and self.backend == "iptables":
                self._remove_block(ip)
            del self.active[ip]
        if gone:
            self._save_state()
        return gone

    # ── firewall backends ─────────────────────────────────────────────────────
    def _run(self, args):
        try:
            subprocess.run(args, check=True, capture_output=True)
            return True
        except Exception as e:
            self.log(f"[IPS] backend cmd failed: {' '.join(args)} ({e})")
            return False

    def _ensure_backend(self):
        if self.backend == "nftables":
            # idempotent setup of a dedicated table/set/chain
            self._run(["nft", "add", "table", "inet", NFT_TABLE])
            self._run(["nft", "add", "set", "inet", NFT_TABLE, NFT_SET,
                       "{ type ipv4_addr; flags timeout; }"])
            self._run(["nft", "add", "chain", "inet", NFT_TABLE, "input",
                       "{ type filter hook input priority -1; }"])
            self._run(["nft", "add", "rule", "inet", NFT_TABLE, "input",
                       "ip", "saddr", f"@{NFT_SET}", "drop"])

    def _apply_block(self, ip):
        if self.backend == "nftables":
            self._run(["nft", "add", "element", "inet", NFT_TABLE, NFT_SET,
                       "{ %s timeout %ds }" % (ip, self.block_seconds)])
        elif self.backend == "iptables":
            self._run(["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"])

    def _remove_block(self, ip):
        if self.backend == "iptables":
            self._run(["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"])

    def status(self):
        return {"mode": "enforce" if self.effective_enforce else "dry-run",
                "backend": self.backend, "active_blocks": len(self.active),
                "min_conf": self.min_conf, "block_seconds": self.block_seconds}


def _demo():
    print("IPS responder demo (dry-run)\n")
    r = Responder(mode="dry-run", min_conf=0.9, block_seconds=60,
                  allowlist=["192.168.1.0/24"])
    print("status:", r.status(), "\n")
    tests = [
        ("203.0.113.7", "synflood", 0.99),
        ("203.0.113.7", "synflood", 0.99),    # repeat -> already-blocked
        ("192.168.1.50", "portscan", 0.99),   # allowlisted subnet
        ("198.51.100.9", "portscan", 0.55),   # below threshold
        ("198.51.100.9", "portscan", 0.97),   # now acts
    ]
    for ip, kind, conf in tests:
        print(" ", r.handle(ip, kind, conf))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IPS active-response")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
    elif args.status:
        print(json.dumps(Responder().status(), indent=2))
    else:
        print("use --demo or --status")
