#!/usr/bin/env python3
"""
dashboard.py — live web dashboard for the IoT-IDS sensor.

Reads the daemon's alert feed (logs/alerts.jsonl) and IPS state
(logs/ips_state.json) and serves a self-contained, auto-refreshing web page.
Stdlib only (http.server) — no Flask, no external assets — so it runs on the
Pi alongside the sensor.

    # terminal 1: the sensor writes alerts
    sudo python src/ids_daemon.py --iface eth0 --ips
    # terminal 2: the dashboard reads them
    python src/dashboard.py --port 8080
    # browse http://<pi-ip>:8080

For a no-hardware preview, feed it a demo run first:
    python src/ids_daemon.py --pcap data/pcaps/demo_mixed.pcap --ips
    python src/dashboard.py


BINDING AND AUTH  (see vault/Findings/F10)
-------------------------------------------
This page exposes the full alert feed: which hosts are attacking, which are
being blocked, and the internal addressing of the monitored segment. That is
reconnaissance material, so it is not served to a network by accident.

  * default bind is 127.0.0.1 (was 0.0.0.0)
  * binding to a non-loopback address REQUIRES --token (or the
    IOTIDS_DASHBOARD_TOKEN env var), or an explicit --insecure acknowledgement
  * with a token set, every request must carry ?token=... or an
    X-Auth-Token header

There is no TLS here — it is a stdlib http.server. Over an untrusted network,
prefer an SSH tunnel:  ssh -L 8080:127.0.0.1:8080 pi@<host>


INCREMENTAL READS
-----------------
build_state() used to readlines() the whole alert log on every request, from
every client, every 2 seconds. The log has no size bound, so cost grew without
limit. State is now cached and only appended bytes are parsed; truncation or
rotation is detected and triggers a clean re-read.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hmac
import secrets
import argparse
import threading
import ipaddress
import datetime as dt
import collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_LOG = os.path.join(ROOT, "logs", "alerts.jsonl")
IPS_STATE = os.path.join(ROOT, "logs", "ips_state.json")
META = os.path.join(ROOT, "models", "live_meta.json")

# category -> validated categorical hue (dark mode; see dataviz palette)
CATEGORY_COLORS = {
    "recon": "#3987e5", "dos": "#d95926",
    "botnet": "#199e70", "bruteforce": "#c98500",
    "attack": "#d55181",
}


def _categories():
    try:
        return json.load(open(META)).get("categories", {})
    except Exception:
        return {}


class AlertFeed:
    """Incrementally-parsed view of the alert log.

    Keeps a rolling deque of recent records and only reads bytes appended since
    the last poll. Detects truncation/rotation (file shrank or the inode
    changed) and re-reads from the start.
    """

    MAX_RECORDS = 5000

    def __init__(self, path, max_records=None):
        self.path = path
        self.max_records = max_records or self.MAX_RECORDS
        self.records = collections.deque(maxlen=self.max_records)
        self._pos = 0
        self._ino = None
        self._lock = threading.Lock()
        self._partial = ""

    def _reset(self):
        self.records.clear()
        self._pos = 0
        self._partial = ""

    def refresh(self):
        """Read whatever is new. Cheap enough to call on every request."""
        with self._lock:
            try:
                stat = os.stat(self.path)
            except FileNotFoundError:
                self._reset()
                self._ino = None
                return
            # rotated (new inode) or truncated (smaller than our offset)
            if self._ino is not None and (stat.st_ino != self._ino
                                          or stat.st_size < self._pos):
                self._reset()
            self._ino = stat.st_ino
            if stat.st_size == self._pos:
                return
            with open(self.path, "r") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
            data = self._partial + chunk
            # a writer may be mid-line; hold the tail back until it completes
            if data and not data.endswith("\n"):
                data, _, self._partial = data.rpartition("\n")
            else:
                self._partial = ""
            for ln in data.splitlines():
                if not ln.strip():
                    continue
                try:
                    self.records.append(json.loads(ln))
                except Exception:
                    pass

    def snapshot(self):
        with self._lock:
            return list(self.records)


def build_state(feed):
    """Aggregate the feed into the shape the page renders.

    Accepts an AlertFeed, or a path (for callers and tests that pass one).
    """
    if isinstance(feed, str):
        feed = AlertFeed(feed)
    feed.refresh()
    records = feed.snapshot()
    cats = _categories()

    # dedupe to unique incidents keyed by (src_ip, kind): keep the largest/last
    incidents = {}
    for r in records:
        key = (r.get("src_ip"), r.get("kind"))
        cur = incidents.get(key)
        if cur is None or r.get("flows", 0) >= cur.get("flows", 0):
            incidents[key] = r
    inc = list(incidents.values())

    by_type = collections.Counter()
    by_cat = collections.Counter()
    by_src = collections.Counter()
    src_kinds = collections.defaultdict(set)
    for r in inc:
        k = r.get("kind", "?")
        by_type[k] += 1
        by_cat[cats.get(k, "attack")] += 1
        by_src[r.get("src_ip", "?")] += 1
        src_kinds[r.get("src_ip", "?")].add(k)

    # timeline: incidents per minute over the last 30 minutes
    buckets = collections.Counter()
    for r in inc:
        ts = r.get("ts", "")
        try:
            t = dt.datetime.fromisoformat(ts)
            buckets[t.replace(second=0, microsecond=0).isoformat()] += 1
        except Exception:
            pass
    timeline = [{"t": k, "count": v} for k, v in sorted(buckets.items())][-30:]

    # active IPS blocks and throttles
    blocks, throttles = [], []
    if os.path.exists(IPS_STATE):
        try:
            now = time.time()
            raw = json.load(open(IPS_STATE))
            for bucket, out in (("active", blocks), ("throttled", throttles)):
                # tolerate the pre-2026-08 flat format
                items = raw.get(bucket, raw if bucket == "active" else {})
                for ip, v in items.items():
                    if isinstance(v, dict) and v.get("until", 0) > now:
                        out.append({"ip": ip, "kind": v.get("kind", "?"),
                                    "expires_in": int(v["until"] - now)})
        except Exception:
            pass

    recent = sorted(inc, key=lambda r: r.get("ts", ""), reverse=True)[:15]
    for r in recent:
        r["category"] = cats.get(r.get("kind"), "attack")

    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "totals": {
            "incidents": len(inc),
            "sources": len(by_src),
            "active_blocks": len(blocks),
            "active_throttles": len(throttles),
            "top_attack": (by_type.most_common(1)[0][0] if by_type else "—"),
        },
        "by_type": [{"type": k, "category": cats.get(k, "attack"), "count": c}
                    for k, c in by_type.most_common()],
        "by_category": [{"category": k, "count": c} for k, c in by_cat.most_common()],
        "top_sources": [{"src_ip": s, "count": c, "kinds": sorted(src_kinds[s])}
                        for s, c in by_src.most_common(8)],
        "recent": recent,
        "blocks": sorted(blocks, key=lambda b: b["expires_in"]),
        "throttles": sorted(throttles, key=lambda b: b["expires_in"]),
        "timeline": timeline,
        "colors": CATEGORY_COLORS,
    }


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>IoT-IDS — live monitor</title>
<style>
:root{--bg:#111110;--surface:#1a1a19;--surface2:#232322;--line:#33322f;
--ink:#ffffff;--ink2:#c3c2b7;--muted:#8a897e;--good:#199e70;--crit:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:16px 22px;
border-bottom:1px solid var(--line);background:var(--surface)}
header h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--good);
box-shadow:0 0 0 0 rgba(25,158,112,.6);animation:p 2s infinite}
@keyframes p{0%{box-shadow:0 0 0 0 rgba(25,158,112,.5)}70%{box-shadow:0 0 0 7px rgba(25,158,112,0)}100%{box-shadow:0 0 0 0 rgba(25,158,112,0)}}
.sub{color:var(--muted);font-size:12px;margin-left:auto}
main{padding:20px;max-width:1180px;margin:0 auto;display:grid;gap:16px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.tile .v{font-size:26px;font-weight:680;margin-top:6px}
.tile .v.crit{color:var(--crit)}.tile .v.good{color:var(--good)}
.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--ink2);margin:0 0 12px}
.bar{display:flex;align-items:center;gap:10px;margin:7px 0}
.bar .lab{width:120px;color:var(--ink2);font-size:12.5px;text-align:right;flex:none;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar .track{flex:1;height:16px;background:var(--surface2);border-radius:4px;overflow:hidden}
.bar .fill{height:100%;border-radius:4px;min-width:3px;transition:width .4s}
.bar .n{width:34px;font-variant-numeric:tabular-nums;color:var(--ink);font-size:12.5px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
letter-spacing:.4px;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid var(--line);color:var(--ink2)}
td .ip{color:var(--ink);font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;
font-weight:600;color:#0b0b0b}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}
.legend span{display:flex;align-items:center;gap:6px;color:var(--ink2);font-size:12px}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
.spark{display:flex;align-items:flex-end;gap:2px;height:52px}
.spark div{flex:1;background:var(--crit);border-radius:2px 2px 0 0;min-height:2px;opacity:.85}
.empty{color:var(--muted);padding:18px 0;text-align:center}
@media(max-width:820px){.tiles{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><span class=dot></span><h1>IoT-IDS — live monitor</h1>
<span class=sub id=gen>connecting…</span></header>
<main>
 <div class=tiles>
   <div class=tile><div class=k>Active incidents</div><div class="v crit" id=t_inc>0</div></div>
   <div class=tile><div class=k>Attacking sources</div><div class=v id=t_src>0</div></div>
   <div class=tile><div class=k>IPS blocks active</div><div class="v" id=t_blk>0</div></div>
   <div class=tile><div class=k>Top attack</div><div class=v id=t_top>—</div></div>
 </div>
 <div class=grid>
   <div class=card><h2>Incidents by attack type</h2><div id=bars></div>
     <div class=legend id=legend></div></div>
   <div class=card><h2>Incidents / min</h2><div class=spark id=spark></div>
     <h2 style=margin-top:18px>Top sources</h2><div id=srcs></div></div>
 </div>
 <div class=grid>
   <div class=card><h2>Recent incidents</h2><table><thead><tr><th>Time</th><th>Source</th>
     <th>Type</th><th>Flows</th><th>Conf</th></tr></thead><tbody id=recent></tbody></table></div>
   <div class=card><h2>Blocked sources (IPS)</h2><table><thead><tr><th>IP</th><th>Reason</th>
     <th>Expires</th></tr></thead><tbody id=blocks></tbody></table></div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function tick(){
 let s; try{s=await (await fetch('/api/state')).json()}catch(e){$('gen').textContent='offline';return}
 const C=s.colors;
 $('gen').textContent='updated '+s.generated.replace('T',' ');
 $('t_inc').textContent=s.totals.incidents;
 $('t_src').textContent=s.totals.sources;
 $('t_blk').textContent=s.totals.active_blocks;
 $('t_top').textContent=s.totals.top_attack;
 const max=Math.max(1,...s.by_type.map(d=>d.count));
 $('bars').innerHTML=s.by_type.length?s.by_type.map(d=>{
   const col=C[d.category]||C.attack;
   return `<div class=bar><div class=lab title="${esc(d.type)}">${esc(d.type)}</div>
   <div class=track><div class=fill style="width:${d.count/max*100}%;background:${col}"></div></div>
   <div class=n>${d.count}</div></div>`}).join(''):'<div class=empty>no incidents yet</div>';
 const cats=[...new Set(s.by_type.map(d=>d.category))];
 $('legend').innerHTML=cats.map(c=>`<span><i style="background:${C[c]||C.attack}"></i>${esc(c)}</span>`).join('');
 const tl=s.timeline,tm=Math.max(1,...tl.map(d=>d.count));
 $('spark').innerHTML=tl.length?tl.map(d=>`<div title="${esc(d.t)}: ${d.count}" style="height:${d.count/tm*100}%"></div>`).join(''):'<div class=empty>—</div>';
 $('srcs').innerHTML=s.top_sources.length?s.top_sources.map(d=>
   `<div class=bar><div class=lab title="${esc(d.src_ip)}"><span class=ip>${esc(d.src_ip)}</span></div>
   <div class=track><div class=fill style="width:${d.count/Math.max(1,s.top_sources[0].count)*100}%;background:var(--crit)"></div></div>
   <div class=n>${d.count}</div></div>`).join(''):'<div class=empty>—</div>';
 $('recent').innerHTML=s.recent.length?s.recent.map(r=>{
   const col=C[r.category]||C.attack;
   return `<tr><td>${esc((r.ts||'').split('T')[1]||r.ts)}</td>
   <td class=ip>${esc(r.src_ip)}</td>
   <td><span class=pill style="background:${col}">${esc(r.category)}/${esc(r.kind)}</span></td>
   <td>${r.flows||''}</td><td>${(r.confidence!=null?r.confidence:'')}</td></tr>`}).join(''):'<tr><td colspan=5 class=empty>no incidents yet</td></tr>';
 $('blocks').innerHTML=s.blocks.length?s.blocks.map(b=>
   `<tr><td class=ip>${esc(b.ip)}</td><td>${esc(b.kind)}</td><td>${b.expires_in}s</td></tr>`).join(''):'<tr><td colspan=3 class=empty>none</td></tr>';
}
tick();setInterval(tick,2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    log_path = DEFAULT_LOG
    feed = None
    token = None

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # this page renders only its own inlined CSS/JS and talks to itself
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; style-src 'unsafe-inline'; "
                         "script-src 'unsafe-inline'; connect-src 'self'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if not self.token:
            return True
        supplied = self.headers.get("X-Auth-Token", "")
        if not supplied and "?" in self.path:
            from urllib.parse import urlparse, parse_qs
            supplied = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        # constant-time compare so the token can't be recovered by timing
        return hmac.compare_digest(supplied, self.token)

    def do_GET(self):
        if not self._authorized():
            self._send(401, b"unauthorized", "text/plain")
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            body = json.dumps(build_state(self.feed)).encode()
            self._send(200, body, "application/json")
        elif path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass   # quiet


def _is_loopback(host):
    if host in ("", "0.0.0.0", "::"):
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def main():
    ap = argparse.ArgumentParser(description="IoT-IDS live web dashboard")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("IOTIDS_DASHBOARD_PORT", 8080)))
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1 — loopback only)")
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--token", default=os.environ.get("IOTIDS_DASHBOARD_TOKEN"),
                    help="require this token on every request "
                         "(?token=... or X-Auth-Token). Use 'generate' to mint one.")
    ap.add_argument("--insecure", action="store_true",
                    help="acknowledge serving the alert feed to a network with "
                         "no authentication")
    args = ap.parse_args()

    token = args.token
    if token == "generate":
        token = secrets.token_urlsafe(24)

    if not _is_loopback(args.host) and not token and not args.insecure:
        sys.exit(
            f"[ERROR] refusing to serve the alert feed on {args.host} without auth.\n"
            f"        The dashboard exposes attacking hosts, blocked hosts and the\n"
            f"        addressing of the monitored segment.\n"
            f"        Pick one:\n"
            f"          --token generate      mint a token and print the URL\n"
            f"          --token <secret>      use your own\n"
            f"          --host 127.0.0.1      keep it local, reach it over an SSH tunnel:\n"
            f"                                  ssh -L {args.port}:127.0.0.1:{args.port} pi@<host>\n"
            f"          --insecure            you have read the above and accept it")

    Handler.log_path = args.log
    Handler.feed = AlertFeed(args.log)
    Handler.token = token
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{shown}:{args.port}"
    if token:
        url += f"/?token={token}"
    print(f"IoT-IDS dashboard → {url}  (reading {args.log})")
    if token:
        print(f"  auth: token required on every request")
    elif not _is_loopback(args.host):
        print("  auth: NONE — serving the alert feed to the network (--insecure)")
    print("Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
