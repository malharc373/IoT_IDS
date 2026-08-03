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
"""
from __future__ import annotations

import os
import io
import json
import time
import argparse
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


def build_state(log_path):
    cats = _categories()
    records = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()[-5000:]
        for ln in lines:
            try:
                records.append(json.loads(ln))
            except Exception:
                pass

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

    # active IPS blocks
    blocks = []
    if os.path.exists(IPS_STATE):
        try:
            now = time.time()
            for ip, v in json.load(open(IPS_STATE)).items():
                if v.get("until", 0) > now:
                    blocks.append({"ip": ip, "kind": v.get("kind", "?"),
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
            "top_attack": (by_type.most_common(1)[0][0] if by_type else "—"),
        },
        "by_type": [{"type": k, "category": cats.get(k, "attack"), "count": c}
                    for k, c in by_type.most_common()],
        "by_category": [{"category": k, "count": c} for k, c in by_cat.most_common()],
        "top_sources": [{"src_ip": s, "count": c, "kinds": sorted(src_kinds[s])}
                        for s, c in by_src.most_common(8)],
        "recent": recent,
        "blocks": sorted(blocks, key=lambda b: b["expires_in"]),
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

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(build_state(self.log_path)).encode()
            self._send(200, body, "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass   # quiet


def main():
    ap = argparse.ArgumentParser(description="IoT-IDS live web dashboard")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--log", default=DEFAULT_LOG)
    args = ap.parse_args()
    Handler.log_path = args.log
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"IoT-IDS dashboard → http://{shown}:{args.port}  (reading {args.log})")
    print("Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
