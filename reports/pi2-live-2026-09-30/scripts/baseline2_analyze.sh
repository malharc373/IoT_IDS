set -u
cd ~/IOT-IDS
D=acceptance/baseline2
rm -f $D/alerts.jsonl
sudo -n chown malharfalke $D/benign-10min.pcap
echo "== capture"; cat $D/tcpdump.log; ls -l $D/benign-10min.pcap; sha256sum $D/benign-10min.pcap
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,"src")
from flow_features import read_pcap
p=sorted(read_pcap("acceptance/baseline2/benign-10min.pcap"), key=lambda x:x[0])
print(f"pcap_packets_parsed={len(p)} span_s={p[-1][0]-p[0][0]:.1f}")
PY
echo "== replay (C backend, service defaults: min-conf 0.5, step 1 s)"
.venv/bin/python src/ids_daemon.py --replay $D/benign-10min.pcap --backend c \
    --log $D/alerts.jsonl 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -vE "^\s*(⚠|\[ALERT|ATTACK)" | tail -25
echo "== incidents"
.venv/bin/python - <<'PY'
import json, collections, ipaddress
recs=[json.loads(l) for l in open("acceptance/baseline2/alerts.jsonl") if l.strip()]
last={}
for r in recs: last[(r["src_ip"],r["kind"])]=r
kinds=collections.Counter(k for _,k in last)
def cls(ip):
    a=ipaddress.ip_address(ip)
    if a.is_link_local: return "link-local"
    if a.is_multicast: return "multicast"
    if a.is_unspecified: return "unspecified"
    return f"ipv{a.version}"
src=collections.Counter(cls(s) for s,_ in last)
v6mc=sum(1 for s,_ in last if ipaddress.ip_address(s).version==6 or ipaddress.ip_address(s).is_multicast)
hi=sum(1 for r in last.values() if r["confidence"]>=0.9)
big=sum(1 for r in last.values() if r["flows"]>=5)
print(f"alert_records={len(recs)} distinct_incidents={len(last)}")
print("by_label", dict(kinds.most_common()))
print("by_source_type", dict(src))
print(f"ipv6_or_multicast_incidents={v6mc}  conf>=0.9: {hi}  flows>=5: {big}")
print("max_flows_in_incident", max((r["flows"] for r in last.values()), default=0))
PY
