import sys, collections; sys.path.insert(0, "src")
from flow_features import read_pcap, FlowTable, parse_raw
from ids_daemon import Detector
det = Detector("models/live_ids.onnx", "models/live_meta.json", backend="c")
pk = sorted(read_pcap("acceptance/baseline2/benign-10min.pcap"), key=lambda x: x[0])
t = FlowTable(); flagged = {}; nxt = pk[0][0] + 1.0; n = 0
def flush():
    rows = t.extract_live(min_pkts=1, window=60.0)
    if not rows: return
    res = det.classify([r[2] for r in rows])
    for (key, meta, _, _), (lab, conf) in zip(rows, res):
        if lab != "benign" and conf >= 0.5 and key not in flagged:
            flagged[key] = lab
for ts, raw, ol in pk:
    p = parse_raw(raw, ol)
    if p is not None:
        t.add_packet(p, ts); n += 1
    if ts >= nxt:
        flush(); nxt += 1.0
flush()
total = len(t.extract(min_pkts=1, window=None))
print(f"ip_packets={n} distinct_flows={total} flows_ever_flagged={len(flagged)} per_1000={1000*len(flagged)/total:.0f}")
print("flagged_by_label", dict(collections.Counter(flagged.values())))
