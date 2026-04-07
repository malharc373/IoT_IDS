import struct
import socket
import collections
import math

# ── pcap reading ──────────────────────────────────────────────────────────────
def read_pcap(filename):
    packets = []
    with open(filename, 'rb') as f:
        hdr = f.read(24)
        magic = struct.unpack('<I', hdr[:4])[0]
        # determine byte order
        endian = '<' if magic == 0xa1b2c3d4 else '>'
        while True:
            rec = f.read(16)
            if len(rec) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + 'IIII', rec)
            raw = f.read(incl_len)
            if len(raw) < incl_len:
                break
            ts = ts_sec + ts_usec / 1e6
            packets.append((ts, orig_len, raw))
    return packets

# ── Ethernet → IP → TCP/UDP parser ───────────────────────────────────────────
def parse_packet(raw):
    if len(raw) < 14:
        return None
    eth_type = struct.unpack('!H', raw[12:14])[0]
    if eth_type != 0x0800:   # IPv4 only
        return None
    ip = raw[14:]
    if len(ip) < 20:
        return None
    ihl = (ip[0] & 0x0F) * 4
    proto = ip[9]
    src_ip = socket.inet_ntoa(ip[12:16])
    dst_ip = socket.inet_ntoa(ip[16:20])
    payload = ip[ihl:]

    src_port = dst_port = flags = 0
    if proto == 6 and len(payload) >= 20:       # TCP
        src_port, dst_port = struct.unpack('!HH', payload[0:4])
        flags = payload[13] & 0x3F              # SYN FIN RST PSH ACK URG
    elif proto == 17 and len(payload) >= 8:     # UDP
        src_port, dst_port = struct.unpack('!HH', payload[0:4])

    return {
        'src': (src_ip, src_port),
        'dst': (dst_ip, dst_port),
        'proto': proto,
        'length': len(raw),
        'flags': flags,
        'payload_len': len(payload)
    }

# ── Flow aggregator ───────────────────────────────────────────────────────────
def build_flows(packets):
    flows = collections.defaultdict(lambda: {
        'pkts': [], 'lengths': [], 'ts_start': None, 'ts_end': None,
        'flags_or': 0, 'proto': 0
    })
    for ts, orig_len, raw in packets:
        p = parse_packet(raw)
        if not p:
            continue
        # bidirectional flow key
        key = tuple(sorted([p['src'], p['dst']])) + (p['proto'],)
        f = flows[key]
        if f['ts_start'] is None:
            f['ts_start'] = ts
            f['proto'] = p['proto']
        f['ts_end'] = ts
        f['pkts'].append(ts)
        f['lengths'].append(p['length'])
        f['flags_or'] |= p['flags']
    return flows

# ── Feature extraction (12 features) ─────────────────────────────────────────
def extract_features(flows):
    results = []
    for key, f in flows.items():
        n = len(f['pkts'])
        if n < 2:
            continue
        duration = max(f['ts_end'] - f['ts_start'], 1e-9)
        lengths   = f['lengths']
        iats      = [f['pkts'][i+1] - f['pkts'][i] for i in range(n-1)]

        mean_len  = sum(lengths) / n
        var_len   = sum((x - mean_len)**2 for x in lengths) / n
        std_len   = math.sqrt(var_len)

        mean_iat  = sum(iats) / len(iats)
        var_iat   = sum((x - mean_iat)**2 for x in iats) / len(iats)
        std_iat   = math.sqrt(var_iat)

        features = {
            'flow_key'       : str(key),
            'proto'          : f['proto'],          # 6=TCP 17=UDP
            'duration'       : round(duration, 6),
            'pkt_count'      : n,
            'total_bytes'    : sum(lengths),
            'mean_pkt_len'   : round(mean_len, 2),
            'std_pkt_len'    : round(std_len, 2),
            'mean_iat'       : round(mean_iat, 6),
            'std_iat'        : round(std_iat, 6),
            'bytes_per_sec'  : round(sum(lengths) / duration, 2),
            'pkts_per_sec'   : round(n / duration, 2),
            'has_syn'        : int(bool(f['flags_or'] & 0x02)),
            'has_fin'        : int(bool(f['flags_or'] & 0x01)),
        }
        results.append(features)
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json
    pkts   = read_pcap('/tmp/test_capture.pcap')
    flows  = build_flows(pkts)
    feats  = extract_features(flows)

    print(f"Flows extracted: {len(feats)}\n")
    for f in feats[:5]:   # print first 5 flows
        print(json.dumps(f, indent=2))

