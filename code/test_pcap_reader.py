import struct
import socket

def read_pcap(filename):
    with open(filename, 'rb') as f:
        # Global header (24 bytes)
        header = f.read(24)
        magic, ver_maj, ver_min, thiszone, sigfigs, snaplen, network = struct.unpack('<IHHiIII', header)
        
        print(f"Magic: {hex(magic)}")
        print(f"Version: {ver_maj}.{ver_min}")
        print(f"Link type: {network}")  # 1 = Ethernet

        packet_count = 0
        while True:
            # Packet record header (16 bytes)
            rec = f.read(16)
            if len(rec) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack('<IIII', rec)
            
            # Raw packet data
            raw = f.read(incl_len)
            if len(raw) < incl_len:
                break

            packet_count += 1
            if packet_count <= 5:  # Print first 5 packets
                print(f"\nPacket {packet_count}: {orig_len} bytes @ {ts_sec}.{ts_usec:06d}")

        print(f"\nTotal packets read: {packet_count}")

read_pcap('/tmp/test_capture.pcap')

