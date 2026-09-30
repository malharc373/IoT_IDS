# Raspberry Pi 2 results, 30 September 2026

**Scope: below target.** The documented target is a Raspberry Pi 4 on a
64-bit OS. The lab board is a Raspberry Pi 2 Model B, so this is **not** the
acceptance run described in [`deploy/PI_ACCEPTANCE.md`](../../deploy/PI_ACCEPTANCE.md).
It establishes what works on the board available now. The 24-hour soak is
still pending.

## Target identity

From [`identity.txt`](identity.txt):

| Item | Value |
|---|---|
| Board | Raspberry Pi 2 Model B Rev 1.1 (BCM2836, 4 × Cortex-A7) |
| OS | Raspberry Pi OS (Debian 13 trixie), kernel 6.18.50+rpt-rpi-v7, armv7l, 32-bit |
| glibc / Python | 2.41 / 3.13.5 |
| RAM | 920 MB |
| Network | eth0 on the college LAN |

The code on the Pi was copied with rsync, so it has no `.git`. Its identity
is fixed by hashes instead. `models/live_ids.onnx`, `models/live_ids.h`,
`models/live_meta.json`, `src/c_backend.py`, `src/ids_daemon.py` and
`src/dashboard.py` on the Pi hash identically to commit `b64faad` on branch
`feat/c-inference-fallback`.

## Inference engine

onnxruntime publishes no wheels for 32-bit ARM, and the Cortex-A7 cannot run
a 64-bit OS. The Pi runs the **C export of the same model**
(`models/live_ids.h`), compiled on first use and called through ctypes
(`src/c_backend.py`, branch `feat/c-inference-fallback`).

- **Parity:** the same 5,000 inputs were scored by the Pi's C engine and the
  Mac's onnxruntime 1.23.2: identical labels on all 5,000, max probability
  difference 1.07 × 10⁻⁶.
- A native armv7 build of onnxruntime 1.23.2 from official source is in
  progress, so the ONNX path can be compared on the board itself.

## Benchmark (clean run, Pi 2, C engine)

`demo/benchmark.py` run alone: the dashboard, replay feed and browser were
stopped 20 s before it started (1-minute load average still 4.1 at start,
decaying). Raw output: [`benchmark-pi2-console.txt`](benchmark-pi2-console.txt),
identity and hashes: [`benchmark-pi2-identity.txt`](benchmark-pi2-identity.txt).

| Measure | Pi 2 (armv7, C engine) | Mac host run, 30 Sep (onnxruntime) |
|---|---|---|
| Single-flow inference, daemon path | 535 µs (p99 799 µs) | 14.4 µs (p99 22.0 µs) |
| Peak batched inference | 17,506 flows/s (batch 1024) | 396,744 flows/s |
| Native C `ids_predict`, no Python | 50.8 µs/flow (19,667 flows/s) | 1.1 µs/flow |
| Packet parse + flow aggregation | 6,758 packets/s (1,770 flows/s) | 213,947 packets/s |
| End-to-end pcap to verdicts | 1,316 flows/s | 50,454 flows/s |
| Verdicts on the demo capture | 4,573 attack / 101 benign | 4,573 attack / 101 benign |

Reading these honestly:

- The single-flow daemon figure is dominated by the ctypes call and numpy
  softmax, not the trees: the bare C walk is about 51 µs. Batching recovers
  most of it (57 µs/flow at batch 1024).
- **Packet parsing, not the model, is the bottleneck on this board.** At
  about 6.8k packets/s in Python, a Pi 2 keeps up with a quiet IoT segment
  but not with a flood at line rate; expect dropped packets under a real
  SYN/UDP flood. This is a property of the Pi 2 and of pure-Python parsing.
- The Mac column is a different machine and engine, shown only for scale.
  Neither column predicts Pi 4 performance.

If the accuracy and memory sections did not finish before access to the
board ended, the console file stops at section 6; accuracy is
engine-independent here (the C and ONNX engines agree on every label).

## Feature checks without root (15 / 15 pass)

From [`feature-results.txt`](feature-results.txt). Raw output for each check
was kept on the Pi under `acceptance/features/`.

| # | Feature | Result |
|---|---|---|
| 01 | Runtime preflight (`demo/preflight.py --runtime-only --iface eth0`) | PASS |
| 02 | Offline pcap mode and per-flow CSV (4,674 flows) | PASS |
| 03 | Replay mode: 33 incidents, all 9 attack types | PASS |
| 04 | Per-class threshold (`xmas_scan=0.99`) and abstention (`0.9`, 8 `unknown`) | PASS |
| 05 | IPS dry-run ladder without root: 6 would-block, 34 throttle lines, firewall untouched | PASS |
| 06 | SIEM syslog export, CEF (67 messages) | PASS |
| 06 | SIEM syslog export, JSON (67 messages) | PASS |
| 08 | SIEM unreachable: sensor keeps running | PASS |
| 09 | Dashboard refuses `0.0.0.0` without a token | PASS |
| 09 | Dashboard token auth: none 401, wrong 401, query 200, header 200 | PASS |
| 10 | Dashboard `/api/state` JSON | PASS |
| 11 | MCU C99 header compiles with `-Wall -Wextra -Werror` and runs on ARM | PASS |
| 12 | `flow_features.py` CLI | PASS |
| 13 | All 10 traffic generators run on the Pi | PASS |
| 14 | `--backend c` works; `--backend onnx` fails cleanly | PASS |

The replay on the Pi ([`replay.log`](replay.log)) gives the same totals as
the Mac: 17,850 packets, 4,674 flows, 4,573 attack and 101 benign, 33
incidents. The demo capture was generated on the Mac (its generator imports
pandas) and copied over; its SHA-256 is in `identity.txt`.

## Smoke suite on the Pi (49 pass, 23 fail, none a defect)

From [`smoke-pi.txt`](smoke-pi.txt). Every failure is one of:

- **Training-only dependency absent (19):** pandas, xgboost or onnxruntime,
  none of which is part of the edge runtime.
- **No `.git` in an rsync copy (3):** governance, metadata and debris checks
  call `git ls-files`.
- **Release bundle (1):** `output/` was not copied.

## Dashboard on the Pi

The dashboard runs on the Pi and is shown on the Pi's own display (Firefox,
kiosk mode; Chromium's GPU path fails on this board) and over an SSH tunnel.
With the heartbeat added in `b64faad`, the sensor strip reported engine `c`,
about 810 packets/s during real-time replay, 39 MB daemon RSS and a CPU
temperature of 54–56 °C.

## Still to do on this board (needs root)

- `deploy/setup_pi.sh` from a clean venv (tests the armv7 path: gcc,
  libopenblas0, piwheels numpy)
- live capture on `eth0` (and `lo` for self-generated floods)
- IPS enforcement with nftables, including block expiry and allowlist
- systemd services surviving a reboot, then the 24-hour soak
