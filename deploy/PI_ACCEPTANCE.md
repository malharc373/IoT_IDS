# Raspberry Pi acceptance protocol

This protocol closes R22. Run it on the actual target, preserve raw output, and
do not replace the host benchmark until every identity and soak item is present.

## 1. Record the target

```bash
date -Is
cat /proc/device-tree/model; echo
uname -a
getconf GNU_LIBC_VERSION
python3 --version
vcgencmd get_throttled 2>/dev/null || true
vcgencmd measure_temp 2>/dev/null || true
git rev-parse HEAD
sha256sum models/live_ids.onnx models/live_ids.h models/live_meta.json
```

The benchmark now accepts a host as Raspberry Pi only when the Linux device
tree contains `Raspberry Pi`; ARM architecture alone is rejected as evidence.

## 2. Reproduce the benchmark

Generate the ignored demo capture if it is absent, then run the complete report:

```bash
.venv/bin/python attacks/build_corpus.py --scenarios 30
bash demo/run_demo.sh
.venv/bin/python demo/benchmark.py | tee pi-benchmark-console.txt
```

Confirm section 8 names the exact Pi hardware and says no scaling was applied.
Archive `pi-benchmark-console.txt`, `demo/results/BENCHMARK.md`, and the latency
chart with hashes. A report containing `section FAILED`, `projection`, or a
different commit/model hash fails acceptance.

## 3. Run a passive soak

Use a representative interface and remain in IDS-only mode. Do not enable
firewall enforcement merely to run a stability test.

```bash
sudo systemctl restart iot-ids
systemctl show iot-ids -p ActiveEnterTimestamp -p NRestarts -p MemoryCurrent -p CPUUsageNSec
journalctl -u iot-ids --since now -f
```

At a fixed interval for at least 24 hours, record:

```bash
date -Is
systemctl show iot-ids -p ActiveState -p NRestarts -p MemoryCurrent -p CPUUsageNSec
ps -o pid,etimes,rss,%cpu,cmd -C python3
vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true
ip -s link show "${IOTIDS_IFACE:-eth0}"
```

Acceptance requires no daemon restart/OOM, no unbounded RSS trend after traffic
returns to baseline, no growth of internal flow bookkeeping after idle eviction,
and no undisclosed thermal throttling or interface drops. Record traffic rate
and topology so a quiet lab is not presented as a load test.

## 4. Publish without overclaiming

Report median and p99 single-flow latency, end-to-end flow throughput, packet
extraction throughput, daemon RSS, temperature/throttle state, soak duration,
traffic volume, restarts, drops, exact hardware/OS, commit, and artifact hashes.
Keep Apple M4 results as host evidence; label Pi results separately.
