# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.1-arm64-arm-64bit
python=3.10.14  time=2026-08-22 14:18

====================================================================
  1. MODEL PARAMETERS & FOOTPRINT
====================================================================
  Features               : 22
  Classes                : 10  (benign, portscan, synflood, icmpflood, udpflood, ssh_bruteforce, slowloris, mirai, xmas_scan, mqtt_flood)
  Boosted trees          : 1200
  Total nodes            : 2,780 / 1,990 leaves
  ONNX model size        : 91.8 KB
  C header size          : 102.3 KB (~43 KB const data)
  In-domain metrics      : {'multiclass_accuracy': 1.0, 'macro_f1': 1.0, 'binary_accuracy': 1.0, 'binary_f1_attack': 1.0}
  Split                  : GroupShuffleSplit on scenario (no scenario spans the split)
  Without dst_port       : acc=1.0 (delta +0.0000) — the model is not a port lookup
  CAVEAT                 : these are SYNTHETIC-traffic numbers. The
                           corpus is trivially separable (scenario-level
                           split, mixed benign background and a dst_port
                           ablation all leave the score unchanged), so
                           read them as a property of the generators,
                           not as detection accuracy on real traffic.

====================================================================
  2. ONNX INFERENCE LATENCY & THROUGHPUT
====================================================================
   batch   mean_ms   p50_ms   p99_ms   us/flow      flows/s
       1    0.0111   0.0095   0.0197    11.081       90,247
       8    0.0461   0.0451   0.1078     5.759      173,632
      32    0.1648   0.1625   0.2537     5.150      194,165
      64    0.1843   0.1796   0.2553     2.880      347,214
     128    0.3390   0.3178   0.4523     2.649      377,530
     512    1.2936   1.2302   1.6087     2.527      395,801
    1024    2.5647   2.4395   3.1053     2.505      399,270

  single-flow latency    : 11.1 us (p99 19.7 us)
  peak throughput        : 399,270 flows/s (batch 1024)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1111.2 ns/flow (1.111 us)
  C throughput           : 899,964 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 10.2 ms (1,746,625 packets/s)
  parse+aggregate        : 85.6 ms (208,465 packets/s, 54,586 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 91.1 ms (51,321 flows/s end-to-end)
  detected 4,573 attack flows / 101 benign

====================================================================
  6. ACCURACY (held-out, unseen-seed synthetic)
====================================================================
  flows evaluated        : 35,057 (unseen seeds)
  multiclass accuracy    : 99.65%
  macro F1               : 0.9961
  attack detection rate  : 100.00%
  benign false-pos rate  : 0.00%
  per-class recall:
    benign           100.0%
    portscan         100.0%
    synflood         100.0%
    icmpflood        100.0%
    udpflood         100.0%
    ssh_bruteforce   100.0%
    slowloris        100.0%
    mirai             94.2%
    xmas_scan        100.0%
    mqtt_flood       100.0%

  NOTE: synthetic traffic is separable; see CROSS_DATASET_FINDINGS.md
        cross-dataset exact numbers withdrawn; protocol-correct rerun pending.

====================================================================
  7. MEMORY FOOTPRINT
====================================================================
  daemon runtime RSS     : 56.6 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 317.4 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. UNVALIDATED RASPBERRY PI 4 PROJECTION (host x 12)
====================================================================
  host                   : Apple M4 (arm64)
  ONNX single-flow (Pi)  : ~133.0 us/flow (~7,521 flows/s)
  C model (Pi)           : ~13.33 us/flow (~74,997 flows/s)
  feature extraction (Pi): ~17,372 packets/s (the real bottleneck on a live link)
  status                 : estimate only — PI_FACTOR is not a measurement.
  acceptance gate        : run this benchmark on the target Pi and publish the raw output before making a real-time throughput claim.
```
