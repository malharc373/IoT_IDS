# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6-arm64-arm-64bit
python=3.10.14  time=2026-08-04 14:27

====================================================================
  1. MODEL PARAMETERS & FOOTPRINT
====================================================================
  Features               : 22
  Classes                : 10  (benign, portscan, synflood, icmpflood, udpflood, ssh_bruteforce, slowloris, mirai, xmas_scan, mqtt_flood)
  Boosted trees          : 1200
  Total nodes / leaves   : 2,720 / 1,960
  ONNX model size        : 90.2 KB
  C header size          : 101.1 KB (~42 KB const data)
  Booster JSON (train)   : 582.9 KB
  In-domain metrics      : {'multiclass_accuracy': 1.0, 'macro_f1': 1.0, 'binary_accuracy': 1.0, 'binary_f1_attack': 1.0}

====================================================================
  2. ONNX INFERENCE LATENCY & THROUGHPUT
====================================================================
   batch   mean_ms   p50_ms   p99_ms   us/flow      flows/s
       1    0.0077   0.0074   0.0113     7.705      129,794
       8    0.0464   0.0456   0.0526     5.803      172,331
      32    0.1442   0.1423   0.1538     4.505      221,973
      64    0.1719   0.1625   0.2273     2.686      372,292
     128    0.3287   0.3123   0.4834     2.568      389,357
     512    1.3669   1.2792   1.9583     2.670      374,566
    1024    2.6638   2.5175   3.6672     2.601      384,413

  single-flow latency    : 7.7 us (p99 11.3 us)
  peak throughput        : 389,357 flows/s (batch 128)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1152.8 ns/flow (1.153 us)
  C throughput           : 867,449 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 7.9 ms (2,264,343 packets/s)
  parse+aggregate        : 68.8 ms (259,274 packets/s, 67,891 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 74.7 ms (62,581 flows/s end-to-end)
  detected 4,574 attack flows / 100 benign

====================================================================
  6. ACCURACY (held-out, unseen-seed synthetic)
====================================================================
  flows evaluated        : 35,056 (unseen seeds)
  multiclass accuracy    : 99.54%
  macro F1               : 0.9844
  attack detection rate  : 99.91%
  benign false-pos rate  : 0.00%
  per-class recall:
    benign           100.0%
    portscan         100.0%
    synflood         100.0%
    icmpflood        100.0%
    udpflood         100.0%
    ssh_bruteforce    93.8%
    slowloris        100.0%
    mirai             94.2%
    xmas_scan        100.0%
    mqtt_flood       100.0%

  NOTE: synthetic traffic is separable; see CROSS_DATASET_FINDINGS.md
        for the honest cross-dataset numbers (in-domain 0.98 vs cross 0.45).

====================================================================
  7. MEMORY FOOTPRINT
====================================================================
  daemon runtime RSS     : 56.0 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 318.0 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. RASPBERRY PI 4 PROJECTION (host x 12)
====================================================================
  host                   : Apple M4 (arm64)
  ONNX single-flow (Pi)  : ~92.5 us/flow (~10,816 flows/s)
  C model (Pi)           : ~13.83 us/flow (~72,287 flows/s)
  feature extraction (Pi): ~21,606 packets/s (the real bottleneck on a live link)
  verdict                : easily real-time on a Pi 4 for home/IIoT link rates; sniffing/aggregation, not inference, is the limit.
```
