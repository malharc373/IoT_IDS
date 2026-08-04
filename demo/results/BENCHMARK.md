# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6-arm64-arm-64bit
python=3.10.14  time=2026-08-04 14:40

====================================================================
  1. MODEL PARAMETERS & FOOTPRINT
====================================================================
  Features               : 22
  Classes                : 10  (benign, portscan, synflood, icmpflood, udpflood, ssh_bruteforce, slowloris, mirai, xmas_scan, mqtt_flood)
  Boosted trees          : 1200
  Total nodes            : 2,720 / 1,960 leaves
  ONNX model size        : 90.2 KB
  C header size          : 101.1 KB (~42 KB const data)
  In-domain metrics      : {'multiclass_accuracy': 1.0, 'macro_f1': 1.0, 'binary_accuracy': 1.0, 'binary_f1_attack': 1.0}

====================================================================
  2. ONNX INFERENCE LATENCY & THROUGHPUT
====================================================================
   batch   mean_ms   p50_ms   p99_ms   us/flow      flows/s
       1    0.0099   0.0074   0.0177     9.912      100,884
       8    0.0453   0.0450   0.0485     5.658      176,728
      32    0.1629   0.1630   0.1706     5.089      196,495
      64    0.1684   0.1576   0.2546     2.631      380,100
     128    0.3320   0.3084   0.5032     2.594      385,519
     512    1.3450   1.2412   1.9692     2.627      380,666
    1024    2.6898   2.5011   3.7800     2.627      380,692

  single-flow latency    : 9.9 us (p99 17.7 us)
  peak throughput        : 385,519 flows/s (batch 128)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1123.9 ns/flow (1.124 us)
  C throughput           : 889,727 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 10.4 ms (1,721,768 packets/s)
  parse+aggregate        : 72.8 ms (245,061 packets/s, 64,169 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 76.7 ms (60,971 flows/s end-to-end)
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
  daemon runtime RSS     : 57.8 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 319.5 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. RASPBERRY PI 4 PROJECTION (host x 12)
====================================================================
  host                   : Apple M4 (arm64)
  ONNX single-flow (Pi)  : ~118.9 us/flow (~8,407 flows/s)
  C model (Pi)           : ~13.49 us/flow (~74,144 flows/s)
  feature extraction (Pi): ~20,422 packets/s (the real bottleneck on a live link)
  verdict                : easily real-time on a Pi 4 for home/IIoT link rates; sniffing/aggregation, not inference, is the limit.
```
