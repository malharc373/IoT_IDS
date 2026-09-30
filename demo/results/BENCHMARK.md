# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.2-arm64-arm-64bit
python=3.10.14  time=2026-09-30 12:48

====================================================================
  1. MODEL PARAMETERS & FOOTPRINT
====================================================================
  Features               : 22
  Classes                : 10  (benign, portscan, synflood, icmpflood, udpflood, ssh_bruteforce, slowloris, mirai, xmas_scan, mqtt_flood)
  Boosted trees          : 1200
  Total nodes            : 2,780
  ONNX model size        : 91.8 KB
  C header size          : 102.7 KB (~43 KB const data)
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
       1    0.0111   0.0098   0.0189    11.130       89,848
       8    0.0417   0.0417   0.0485     5.212      191,857
      32    0.1523   0.1510   0.1644     4.761      210,046
      64    0.1795   0.1683   0.2497     2.805      356,450
     128    0.3293   0.3129   0.4146     2.573      388,693
     512    1.2789   1.2096   1.6058     2.498      400,352
    1024    2.5414   2.3980   3.1575     2.482      402,921

  single-flow latency    : 11.1 us (p99 18.9 us)
  peak throughput        : 402,921 flows/s (batch 1024)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1125.5 ns/flow (1.126 us)
  C throughput           : 888,470 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 8.6 ms (2,084,834 packets/s)
  parse+aggregate        : 82.9 ms (215,292 packets/s, 56,374 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 88.8 ms (52,612 flows/s end-to-end)
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
  daemon runtime RSS     : 55.9 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 325.6 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. TARGET-HARDWARE ACCEPTANCE GATE
====================================================================
  measured host          : macOS-26.6.2-arm64-arm-64bit
  Raspberry Pi result    : NOT MEASURED
  projection             : intentionally omitted; host scaling is not evidence
  acceptance gate        : run this benchmark on the target Pi and retain the identity, hashes, raw output and soak record in deploy/PI_ACCEPTANCE.md
```
