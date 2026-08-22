# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.1-arm64-arm-64bit
python=3.10.14  time=2026-08-22 15:12

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
       1    0.0081   0.0078   0.0112     8.074      123,853
       8    0.0468   0.0467   0.0551     5.852      170,876
      32    0.1490   0.1447   0.1808     4.657      214,735
      64    0.1778   0.1744   0.2251     2.778      359,993
     128    0.3226   0.3070   0.4072     2.520      396,763
     512    1.2703   1.1990   1.5730     2.481      403,058
    1024    2.5516   2.4208   3.0677     2.492      401,322

  single-flow latency    : 8.1 us (p99 11.2 us)
  peak throughput        : 403,058 flows/s (batch 512)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1135.8 ns/flow (1.136 us)
  C throughput           : 880,460 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 7.8 ms (2,274,996 packets/s)
  parse+aggregate        : 82.8 ms (215,601 packets/s, 56,455 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 89.5 ms (52,249 flows/s end-to-end)
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
  daemon runtime RSS     : 55.1 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 306.1 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. TARGET-HARDWARE ACCEPTANCE GATE
====================================================================
  measured host          : macOS-26.6.1-arm64-arm-64bit
  Raspberry Pi result    : NOT MEASURED
  projection             : intentionally omitted; host scaling is not evidence
  acceptance gate        : run this benchmark on the target Pi and retain the identity, hashes, raw output and soak record in deploy/PI_ACCEPTANCE.md
```
