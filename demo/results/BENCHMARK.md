# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.2-arm64-arm-64bit
python=3.10.14  time=2026-09-30 13:36

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
       1    0.0144   0.0152   0.0220    14.429       69,306
       8    0.0464   0.0455   0.0581     5.802      172,366
      32    0.1515   0.1457   0.1767     4.734      211,230
      64    0.1749   0.1642   0.2413     2.732      365,976
     128    0.3355   0.3144   0.4565     2.621      381,571
     512    1.3000   1.2284   1.6663     2.539      393,848
    1024    2.5810   2.4920   3.1861     2.521      396,744

  single-flow latency    : 14.4 us (p99 22.0 us)
  peak throughput        : 396,744 flows/s (batch 1024)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1106.3 ns/flow (1.106 us)
  C throughput           : 903,930 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 9.3 ms (1,925,065 packets/s)
  parse+aggregate        : 83.4 ms (213,947 packets/s, 56,022 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 92.6 ms (50,454 flows/s end-to-end)
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
  daemon runtime RSS     : 56.3 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 315.2 MB (harness — imports pandas/xgboost; NOT the daemon)
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
