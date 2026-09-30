# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.2-arm64-arm-64bit
python=3.10.14  time=2026-09-30 21:58

====================================================================
  1. MODEL PARAMETERS & FOOTPRINT
====================================================================
  Features               : 22
  Classes                : 10  (benign, portscan, synflood, icmpflood, udpflood, ssh_bruteforce, slowloris, mirai, xmas_scan, mqtt_flood)
  Boosted trees          : 1200
  Total nodes            : 2,780
  ONNX model size        : 91.8 KB
  C header size          : 103.0 KB (~49 KB const data)
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
       1    0.0091   0.0082   0.0126     9.113      109,734
       8    0.0449   0.0442   0.0547     5.612      178,201
      32    0.2312   0.1647   0.5161     7.225      138,402
      64    0.1862   0.1740   0.2461     2.909      343,780
     128    0.3404   0.3255   0.4585     2.659      376,036
     512    1.3138   1.3110   1.6033     2.566      389,702
    1024    2.6999   2.5378   5.2319     2.637      379,271

  single-flow latency    : 9.1 us (p99 12.6 us)
  peak throughput        : 389,702 flows/s (batch 512)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1138.1 ns/flow (1.138 us)
  C throughput           : 878,623 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  (skipped — no demo pcap)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  (section FAILED — FileNotFoundError: [Errno 2] No such file or directory: '/Users/malharfalke/IOT-IDS-scope/data/pcaps/demo_mixed.pcap')

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
  benchmark process RSS  : 306.0 MB (harness — imports pandas/xgboost; NOT the daemon)
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
