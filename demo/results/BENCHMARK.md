# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.1-arm64-arm-64bit
python=3.10.14  time=2026-08-20 00:44

====================================================================
  1. MODEL PARAMETERS & FOOTPRINT
====================================================================
  Features               : 22
  Classes                : 10  (benign, portscan, synflood, icmpflood, udpflood, ssh_bruteforce, slowloris, mirai, xmas_scan, mqtt_flood)
  Boosted trees          : 1200
  Total nodes            : 2,910 / 2,055 leaves
  ONNX model size        : 95.9 KB
  C header size          : 106.5 KB (~45 KB const data)
  In-domain metrics      : {'multiclass_accuracy': 0.9999, 'macro_f1': 0.9999, 'binary_accuracy': 0.9999, 'binary_f1_attack': 0.9999}
  Split                  : GroupShuffleSplit on scenario (no scenario spans the split)
  Without dst_port       : acc=0.9999 (delta +0.0000) — the model is not a port lookup
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
       1    0.0068   0.0068   0.0075     6.801      147,036
       8    0.0423   0.0421   0.0438     5.282      189,339
      32    0.1430   0.1429   0.1476     4.470      223,719
      64    0.1674   0.1555   0.2128     2.616      382,261
     128    0.3186   0.2960   0.4456     2.489      401,790
     512    1.2981   1.2087   1.8405     2.535      394,429
    1024    2.6027   2.4396   3.5545     2.542      393,441

  single-flow latency    : 6.8 us (p99 7.5 us)
  peak throughput        : 401,790 flows/s (batch 128)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1214.6 ns/flow (1.215 us)
  C throughput           : 823,330 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  (section skipped — ValueError: too many values to unpack (expected 2))

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 100.2 ms (46,651 flows/s end-to-end)
  detected 4,575 attack flows / 99 benign

====================================================================
  6. ACCURACY (held-out, unseen-seed synthetic)
====================================================================
  flows evaluated        : 35,057 (unseen seeds)
  multiclass accuracy    : 99.76%
  macro F1               : 0.9974
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
    mirai             96.1%
    xmas_scan        100.0%
    mqtt_flood       100.0%

  NOTE: synthetic traffic is separable; see CROSS_DATASET_FINDINGS.md
        for the honest cross-dataset numbers: in-domain ROC-AUC 0.996 vs
        cross-domain 0.514 against a chance baseline of 0.500 (MCC -0.002).

====================================================================
  7. MEMORY FOOTPRINT
====================================================================
  daemon runtime RSS     : 56.1 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 314.9 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. RASPBERRY PI 4 PROJECTION (host x 12)
====================================================================
  host                   : Apple M4 (arm64)
  ONNX single-flow (Pi)  : ~81.6 us/flow (~12,253 flows/s)
  C model (Pi)           : ~14.57 us/flow (~68,611 flows/s)
  verdict                : easily real-time on a Pi 4 for home/IIoT link rates; sniffing/aggregation, not inference, is the limit.
```
