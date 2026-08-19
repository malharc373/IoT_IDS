# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.1-arm64-arm-64bit
python=3.10.14  time=2026-08-19 23:37

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
       1    0.0067   0.0067   0.0073     6.740      148,363
       8    0.0451   0.0448   0.0530     5.636      177,422
      32    0.1527   0.1518   0.1631     4.772      209,544
      64    0.1671   0.1551   0.2307     2.612      382,913
     128    0.3229   0.2988   0.4435     2.523      396,422
     512    1.2836   1.2006   1.7153     2.507      398,883
    1024    2.5582   2.3862   3.4164     2.498      400,282

  single-flow latency    : 6.7 us (p99 7.3 us)
  peak throughput        : 400,282 flows/s (batch 1024)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1214.7 ns/flow (1.215 us)
  C throughput           : 823,221 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  (section skipped — ValueError: too many values to unpack (expected 2))

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 100.7 ms (46,435 flows/s end-to-end)
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
  daemon runtime RSS     : 57.3 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 312.7 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. RASPBERRY PI 4 PROJECTION (host x 12)
====================================================================
  host                   : Apple M4 (arm64)
  ONNX single-flow (Pi)  : ~80.9 us/flow (~12,364 flows/s)
  C model (Pi)           : ~14.58 us/flow (~68,602 flows/s)
  verdict                : easily real-time on a Pi 4 for home/IIoT link rates; sniffing/aggregation, not inference, is the limit.
```
