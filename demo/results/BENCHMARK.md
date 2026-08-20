# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.1-arm64-arm-64bit
python=3.10.14  time=2026-08-21 05:01

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
       1    0.0081   0.0077   0.0170     8.086      123,677
       8    0.0430   0.0403   0.0982     5.380      185,882
      32    0.1643   0.1657   0.1928     5.134      194,779
      64    0.1809   0.1653   0.2760     2.827      353,790
     128    0.3348   0.3148   0.4724     2.616      382,317
     512    1.2957   1.2255   1.6706     2.531      395,154
    1024    2.5759   2.4719   3.2171     2.516      397,528

  single-flow latency    : 8.1 us (p99 17.0 us)
  peak throughput        : 397,528 flows/s (batch 1024)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1120.0 ns/flow (1.120 us)
  C throughput           : 892,873 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 8.0 ms (2,229,415 packets/s)
  parse+aggregate        : 81.3 ms (219,576 packets/s, 57,496 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 86.4 ms (54,094 flows/s end-to-end)
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
        for the honest cross-dataset numbers, over 11 datasets (110 ordered pairs):
        in-domain ROC-AUC 0.995 vs cross-domain 0.509 against a chance baseline of 0.500
        (cross-domain MCC -0.007, chance 0.000).

====================================================================
  7. MEMORY FOOTPRINT
====================================================================
  daemon runtime RSS     : 57.0 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 316.2 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. RASPBERRY PI 4 PROJECTION (host x 12)
====================================================================
  host                   : Apple M4 (arm64)
  ONNX single-flow (Pi)  : ~97.0 us/flow (~10,306 flows/s)
  C model (Pi)           : ~13.44 us/flow (~74,406 flows/s)
  feature extraction (Pi): ~18,298 packets/s (the real bottleneck on a live link)
  verdict                : easily real-time on a Pi 4 for home/IIoT link rates; sniffing/aggregation, not inference, is the limit.
```
