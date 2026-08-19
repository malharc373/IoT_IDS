# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=macOS-26.6.1-arm64-arm-64bit
python=3.10.14  time=2026-08-19 23:11

====================================================================
  1. MODEL PARAMETERS & FOOTPRINT
====================================================================
  Features               : 22
  Classes                : 10  (benign, portscan, synflood, icmpflood, udpflood, ssh_bruteforce, slowloris, mirai, xmas_scan, mqtt_flood)
  Boosted trees          : 1200
  Total nodes            : 2,920 / 2,060 leaves
  ONNX model size        : 96.2 KB
  C header size          : 106.8 KB (~45 KB const data)
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
       1    0.0111   0.0086   0.0275    11.068       90,353
       8    0.0451   0.0445   0.0579     5.642      177,228
      32    0.1518   0.1480   0.1974     4.743      210,830
      64    0.1795   0.1744   0.2522     2.805      356,534
     128    0.3623   0.3309   0.8483     2.830      353,299
     512    1.3406   1.3067   1.7319     2.618      381,922
    1024    2.9426   2.4851   7.4699     2.874      347,989

  single-flow latency    : 11.1 us (p99 27.5 us)
  peak throughput        : 381,922 flows/s (batch 512)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1260.8 ns/flow (1.261 us)
  C throughput           : 793,119 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  (section skipped — ValueError: too many values to unpack (expected 2))

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,735 flows classified in 103.3 ms (45,833 flows/s end-to-end)
  detected 4,636 attack flows / 99 benign

====================================================================
  6. ACCURACY (held-out, unseen-seed synthetic)
====================================================================
  flows evaluated        : 35,154 (unseen seeds)
  multiclass accuracy    : 99.37%
  macro F1               : 0.9724
  attack detection rate  : 100.00%
  benign false-pos rate  : 36.47%
  per-class recall:
    benign            63.5%
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
        for the honest cross-dataset numbers: in-domain ROC-AUC 0.996 vs
        cross-domain 0.514 against a chance baseline of 0.500 (MCC -0.002).

====================================================================
  7. MEMORY FOOTPRINT
====================================================================
  daemon runtime RSS     : 55.9 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 320.7 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. RASPBERRY PI 4 PROJECTION (host x 12)
====================================================================
  host                   : Apple M4 (arm64)
  ONNX single-flow (Pi)  : ~132.8 us/flow (~7,529 flows/s)
  C model (Pi)           : ~15.13 us/flow (~66,093 flows/s)
  verdict                : easily real-time on a Pi 4 for home/IIoT link rates; sniffing/aggregation, not inference, is the limit.
```
