# IoT-IDS system benchmark

```
IoT-IDS SYSTEM BENCHMARK   host=Linux-6.18.44-fc-v310-x86_64-with-glibc2.39
python=3.10.20  time=2026-11-10 00:34

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
       1    0.0097   0.0094   0.0239     9.707      103,018
       8    0.0671   0.0652   0.0880     8.392      119,165
      32    0.3026   0.2742   0.5856     9.455      105,760
      64    0.2193   0.1864   0.5404     3.427      291,784
     128    0.3885   0.3498   0.7268     3.035      329,492
     512    1.2883   1.2378   2.1281     2.516      397,410
    1024    2.5078   2.4198   4.5634     2.449      408,321

  single-flow latency    : 9.7 us (p99 23.9 us)
  peak throughput        : 408,321 flows/s (batch 1024)

====================================================================
  3. NATIVE C MODEL (MCU PATH)
====================================================================
  C ids_predict latency  : 1597.4 ns/flow (1.597 us)
  C throughput           : 626,009 flows/s (single thread)
  runtime deps           : none (pure C99, ~130 B stack)

====================================================================
  4. FEATURE EXTRACTION THROUGHPUT
====================================================================
  pcap                   : demo_mixed.pcap (17,850 packets -> 4,674 flows)
  parse+read             : 17.6 ms (1,016,893 packets/s)
  parse+aggregate        : 138.4 ms (129,003 packets/s, 33,779 flows/s)

====================================================================
  5. END-TO-END (pcap -> verdicts)
====================================================================
  4,674 flows classified in 131.4 ms (35,575 flows/s end-to-end)
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
        cross-dataset (corrected rerun): in-domain AUC 0.979, cross-domain 0.497.

====================================================================
  7. MEMORY FOOTPRINT
====================================================================
  daemon runtime RSS     : 57.8 MB (onnxruntime + numpy only, clean process)
  benchmark process RSS  : 316.5 MB (harness — imports pandas/xgboost; NOT the daemon)
  edge runtime deps      : onnxruntime + numpy (+ scapy for live sniff)
  MCU C model RAM        : ~130 bytes stack, 0 heap

====================================================================
  8. TARGET HARDWARE
====================================================================
  measured host          : Linux-6.18.44-fc-v310-x86_64-with-glibc2.39
  this run               : NOT a Raspberry Pi; numbers above are host-only
  projection             : intentionally omitted; host scaling is not evidence
  Pi 2 measurements      : run on the board itself; see reports/pi2-acceptance-2026-09-30/ and reports/pi2-live-2026-09-30/
```
