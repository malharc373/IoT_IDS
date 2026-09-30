# Review of the B.Tech project report (30 September 2026)

Reviewed: [`reports/btech/2026-09-30_BTech_Project_Report.pdf`](../btech/2026-09-30_BTech_Project_Report.pdf)
(42 pages, generated 30 Sep 2026 14:48). Every number and claim was checked
against branch `integration/final-capstone` at `46a9cd3`. The slide deck is
built from this report, so each fix below should also reach the slides.

Items are ordered by how badly they would hurt in an evaluation.

## 1. Factual errors against the repository

| # | Where | Report says | Repository says | Fix |
|---|---|---|---|---|
| 1 | Abstract, §4.6, §5.1, §6.1, §7.1 | 99.65% accuracy on ~12.2k flows from 70 unseen scenarios | Two different evaluations are merged. 99.65% / macro-F1 0.9961 / Mirai 94.2% come from `demo/benchmark.py` §6: **35,057 flows from unseen generator seeds**. The 12,191-flow / 70-scenario figure is the training script's scenario-held-out test split, which scores **100%** (`models/live_meta.json`, `demo/results/live_classification_report.csv`). | State both evaluations separately, each with its flow count and source. |
| 2 | Fig. 6.2 | Held-out matrix, acc 99.9%, ssh_bruteforce 818/867 correct, Mirai all correct | The committed figure (`demo/results/heldout_confusion_matrix.png`) shows **100.0% on 47,485 flows**. The report's image is an older render. It also contradicts the text: it makes ssh_bruteforce the hardest class, the text says Mirai. | Replace the image, and name the run it comes from (`demo/validate.py`). The report now carries three "held-out" numbers (99.65, 99.9, 100); reconcile them. |
| 3 | Fig. 6.1 | Training-test matrix totalling 10,982 flows, 612 benign | Current split is 12,191 flows, 1,679 benign. | Replace with `demo/results/live_confusion_matrix.png`. |
| 4 | Table 6.1, §6.2, §7.1, Fig. 6.3 | 10.5 µs (p99 16.2), 407k flows/s, C 1.16 µs, 52.7k flows/s end-to-end, 55.4 MB, "5 Sep run" | Committed `demo/results/BENCHMARK.md` is the 30 Sep run: **14.4 µs (p99 22.0), 396,744 flows/s, C 1.106 µs, 213,947 packets/s, 50,454 flows/s end-to-end, 56.3 MB**. The 5 Sep output is no longer in the repository, so Appendix A cannot reproduce the report. | Quote the committed run, or archive the 5 Sep output next to the report. |
| 5 | §5.4, Table 6.1, §7.1 | 67 automated tests | 70 on `integration/final-capstone`; 74 on `feat/c-inference-fallback`. | Update the count, or say "70+". |
| 6 | Timeline 31 Aug, §5.4, §7.1 | CodeQL is set up | No CodeQL workflow or configuration exists (`.github/` has `ci.yml`, `release.yml`, `dependabot.yml`). | Remove CodeQL, or add the workflow before claiming it. |
| 7 | Table 4.2 | TCP flags SYN, ACK, FIN, RST, PSH, URG; packet-length "variance" | The contract has SYN/FIN/RST/ACK ratios only, and standard deviation (`std_pkt_len`, `std_iat`). As written the table adds up to 24 features. Appendix A.1 lists the correct 22, so the report contradicts itself. | Correct the table from `models/live_meta.json`. |
| 8 | §2.2, ref. [8], §4.9, Table 5.1 | ONNX Runtime runs the model "on laptop and Pi"; the Pi target is a Pi 4 | The lab Pi is a **Raspberry Pi 2 Model B** (32-bit armv7l, Python 3.13). onnxruntime publishes no 32-bit ARM wheels. The model runs there through the C export (see [Pi 2 results](../pi2-acceptance-2026-09-30/README.md)). | Correct the claim and fill the Table 5.1 rows that are now measured. |
| 9 | §2.1, §5.1 | 11 datasets, but only 8 named | The loaders (`code/multidataset.py`) are CICIDS2017, UNSW-NB15, TON_IoT, Bot-IoT, CICDDoS2019, IoTID20, X-IIoTID, MQTT-IoT-IDS2020, CIC-IoT-2023, WUSTL-IIoT and IoT-23. | Name all 11. The "MQTT and related captures" are the MQTT-IoT-IDS2020 dataset. |

## 2. References to verify

Check each against the original publication before submission.

- **[1]** credits TON_IoT to "A. Alhazmi et al.". TON_IoT is published by
  the UNSW Canberra Cyber group (Moustafa et al.). Verify authorship.
- **[5]** is Edge-IIoTset (Ferrag et al., IEEE Access 2022), cited for
  WUSTL-IIoT. They are different datasets; WUSTL-IIoT-2021 comes from
  Washington University in St. Louis. Cite the WUSTL paper instead.
- **[10]** is the MQTT 5.0 specification, cited as the MQTT attack dataset.
  Cite the MQTT-IoT-IDS2020 dataset paper for the data; keep the spec only
  where the protocol itself is discussed.
- **[4]** "Y. Chen et al., Applied Sciences / IDS literature, 2023" has no
  title, volume or DOI and may be a placeholder. Verify or replace.
- **[8]** says ONNX Runtime was used "on laptop and Pi"; false for this Pi.
- §2.2 claims prior edge-IDS work shows "sub-100 KB boosted-tree models can
  meet microsecond-scale budgets" with no citation.
- Chapter 1 cites dataset papers [11, 9, 1, 4, 8] as support for the
  team's own Semester 6 work; cite the Semester 6 report instead.
- IoTID20, X-IIoTID, CIC-IoT-2023, Mirai and slowloris have no citation.

## 3. Consistency and presentation

- §2.3 contains a stray internal reference: "Section 2.9 of the source
  progress file". Remove it.
- §1.4 says the report covers 3 Aug–5 Sep; the title page and timeline run
  to 30 Sep.
- The certificate has no name for the Head of Department.
- Front matter is numbered 2–9 in arabic numerals and Chapter 1 restarts
  at 1. Use roman numerals for the front matter.
- Page 3 is an almost empty continuation of the abstract.
- Appendix A is a reproduction guide with no commands. Add
  `bash demo/run_demo.sh`, `python demo/benchmark.py` and
  `python src/dashboard.py`.
- Fig. 4.1 is a row of text boxes; a real block diagram will read better on
  a slide.

## 4. Algorithms that do not match the code

- **A.2 step 2** says the loop "collects expired flows". The daemon rescores
  every flow that changed since the last flush (`VerdictCache`), not only
  expired ones.
- **A.2 step 3** merges two separate settings: `--min-conf` gates whether an
  incident is alerted; `--abstain-conf` relabels a low-confidence verdict as
  `unknown`.
- **A.2 step 4** says incidents are grouped per (source IP, attack
  category). They are grouped per (source IP, attack **type**); a host doing
  a SYN flood and a UDP flood raises two incidents.
- **A.3** leaves out strike corroboration: by default the IPS blocks a source
  only after 3 qualifying incidents within 120 s (`--ips-strikes`,
  `--ips-strike-window`). This is the main IPS safety feature and belongs in
  the algorithm.
- §4.7 "inline bridge mode" is the `--ips-scope network` setting (INPUT +
  FORWARD). The bridge itself is an operator setup documented in
  `deploy/README_PI.md` §6.

## 5. Claims worth rewording

- "Accuracy without destination port: unchanged" cannot show anything when
  accuracy is already 100% (a ceiling effect). Say that the ablation is
  uninformative on this corpus.
- The cross-dataset result is withdrawn in full, but
  `demo/results/CROSS_DATASET_FINDINGS.md` notes that the off-diagonal
  measurement (mean ROC-AUC 0.509 over 110 pairs) was a genuine
  cross-dataset test; only the diagonal was resubstituted. Four protocol
  issues may still move the exact value, so it can be mentioned only as
  preliminary: transfer was near chance.
- "26 findings (F01–F26)" could not be checked against the repository,
  which references IDs up to F20. Confirm the count against the 22 August
  technical report.

## 6. For the presentation

- Open with the live Pi demo; it now works end to end (see the Pi 2 results).
- Expect "why is accuracy 100%?". Answer it before it is asked: the
  generated attacks are separable by construction, and there is no
  real-traffic number yet (P0 for October).
- Keep every speed number labelled with the host it was measured on.
