# Conference paper (IEEE format)

`main.tex` is the paper draft in the IEEE conference template (`IEEEtran`,
`conference` option). `references.bib` holds the bibliography, which is
formatted with `IEEEtran.bst`. Every number in the paper comes from evidence
committed in this repository. The table at the end lists the source of each.

## Build

**Overleaf (easiest):** create a blank project, upload `main.tex`,
`references.bib` and the `data/` folder, and set the compiler to pdfLaTeX.
`IEEEtran` and every package used are already installed on Overleaf.

**Locally** (TeX Live with `texlive-publishers` and `texlive-science`):

```bash
python paper/make_data.py        # regenerate data/soak.dat from the raw soak CSV
cd paper && latexmk -pdf main.tex
```

## Before submission

1. **Fill every red `[TBD: ...]`.** Then change `\finalfalse` to `\finaltrue`
   near the top of `main.tex`. In that mode, the build stops with an error if
   any `[TBD]` is left.
   - Author e-mails and author order. Confirm the guide's name and initials.
   - Section V-F, the cross-dataset results. Fill this only after
     `python code/run_sfaf_rerun.py` passes its gate. Add the 11×11 ROC-AUC
     matrix as a figure (`demo/results/cross_dataset_auc_matrix.png`), and
     report the mean in-domain and off-diagonal ROC-AUC and MCC.
   - The repository URL, if the repository is made public. Also complete the
     Acknowledgment.
2. **Use the venue's own template.** Most IEEE conferences accept this
   `IEEEtran` conference layout. Check the call for papers for the page limit
   (often 4–6 pages), anonymous submission, and any template change. For
   double-blind review, remove the author block and the repository URL.
3. **Check the PDF with IEEE PDF eXpress** if the venue asks for it. It
   checks embedded fonts and the PDF version.
4. **Verify every reference** against its DOI or the publisher's page:
   authors, title, venue, volume, pages and year. Add only papers you have
   read. The entries for TON_IoT, WUSTL-IIoT-2021 and MQTT-IoT-IDS2020
   replace incorrect citations found in the September report review.
5. **Similarity check (Turnitin or the venue's iThenticate).** The text is
   new prose and is not copied from the README or the reports. Your earlier
   reports can still match it, so review the matches and rephrase any that
   come from your own earlier writing.

## IEEE style rules this draft follows

- The `conference` class option, two columns, 10 pt. No page numbers, headers
  or footers, because IEEE adds these.
- The abstract is one paragraph of 233 words (IEEE asks for 150–250), with
  no citations, equations or abbreviations left undefined. It is followed by
  Index Terms.
- Abbreviations are defined at first use (IoT, ML). Units are typeset with
  `siunitx`, with a space between number and unit.
- Table captions are above the table, in IEEE's small caps. Figure captions
  are below the figure. Every table and figure is cited in the text before
  it appears.
- Figures are vector graphics (TikZ/pgfplots) and readable in grayscale.
- References are numbered in order of first citation. `IEEEtran.bst`
  produces IEEE reference format.
- The acknowledgment is spelled "Acknowledgment", as IEEE requires.

## Where each number comes from

| Paper item | Source in the repository |
|---|---|
| Table I, synthetic evaluations | `models/live_meta.json`, `demo/results/heldout_report.csv`, `demo/results/BENCHMARK.md` §6 |
| Table II, Pi 2 column and the parity figures | `reports/pi2-acceptance-2026-09-30/README.md` |
| Table II, host column | `demo/results/BENCHMARK.md` (10 Nov 2026 run) |
| Table III, attacks | `reports/pi2-live-2026-09-30/raw/attacks.json` |
| Table IV, benign baselines | `reports/pi2-live-2026-09-30/README.md` §2 and §2b |
| Fig. 2 and the soak text | `reports/pi2-live-2026-09-30/raw/soak.csv` (via `make_data.py`) |
| Reboot and IPS sentences | `reports/pi2-live-2026-09-30/README.md` §1 and §4 |
| Section V-F | `demo/results/cross_dataset_metrics_long.csv` (after the rerun) |
