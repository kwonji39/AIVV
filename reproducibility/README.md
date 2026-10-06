# Data and offline reanalysis

This directory contains the available decision records and source data for the reported UUV and spacecraft analyses. The AIVV implementation and live run scripts are in the repository root.

## Contents

- `data/uuv/`: the three REMUS 100 telemetry streams used in the evaluated missions. The repository root also contains these input streams.
- `raw_traces/`: 600 compressed UUV run records and 60 compressed spacecraft run records, including sample-level decisions and agent outputs.
- `uuv_traces.json` and `satellite_traces.json`: per-run indices of decisions, labels, scores, source files, and original-file SHA-256 hashes.
- `seed_manifest.csv` and `cohort_spec.json`: archived run inventory and evaluated spacecraft cohort.
- `source_data/` and `review_reanalysis_20261005/analysis/`: per-run scores, descriptive summaries, and verification outputs.
- `revision_20260916/analysis/author_reported_counts.json`: author-reported staged-ablation counts. Only 23 unique council-only raw records are available in `revision_20260916/council_traces/`; they are not the complete 75-run council-only cohorts.
- `reanalyze.py`, `review_reanalysis_20261005/audit.py`, and `review_reanalysis_20261005/verify_independently.py`: offline scoring and integrity checks.

The UUV detector-accuracy summary and the full council-only stage counts retain their originally reported values. The available records do not independently reconstruct every reported Experiment 1 aggregate. The archived spacecraft records cover all 15 evaluated runs for each of the four channels.

## Offline checks

Run from this directory with Python 3.12 or later:

```sh
python3 reanalyze.py
python3 review_reanalysis_20261005/audit.py .
python3 review_reanalysis_20261005/verify_independently.py .
```

The first command verifies the original hashes of all 660 available records and recomputes the spacecraft run scores. The next two commands recalculate the available UUV results directly from archived run records and cross-check the summaries. These commands do not call hosted language models.

## Live runs and external telemetry

The repository root contains the implementation, environment files, and UUV and spacecraft run scripts. The SMAP/MSL telemetry and expert labels come from Hundman et al.: <https://github.com/khundman/telemanom>. Follow that project's data access instructions for the four evaluated channels E-8, F-5, T-4, and D-1. The third-party channel arrays are not redistributed here. Exact live re-execution depends on access to the hosted model identifiers documented in the manuscript; changing model assignments produces a new experiment.
