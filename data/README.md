# Kepler Objects of Interest data

`koi_cumulative.csv` is a 30-column snapshot of the NASA Exoplanet Archive
Kepler Objects of Interest cumulative table used by the tabular notebook.

- Source: <https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html>
- Retrieval API: NASA Exoplanet Archive TAP service
- Rows: 9,564
- Target: `koi_disposition`
- Retrieved: 2026-09-26

The notebook uses this committed snapshot for deterministic `Run all` results.
If the file is absent, it downloads the same selected columns from NASA TAP and
caches them here.

## ViCTSD Vietnamese text data

`victsd_train.csv`, `victsd_valid.csv`, and `victsd_test.csv` are committed
copies of the official UIT-ViCTSD splits used by the text-classification
notebook.

- Source: <https://github.com/tarudesu/ViCTSD>
- Dataset: Vietnamese Constructive and Toxic Speech Detection (ViCTSD)
- Rows: 10,000 human-annotated social-media comments from 10 news topics
- Official split: 7,000 train / 2,000 validation / 1,000 test
- Targets: `Toxicity` and `Constructiveness`
- Retrieved: 2026-09-28

The notebook uses comment text only. It audits and excludes normalized comments
that also occur in the training split from validation and test evaluation to
avoid direct text leakage.
