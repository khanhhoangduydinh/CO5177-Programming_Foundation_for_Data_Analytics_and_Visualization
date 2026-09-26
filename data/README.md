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

