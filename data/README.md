# Surveillance-data snapshot

`ukhsa-measles_weekly_cases.csv` is the raw UKHSA dashboard export read by the
final London forecast. The committed snapshot was refreshed on 26 August 2026 (SO IT IS NOT VISIBLE IN THE DISSERTATION RESULTS!).

```text
SHA-256: 67bd647db7671a9a6550ffc2d93ace923fef3f8f16422ef8a09c9dc66be702c6
London onset-week coverage in the raw export: 2024-01-01 to 2026-08-03
```

The notebook removes rows for which UKHSA sets
`in_reporting_delay_period=True`; therefore, the final forecast origin can be
earlier than the last raw date in the file. 

To refresh the forecast input:

1. Download a new CSV export of the UKHSA measles dashboard data.
2. Replace `ukhsa-measles_weekly_cases.csv` without changing its name.
3. Confirm that the columns documented in
   `outbreak_probability_model/data_loader.py` are present.
4. Restart the notebook kernel and run
   `Poisson_Outbreak_Probability_Run.ipynb` from the first cell.

The remaining CSVs in this directory belong to earlier preprocessing and
baseline experiments. They are not all direct inputs to the final eight-step
workflow (!!!) 

## Optional raw mobility inputs

The processed regional mobility matrices required by the final model are
committed under `outbreak_risk_model/graph_outputs/`. To rebuild them, place
the untracked Census source files at:

```text
data/raw/odwp/odwp01ew/ODWP01EW_RGN.csv
data/raw/odst/odst01ew.zip
```
