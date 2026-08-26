# Archived all-series seasonal calibration

This directory preserves the completed descriptive sensitivity run that used
all 123 London observations and 117 overlapping six-week windows, including the
exceptional 2024 wave. It must not be confused with the selected post-wave
design, which excludes observations through 4 August 2024 and uses 86 windows.

## Retained result from this sensitivity run

- Method: bounded random search with local/Powell refinement
- Selected Poisson NLL: 3.9224787271
- Seasonal amplitude: 0.5753886179
- Seasonal peak week: 14.5757008825
- Optional numerical-gradient diagnostic: skipped

The full selected vector is stored in
`03_seasonal_poisson_fitted_parameters.csv`. Screenshots of the notebook output
and the full-series seasonal-versus-non-seasonal curve are retained in
`screenshots/`.

## Metric-definition warning

The archived `04_seasonal_vs_nonseasonal_calibration.csv` was produced by the
earlier comparison cell. Its `weighted_RMSE` is a weighted mean of RMSE values
calculated separately within each six-week window. It is not the corrected
global overlap-weighted RMSE calculated from all flattened weekly residuals.
The Poisson NLL and the fitted vectors remain valid for documenting this
sensitivity experiment, but the archived RMSE values must not be inserted into
the final like-for-like post-wave comparison table.

The current live seasonal output directory had its settings cell rerun at
16:13 on 24 August 2026, so its `00_seasonal_poisson_fit_settings.csv` describes
the new 86-window post-wave design while its remaining outputs still describe
this archived 117-window run. A fresh kernel and Run All are required to make
the live directory internally consistent.
