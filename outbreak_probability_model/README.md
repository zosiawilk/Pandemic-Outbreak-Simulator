# Final project model

This directory contains the complete final London and regional outbreak-
probability workflow. The notebooks are ordered in `FINAL_RUN_ORDER.md`; the
repository-level `README.md` documents inputs and installation. 

## Runtime modules

| File | Responsibility |
|---|---|
| `data_loader.py` | Select and validate the London onset-week series from a raw UKHSA downloaded CSV file. |
| `model.py` | Define inputs, parameters, compartment states and stochastic simulation. (Mathsy Maths version in code) |
| `london_calibration.py` | Calibrate the model, condition hidden states, validate forecasts and run sensitivity analysis. |
| `seasonal_rolling_audit.py` | Run the paired seasonal/non-seasonal historical London audit. |
| `regional_outbreak_forecast.py` | Transfer the London-fitted model to other regions. |
| `__init__.py` | Expose a small public simulator and data-loading API. |

## Final notebooks

```text
London_Calibration_6Week_Rolling_Poisson_Fit.ipynb
London_Seasonal_Poisson_Fit.ipynb
London_Strict_Forecast_Validation.ipynb
Poisson_Outbreak_Probability_Run.ipynb
London_Seasonal_Complete_Rolling_Audit.ipynb
Regional_Outbreak_Probability_Run.ipynb
London_Parameter_Sensitivity.ipynb
```

## Data flow

```text
processed population/protection/contact/mobility inputs
                         +
          observed weekly London cases
                         │
                         ▼
             non-seasonal calibration
                         │
                         ▼
                seasonal calibration
                   ┌─────┴───────────┐
                   ▼                 ▼
           held-out validation   final forecasts
                                     │
                        ┌────────────┼────────────┐
                        ▼            ▼            ▼
                  rolling audit  regional run  sensitivity
```

Steps 1 and 2 create the fitted parameter files used by later all-data
workflows. The strict held-out validation fits its own training-only vectors
and does not load those all-data fits.

Step 2 reads the all-series seasonal vector under
`experiments/measles/London/calibration_6week_rolling_poisson_seasonal_all_series_117_windows/`.
That directory is kept because the project (dissertation) also reports the comparison
between fitting with and without the exceptional 2024 wave.

## Development history

Some of my failed/step/discarded files are in the `archive/development_history/` directory. That directory is ignored
by Git and is not required by any final notebooks (these were mostly experiments run throughout the duration of the project. 

