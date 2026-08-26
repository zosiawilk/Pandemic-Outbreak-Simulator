# Six-week measles outbreak probability model

This repository contains a probabilistic six-week measles forecasting project
for England. The final model is a stochastic
S/H/E/I/Q/D compartment model stratified by English region and age group. It
combines weekly surveillance, vaccination-derived protection, population,
age-contact and mobility inputs. 

This project was done for a dissertation submitted for a deegre of Master of Philosophy in Machine Learning Machine Intelligence at the University of Cambridge. 

## Start here

1. Project environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Read [`outbreak_probability_model/FINAL_RUN_ORDER.md`](outbreak_probability_model/FINAL_RUN_ORDER.md).
3. Run each listed notebook (The data is here, but the run order/read.me in data explains how to get the data to reproduce the experiments) 

The final notebooks are stored without execution output. The
reported results remain available as ordinary CSV, PNG and PDF files.

## Final workflow

| Step | Notebook | Purpose | Main Python dependency |
|---:|---|---|---|
| 0 | `outbreak_risk_model/Measles_Graph_Model.ipynb` | Optional input and mobility preprocessing | `odwp_mobility.py`, `odst_mobility.py`, `measles_graph.py` |
| 1 | `London_Calibration_6Week_Rolling_Poisson_Fit.ipynb` | Non-seasonal Poisson fitting | `london_calibration.py` |
| 2 | `London_Seasonal_Poisson_Fit.ipynb` | Seasonal Poisson fitting | `london_calibration.py` |
| 3 | `London_Strict_Forecast_Validation.ipynb` | Chronological out-of-sample evaluation | `london_calibration.py` |
| 4 | `Poisson_Outbreak_Probability_Run.ipynb` | Final London forecast | `data_loader.py`, `london_calibration.py` |
| 5 | `London_Seasonal_Complete_Rolling_Audit.ipynb` | Rolling audit | `seasonal_rolling_audit.py` |
| 6 | `Regional_Outbreak_Probability_Run.ipynb` | Non-London regional scenarios | `regional_outbreak_forecast.py` |
| 7 | `London_Parameter_Sensitivity.ipynb` |Parameter sensitivity | `london_calibration.py` |

All notebooks in steps 1–7 are in `outbreak_probability_model/`.

## Code map

```text
outbreak_probability_model/
├── data_loader.py                 UKHSA surveillance selection and validation
├── model.py                       region-age state, equations and simulation
├── london_calibration.py          fitting, conditioning, validation, forecast
│                                  and sensitivity functions
├── seasonal_rolling_audit.py      paired historical London audit
├── regional_outbreak_forecast.py  regional transfer and scenario forecasts
└── __init__.py                    small public simulator/data-loader API
```

Superseded fitting modules and the notebooks explicitly listed as non-final in
the run-order document are preserved locally under to show the progress of the project thoughout the Easter/summer term. 
`archive/development_history/` . Final
notebooks do not depend on it.

## Inputs

The final model reads these inputs:

```text
data/ukhsa-measles_weekly_cases.csv
experiments/measles/London/observed_weekly_cases.csv
outbreak_probability_model/data/london_age_confirmed_cases.csv
experiments/measles_local_age/inputs/region_age_population_protection.csv
experiments/measles_local_age/inputs/synthetic_region_age_weekly_cases.csv
experiments/measles_local_age/inputs/region_population_density_ons_2024.csv
experiments/measles_local_age/output/local_age_parameter_summary.csv
experiments/measles_age/inputs/age_contact_matrix_reconnect_figure_normalized.csv
outbreak_risk_model/graph_outputs/odwp_england_region_mobility_adjacency.csv
outbreak_risk_model/graph_outputs/odst_england_region_student_mobility_adjacency.csv
```

(!!!) Importnant notes: 

- London calibration and validation use observed all-age weekly surveillance.
- Regional-age weekly data is extrapolated from observed national data
  and published period totals. 

## Reproducibility and interpretation

- Calibration is deterministic for fixed inputs, configuration and search seed (so examiners can reproduce the results if needed).
- Stochastic forecasts record their simulation count and random seed.
- The strict held-out notebook is the true validation result. The complete
  rolling audit uses all-data fitted vectors and is a conditional visual audit.
- Regional forecasts transfer a London-fitted model and are not independently
  fitted regional forecasts as there is no data to do so.
- Threshold exceedance is an operational model event, not an official UKHSA
  outbreak declaration (chosen based on the best model performance and ratio of cases in specific region).


