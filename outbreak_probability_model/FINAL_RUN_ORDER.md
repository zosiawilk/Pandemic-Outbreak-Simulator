# Final project rerun order (for reproduction of results) 

## Optional preprocessing

0. `../outbreak_risk_model/Measles_Graph_Model.ipynb`

   Run this if the population, protection, mobility, contact, or graph
   source files have changed. It is not a prerequisite when the graph already exists and has updated data. 

## Required final pipeline

1. `London_Calibration_6Week_Rolling_Poisson_Fit.ipynb`

   Fits the final non-seasonal eight-parameter comparator using all 117 rolling
   six-week windows. Final budget: 240 bounded random trial vectors, 80 local
   random refinements. Saved to: 
   `calibration_6week_rolling_poisson/03_poisson_fitted_parameters.csv`.

    L-BFGS-B is disabled by default as it failed to give better results; it was not deleted tho in case of further research. 
3. `London_Seasonal_Poisson_Fit.ipynb`

   Warm-starts from step 1 and fits the ten-parameter seasonal model. Final
   run for dissertation results: 120 bounded random trial vectors, 40 local random refinements. Vector saved to:
   `calibration_6week_rolling_poisson_seasonal/03_seasonal_poisson_fitted_parameters.csv`.

4. `London_Strict_Forecast_Validation.ipynb`

   Performs the out-of-sample validation of the forecast on data not seen by the model, nor on overlapping weeks. It does not load either the 
   all-data vector from previous steps. Each model is fitted once to the first 60 observations, frozen, and tested over ten later non-overlapping six-week blocks. It also reports a pre-        selected, balanced eight-block confusion matrix and classified held-out forecast
   pages with median and p10-p90 bands. This is the primary method of validation here, looking at the scarcity of data. 


5. `Poisson_Outbreak_Probability_Run.ipynb`

   Produces the final current-origin seasonal-Poisson six-week forecast (forecast for "today", last data point) from
   1,000 stochastic paths, using threshold `> 15`, four-week state
   conditioning, the fitted seasonality, default noise scale.

6. `London_Seasonal_Complete_Rolling_Audit.ipynb`

   Runs the independently fitted seasonal and non-seasonal all-data vectors at every eligible historical origin with 100 paths. It saves a
   multipage PDF, individual classified forecast plots, confusion matrices, and the detected/missed/false-alarm timeline. This is a conditional visual
   audit, not the held-out validation result in step 3.

7. `Regional_Outbreak_Probability_Run.ipynb`

   Transfers the final London-fitted model to the other English regions and
   generates 100-path six-week forecasts, population-scaled outbreak
   thresholds, regional p10-p90 plots, probability tables, and historical
   scenario-audit plots.

7. `London_Parameter_Sensitivity.ipynb`

   Runs the full final-model sensitivity analysis. The one-at-a-time analysis
   uses seven grid levels plus the exact baseline where distinct, with 300
   paths per level. 

## Development history (not part of the final pipeline)

- `Outbreak_Probability_Run.ipynb` — different loss notebook.
- `London_Calibration_6Week_Fit.ipynb` — old non-rolling calibration.
- `London_Calibration_6Week_Rolling_Fit.ipynb` — old composite-loss fit.
- `London_Simple_Mathsy_Fit.ipynb` — teaching/simple fit based on my Mathsy Maths
- `London_Seasonality_Sensitivity.ipynb` — scenario grid superseded by fitted
  amplitude and phase.
- `London_Seasonality_Whole_Data_Fit.ipynb` — seasonal experiment with whole data.
- `Future_Wave_Probability.ipynb` and `Mathsy_Stochastic_Outbreak_Forecast.ipynb`
  — older forecast variants.
- `../outbreak_risk_model/General_Regional_Probabilistic_Forecast.ipynb` - a
  separate negative-binomial prototype, not the fitted epidemic model.

## Results-section output checklist

-  Fit: `calibration_6week_rolling_poisson/08_calendar_fit.png` and
  `calibration_6week_rolling_poisson_seasonal/06_calendar_fit_comparison.png`.
- Fit parameters and scores: the fitted-parameter CSVs from steps 1-2
  and `04_seasonal_vs_nonseasonal_calibration.csv`.
- Genuine out-of-sample evaluation: `05_held_out_metrics.csv`,
  `06_classification_metrics.csv`, `07_balanced_confusion_matrices.png`, and
  `08_chronological_probability_audit.png` from step 3. Use
  `09_held_out_classified_forecasts.pdf` for the p10-p90 figures.


