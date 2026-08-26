# Region-age input preparation and parameter fitting

This folder documents and reproduces processed region-age inputs used by the
final outbreak-probability model.

The current UKHSA weekly case CSV in this repo has national England weekly
timing and London weekly timing, but not full weekly cases by region and age.
The file `data/england_measles_cases_by_age_and_region_2023_2026.csv` contains
annual/period totals by UKHSA region and broad age group. Therefore this
prototype creates **synthetic region-age weekly targets**:

```text
synthetic_cases[region, age, week]
  = England_weekly_cases[week]
    * annual_cases[region, age, year]
    / annual_cases[England, all ages, year]
```

That means:

- national weekly timing is real;
- region-age annual burden is real;
- region-age weekly timing is inferred, not directly observed.

## Files

- `prepare_local_age_data.py`
  - Reads national weekly measles cases.
  - Reads annual cases by region and age.
  - Reads 2024 mid-year population by region and single year of age.
  - Reads COVER MMR coverage from the supplied ODS file.
  - Creates model-ready CSV inputs.

- `local_age_model.py`
  - Vectorized local × age stochastic simulator.
  - Compartments are `S[region, age]`, `H[region, age]`, `E[region, age]`,
    `I[region, age]`, `Q[region, age]`, `D[region, age]`.
  - `E` is infected/incubating, `I` is infectious, and `Q` is sick.
    Sick infectiousness is averaged using `quarantine_adherence`,
    `quarantined_sick_infectiousness`, and
    `unquarantined_sick_infectiousness`.

- `fit_local_age_blocks.py`
  - Six-week random-search fitting, same broad style as
    `experiments/fit_measles_blocks.py`.

## Suggested first run

Use a small smoke test first:

```bash
python experiments/measles_local_age/prepare_local_age_data.py

python experiments/measles_local_age/fit_local_age_blocks.py \
  --number-of-blocks 2 \
  --trials-per-block 3 \
  --optimizer-sims 1 \
  --final-sims 3
```

Then increase `--number-of-blocks`, `--trials-per-block`, and simulation counts.

## Important assumptions

1. The annual region-age cases do not identify weekly timing by region/age.
   Weekly region-age targets are synthetic.
2. COVER MMR coverage applies directly only to children up to 5 years. For
   older groups, England MMR1 coverage at age five is used as a national proxy,
   not as a direct measurement of adult immunity. Alternative fixed and
   regional proxies are compared in sensitivity analysis.
3. The age contact matrix is a simple default matrix, not a published POLYMOD
   matrix. Replace it later if you want stronger epidemiological grounding.
4. The model is a first structured extension, not a final calibrated public
   health model.
