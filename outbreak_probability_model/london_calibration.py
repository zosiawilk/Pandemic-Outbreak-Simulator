"""London calibration and sensitivity analysis for the region-age model.

This module deliberately sits beside ``model.py`` rather than changing the
outbreak-probability API.  It calibrates one shared parameter set against
consecutive six-week blocks of confirmed London all-age weekly cases.

The observed data are all-age, so the observed value for a block is compared
with the sum over London's seven age groups.  Hidden initial states are
estimated parsimoniously: each block has one observed conditioning week
(``week 0``), followed by six forecast targets (``weeks 1--6``). ``E(0)`` and
``I(0)`` are distributed over London's age groups and set to fitted multiples
of the week-0 count. The live Monday forecast uses the latest reported week in
the same way, with no hidden time shift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .model import (
    AGE_GROUPS,
    CompartmentState,
    DEFAULT_PARAMETER_SUMMARY,
    ModelParameters,
    RegionAgeInputs,
    _run_one_simulation,
    exceeds_outbreak_threshold,
    load_default_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "mathsy-london-v7-random-search-120-40"
DEFAULT_LONDON_CASES = REPO_ROOT / "experiments" / "measles" / "London" / "observed_weekly_cases.csv"
DEFAULT_LONDON_AGE_CASES = (
    REPO_ROOT / "outbreak_probability_model" / "data" / "london_age_confirmed_cases.csv"
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings for a reproducible calibration run."""

    block_weeks: int = 6
    n_trials: int = 80
    n_refinement_trials: int = 20
    final_simulations: int = 100
    sensitivity_simulations: int = 5
    # Kept in the configuration for backward-readable saved notebooks. The
    # time-aligned pipeline requires zero hidden warm-up weeks.
    warmup_weeks: int = 0
    random_seed: int = 20260727
    # A bounded two-dimensional Powell optimisation is run after the broader
    # search so E(0) and I(0) are explicitly estimated conditional on the
    # remaining best parameters. Set to zero to disable this focused stage.
    initial_state_refinement_maxiter: int = 20
    # When a seasonal parameter set is supplied, refine amplitude and phase
    # explicitly after the broad joint search.  This is conditional on the
    # other currently best quantities, just as the E/I refinement above is.
    seasonal_refinement_maxiter: int = 30


@dataclass(frozen=True)
class LondonAgeForecastResult:
    """Six-week stochastic forecast summaries for every London age group."""

    age_probabilities: pd.DataFrame
    all_age_probability: float
    all_age_mc95_low: float
    all_age_mc95_high: float
    any_age_group_probability: float
    any_age_group_mc95_low: float
    any_age_group_mc95_high: float
    weekly_summary: pd.DataFrame
    trajectories: pd.DataFrame
    all_age_trajectories: pd.DataFrame
    parameters: ModelParameters
    fitted_vector: dict[str, float]
    current_cases: float
    outbreak_threshold: float
    horizon_weeks: int
    event_definition: str
    forecast_start: pd.Timestamp
    uncertainty_settings: dict[str, object]
    age_allocation: pd.DataFrame
    history_conditioning_summary: pd.DataFrame
    episode_probabilities: pd.DataFrame
    episode_simulations: pd.DataFrame


@dataclass(frozen=True)
class RecentLondonOriginsResult:
    """Forecast and outbreak-episode summaries for several recent origins."""

    origin_summary: pd.DataFrame
    episode_probabilities: pd.DataFrame
    forecasts: dict[pd.Timestamp, LondonAgeForecastResult]


@dataclass(frozen=True)
class MathsyPredictiveUncertainty:
    """Scenario scales for uncertainty not identified by the point calibration.

    ``parameter_relative_sd`` perturbs the London-fitted parameter vector once
    per simulation. ``initial_state_log_sd`` perturbs E/I/Q once per path.
    Weekly imported infections are now discrete Poisson events so import-free
    weeks and local-chain fade-out remain possible. ``seed_dispersion`` is
    retained only for backward-readable saved configurations.
    """

    parameter_relative_sd: float = 0.15
    initial_state_log_sd: float = 0.25
    seed_dispersion: float = 2.0
    age_allocation_prior_strength: float = 20.0


@dataclass(frozen=True)
class HistoryConditioningConfig:
    """Settings for estimating the forecast-origin state from recent cases.

    The London-wide calibration remains the long-term model.  At each origin
    this small local optimisation uses exactly ``history_weeks`` trailing
    observations, including the forecast-origin observation, to update E/I
    and a multiplicative recent-contact level. The first observation anchors
    the replay and the remaining observations are fitted transitions.
    Regularisation prevents a short noisy series from replacing the global fit.
    """

    history_weeks: int = 4
    # The short history may make a modest local correction, but must not
    # replace the globally calibrated transmission level. These bounds allow
    # about +/-25% on the log scale and avoid the old 2.5x boundary solutions.
    transmission_multiplier_bounds: tuple[float, float] = (0.8, 1.25)
    initial_exposed_bounds: tuple[float, float] = (0.0, 20.0)
    initial_infectious_bounds: tuple[float, float] = (0.0, 20.0)
    regularization_strength: float = 1.0
    # The most recent observation is the forecast origin and should anchor the
    # state handed to the future simulator. Earlier observations still define
    # the direction of travel, but the origin receives extra fitting weight.
    origin_observation_weight: float = 4.0
    maxiter: int = 80


@dataclass(frozen=True)
class HistoryConditioningResult:
    """Latent state and diagnostics obtained at a forecast origin."""

    state: CompartmentState
    transmission_multiplier: float
    initial_exposed_per_case: float
    initial_infectious_per_case: float
    objective: float
    history_fit: pd.DataFrame
    conditioning_start: pd.Timestamp
    forecast_origin: pd.Timestamp
    optimizer_success: bool
    optimizer_message: str
    origin_exposed_total: float
    origin_infectious_total: float
    origin_sick_total: float


@dataclass(frozen=True)
class ParameterSensitivityResult:
    """Global parameter-sensitivity outputs with stochastic variance separated."""

    design: pd.DataFrame
    simulation_outcomes: pd.DataFrame
    variance_decomposition: pd.DataFrame
    parameter_ranking: pd.DataFrame


@dataclass(frozen=True)
class ParameterSpreadSensitivityResult:
    """One-at-a-time parameter sweeps with stochastic spread at every level."""

    parameter_levels: pd.DataFrame
    simulation_outcomes: pd.DataFrame
    level_summary: pd.DataFrame
    parameter_spread: pd.DataFrame


DEFAULT_LONDON_FITTED_PARAMETERS = (
    REPO_ROOT
    / "experiments"
    / "measles"
    / "London"
    / "calibration_6week_rolling"
    / "04_best_fitted_parameters.csv"
)
DEFAULT_LONDON_POISSON_FITTED_PARAMETERS = (
    REPO_ROOT
    / "experiments"
    / "measles"
    / "London"
    / "calibration_6week_rolling_poisson"
    / "03_poisson_fitted_parameters.csv"
)
DEFAULT_LONDON_SEASONAL_POISSON_FITTED_PARAMETERS = (
    REPO_ROOT
    / "experiments"
    / "measles"
    / "London"
    / "calibration_6week_rolling_poisson_seasonal"
    / "03_seasonal_poisson_fitted_parameters.csv"
)
DEFAULT_LONDON_SIMPLE_FITTED_PARAMETERS = (
    REPO_ROOT
    / "experiments"
    / "measles"
    / "London"
    / "calibration_simple"
    / "04_best_fitted_parameters.csv"
)
DEFAULT_LONDON_OPTIMIZATION_TRIALS = DEFAULT_LONDON_FITTED_PARAMETERS.with_name(
    "02_optimization_trials.csv"
)


# Fixed disease values used by the London calibration. Their provenance is
# mixed: some are literature adaptations and some are inherited defaults from
# the earlier SDE simulator. See ``PARAMETER_PROVENANCE`` below; none are
# estimated from the London weekly series in the main fit.
FIXED_DISEASE_PARAMETERS = {
    "beta_0": 0.0000466,  # w: recruitment/birth rate per day
    "sigma": 0.090,      # transmission rate for susceptible people
    "eta": 0.000027,     # a: natural death rate per day
    "nu": 0.003,         # disease death rate per day
    "delta": 0.0009,     # breakthrough transmission rate
    "phi": 0.12,         # recovery rate per day
    "mu": 10.0,           # daily interaction multiplier
}

# This is a current-model assumption, not the old model's regional-risk
# multiplier and not the old quarantine-rate symbol rho. A three-day
# infectious-before-sick period gives psi = 1/3 per day.
FIXED_MODEL_ASSUMPTIONS = {
    "psi": 1.0 / 3.0,
    # The observations are laboratory-confirmed cases and there is no
    # independent London denominator for infections missed by surveillance.
    # Use a reported-case-scale model rather than inventing an ascertainment
    # fraction that is confounded with transmission, seeding and initial E/I.
    "reporting_rate": 1.0,
    # The primary model applies the common diffusion scale directly to S and H.
    # This implementation switch remains explicit only so the completed
    # forecast-validation sensitivity analysis can be reproduced.
    "sh_noise_multiplier": 1.0,
}

FIXED_CALIBRATION_PARAMETERS = {
    **FIXED_DISEASE_PARAMETERS,
    **FIXED_MODEL_ASSUMPTIONS,
}

# Explicit provenance prevents a fixed value from being presented as a newly
# estimated London parameter. ``legacy/default`` means it needs a separate
# source review before being described as disease-specific literature.
PARAMETER_PROVENANCE = {
    "gamma": ("fitted", "London weekly cases", "E -> I progression rate"),
    "local_mixing": ("fitted", "London weekly cases", "local/commuting pressure balance"),
    "contact_scale": ("fitted", "London weekly cases", "global multiplier on the contact matrix"),
    "sick_contact_multiplier": ("fitted", "London weekly cases", "relative contacts of Q"),
    "sick_mobility_multiplier": ("fitted", "London weekly cases", "relative commuting of Q"),
    "seed_infections_per_week": (
        "fitted",
        "London weekly cases",
        "external infections seeded directly into London per week",
    ),
    "reporting_rate": (
        "fixed model assumption",
        "1.0 confirmed-case-scale convention; not an estimate of infection ascertainment",
        "fraction of I -> Q transitions represented in the confirmed-case process",
    ),
    "initial_exposed_per_case": ("fitted", "London weekly cases", "initial E multiplier"),
    "initial_infectious_per_case": ("fitted", "London weekly cases", "initial I multiplier"),
    "seasonal_amplitude": (
        "fitted in seasonal model",
        "London weekly cases",
        "relative amplitude of annual transmission forcing",
    ),
    "seasonal_peak_week": (
        "fitted in seasonal model",
        "London weekly cases",
        "week of maximum annual transmission pressure after 1 January 2024",
    ),
    "beta_0": ("legacy/default", "earlier SDE simulator (w)", "birth/recruitment rate; not separately verified here"),
    "sigma": ("literature adaptation", "measles adaptation: 4.5 x previous COVID baseline", "S transmission coefficient"),
    "eta": ("legacy/default", "earlier SDE simulator (a)", "natural death rate; not separately verified here"),
    "nu": ("legacy/default", "earlier SDE simulator", "Q disease-death rate; not a direct 0.8% conversion"),
    "delta": ("calculated from literature assumption", "sigma x (1 - 0.99)", "protected breakthrough transmission coefficient"),
    "phi": ("literature adaptation", "measles adaptation: 0.12 per day", "Q recovery rate"),
    "mu": ("legacy/default", "earlier SDE simulator", "overall contact multiplier"),
    "psi": ("fixed model assumption", "1 / 3 per day", "three-day I -> Q progression timescale"),
    "noise_scale": ("fixed numerical/stochastic setting", "explicit forecast setting", "independent compartment diffusion scale"),
    "sh_noise_multiplier": (
        "fixed numerical/stochastic setting",
        "unscaled value 1.0; alternatives checked by held-out sensitivity analysis",
        "sensitivity-only multiplier for noise in the S and H compartments",
    ),
    "dt": ("fixed numerical setting", "model default 0.02 days", "Euler-Maruyama time step"),
    "region_risk_multiplier": (
        "fixed model assumption",
        "canonical rho_r=1; optional outcome-derived scenario disabled to prevent leakage",
        "regional force-of-infection multiplier; distinct from psi",
    ),
    "contact_matrix": (
        "empirical constructed input",
        "Reconnect-derived age contact matrix CSV; normalised by its mean",
        "fixed age-contact pattern; only its global contact_scale is fitted",
    ),
}

# These quantities are identifiable, or at least testable, using an all-age
# weekly incidence series. The contact matrix is fitted through one global
# scale because 49 independent age-contact entries cannot be identified from
# one all-age London time series.
FIT_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    # Mean latent period constrained to 4--18 days.  Expressing the range as
    # reciprocal durations makes the epidemiological interpretation explicit.
    "gamma": (1.0 / 18.0, 1.0 / 4.0),
    "local_mixing": (0.0, 1.0),
    # The default contact matrix is normalised to mean one, while ``mu`` is
    # fixed at ten daily interactions. Allow the fitted global scale to move
    # below one rather than forcing an unrealistically explosive R.
    "contact_scale": (0.001, 3.0),
    "sick_contact_multiplier": (0.0, 1.0),
    "sick_mobility_multiplier": (0.0, 1.0),
    # London-specific weekly seeding.  The earlier model-wide 0--100 bound
    # permitted the London-only likelihood to estimate unobserved England-wide
    # imports.  A 0--20 London range is retained for comparison and sensitivity.
    "seed_infections_per_week": (0.0, 20.0),
    "initial_exposed_per_case": (0.0, 20.0),
    "initial_infectious_per_case": (0.0, 20.0),
}

# The seasonal fit is deliberately separate from the established
# non-seasonal calibration.  The earlier sensitivity grid ended at a=0.30
# and week 13 and selected both boundaries, so it could not identify an
# interior optimum.  The final estimation search therefore doubles the
# amplitude range and covers the full annual phase.  The period remains fixed
# at 52.18 weeks rather than adding an eleventh weakly identified parameter.
SEASONAL_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "seasonal_amplitude": (0.0, 0.60),
    "seasonal_peak_week": (0.0, 52.18),
}
SEASONAL_FIT_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    **FIT_PARAMETER_BOUNDS,
    **SEASONAL_PARAMETER_BOUNDS,
}

# Small, explicit candidate grids for the teaching-first calibration.  These
# are not probability distributions and nothing is sampled from them.  The
# notebook prints every value before fitting so the search is fully visible.
# If a selected value lies at an end of its grid, widen that grid and rerun.
SIMPLE_FIT_PARAMETER_GRIDS: dict[str, tuple[float, ...]] = {
    "contact_scale": (0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50),
    "seed_infections_per_week": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    "initial_exposed_per_case": (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    "initial_infectious_per_case": (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
}

SENSITIVITY_PARAMETERS = list(
    dict.fromkeys(
        [
            *FIT_PARAMETER_BOUNDS,
            *SEASONAL_PARAMETER_BOUNDS,
            "seasonal_period_weeks",
            *FIXED_CALIBRATION_PARAMETERS,
            "noise_scale",
            "dt",
        ]
    )
)

# Sensitivity ranges are scientific scenarios, not fitting bounds or confidence
# intervals. Literature/fixed quantities stay fixed during calibration but are
# varied here so their effect on forecasts is visible rather than hidden.
SENSITIVITY_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    **FIT_PARAMETER_BOUNDS,
    "seasonal_amplitude": (0.0, 0.95),
    "seasonal_peak_week": (0.0, 52.18),
    "seasonal_period_weeks": (50.0, 54.0),
    "beta_0": (0.5 * FIXED_DISEASE_PARAMETERS["beta_0"], 1.5 * FIXED_DISEASE_PARAMETERS["beta_0"]),
    "sigma": (0.5 * FIXED_DISEASE_PARAMETERS["sigma"], 1.5 * FIXED_DISEASE_PARAMETERS["sigma"]),
    "eta": (0.5 * FIXED_DISEASE_PARAMETERS["eta"], 1.5 * FIXED_DISEASE_PARAMETERS["eta"]),
    "nu": (0.5 * FIXED_DISEASE_PARAMETERS["nu"], 1.5 * FIXED_DISEASE_PARAMETERS["nu"]),
    "delta": (0.5 * FIXED_DISEASE_PARAMETERS["delta"], 1.5 * FIXED_DISEASE_PARAMETERS["delta"]),
    "phi": (0.5 * FIXED_DISEASE_PARAMETERS["phi"], 1.5 * FIXED_DISEASE_PARAMETERS["phi"]),
    "mu": (0.5 * FIXED_DISEASE_PARAMETERS["mu"], 1.5 * FIXED_DISEASE_PARAMETERS["mu"]),
    "psi": (0.5 * FIXED_MODEL_ASSUMPTIONS["psi"], 1.5 * FIXED_MODEL_ASSUMPTIONS["psi"]),
    "reporting_rate": (0.8, 1.0),
    "noise_scale": (0.0, 0.10),
    "sh_noise_multiplier": (0.0, 1.0),
    "dt": (0.01, 0.04),
}


def load_london_confirmed_cases(path: Path | str = DEFAULT_LONDON_CASES) -> pd.DataFrame:
    """Load and validate the confirmed all-age London weekly case series."""

    cases = pd.read_csv(path)
    required = {"date", "observed_cases"}
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"London case file is missing columns: {sorted(missing)}")
    cases = cases[["date", "observed_cases"]].copy()
    cases["date"] = pd.to_datetime(cases["date"])
    cases["observed_cases"] = pd.to_numeric(cases["observed_cases"], errors="raise")
    cases = cases.sort_values("date").reset_index(drop=True)
    if cases["date"].duplicated().any():
        raise ValueError("London case file contains duplicate weekly dates.")
    if (cases["observed_cases"] < 0).any():
        raise ValueError("London confirmed cases cannot be negative.")
    gaps = cases["date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=7)).all():
        raise ValueError("London case dates must be consecutive seven-day intervals.")
    return cases


def make_six_week_blocks(
    cases: pd.DataFrame,
    block_weeks: int = 6,
    step_weeks: int | None = None,
) -> pd.DataFrame:
    """Return time-aligned forecast windows with an observed week-0 anchor.

    ``step_weeks`` controls whether windows overlap.  The historical behaviour
    is preserved by default: ``step_weeks`` equals ``block_weeks`` and returns
    non-overlapping blocks.  Set ``step_weeks=1`` for a rolling six-week
    window starting at every forecast week. The observation immediately before
    each window is stored as ``initial_reference_cases`` and is never scored as
    a prediction. ``week_coverage`` records how many windows contain each
    forecast target and is used by the optional overlap-aware objective.
    """

    if block_weeks < 2:
        raise ValueError("block_weeks must be at least 2")
    step_weeks = block_weeks if step_weeks is None else int(step_weeks)
    if step_weeks < 1:
        raise ValueError("step_weeks must be at least 1")
    if len(cases) < block_weeks + 1:
        return cases.iloc[0:0].copy().assign(
            block_id=pd.Series(dtype=int),
            week_in_block=pd.Series(dtype=int),
            source_week_index=pd.Series(dtype=int),
            week_coverage=pd.Series(dtype=int),
            initial_reference_cases=pd.Series(dtype=float),
            conditioning_date=pd.Series(dtype="datetime64[ns]"),
        )

    starts = list(range(1, len(cases) - block_weeks + 1, step_weeks))
    rows = []
    coverage = np.zeros(len(cases), dtype=int)
    for block_id, start in enumerate(starts):
        stop = start + block_weeks
        coverage[start:stop] += 1
        window = cases.iloc[start:stop].copy()
        window["block_id"] = int(block_id)
        window["week_in_block"] = np.arange(1, block_weeks + 1)
        window["source_week_index"] = np.arange(start, stop)
        window["initial_reference_cases"] = float(cases.iloc[start - 1]["observed_cases"])
        window["conditioning_date"] = pd.Timestamp(cases.iloc[start - 1]["date"])
        rows.append(window)

    result = pd.concat(rows, ignore_index=True)
    result["week_coverage"] = result["source_week_index"].map(
        dict(enumerate(coverage))
    ).astype(int)
    return result


def default_calibration_parameters() -> ModelParameters:
    """Use fitted medians as a starting point, with case-fit noise kept small."""

    params = ModelParameters()
    if DEFAULT_PARAMETER_SUMMARY.exists():
        summary = pd.read_csv(DEFAULT_PARAMETER_SUMMARY).set_index("parameter")
        mapping = {
            "mu": "mu",
            "local_mixing": "local_mixing",
            "seed_infections_per_week": "seed_infections_per_week",
            "incubation_rate": "gamma",
            "sick_rate": "psi",
            "sick_contact_multiplier": "sick_contact_multiplier",
            "sick_mobility_multiplier": "sick_mobility_multiplier",
        }
        for source, target in mapping.items():
            if source in summary.index and "median" in summary.columns:
                setattr(params, target, float(summary.loc[source, "median"]))
    # Apply all fixed calibration values last.  In particular, psi is the
    # current model's I-to-Q progression assumption (1/3 per day); it is not
    # inherited from the old simulator's rho quarantine-rate setting.
    for name, value in FIXED_CALIBRATION_PARAMETERS.items():
        setattr(params, name, value)
    return params


def parameter_check_table(
    base_parameters: ModelParameters | None = None,
    inputs: RegionAgeInputs | None = None,
    optimization_trials_path: Path | str | None = DEFAULT_LONDON_OPTIMIZATION_TRIALS,
) -> pd.DataFrame:
    """Return one auditable row for every active model/calibration setting.

    The table is intentionally explicit about provenance. ``fitted`` means
    the London weekly series selected a value other than the supplied starting
    point; ``selected starting value (not identified)`` means the optimiser
    retained an assumed initial value. ``literature adaptation``
    means it is a disease assumption imported from the cited adaptation; and
    ``legacy/default`` means it is inherited from the earlier simulator and
    should not be described as a newly measured measles parameter.  The
    ``meets_requested_provenance`` column is ``False`` for those values so
    they cannot be mistaken for fitted or literature-confirmed quantities.
    """

    params = base_parameters or default_calibration_parameters()
    fit_row: pd.Series | None = None
    trials_path = Path(optimization_trials_path) if optimization_trials_path else None
    if trials_path is not None and trials_path.exists():
        trials = pd.read_csv(trials_path)
        if not trials.empty and "objective" in trials:
            fit_row = trials.loc[pd.to_numeric(trials["objective"], errors="coerce").idxmin()]
            starting_rows = trials.loc[trials.get("stage", pd.Series(dtype=str)).eq("starting_point")]
            starting_row = starting_rows.iloc[0] if not starting_rows.empty else None
        else:
            starting_row = None
    else:
        starting_row = None
    rows: list[dict[str, object]] = []
    accepted_statuses = {
        "fitted",
        "fitted/selected by London calibration",
        "literature adaptation",
        "calculated from literature assumption",
        "fixed model assumption",
        "fixed numerical/stochastic setting",
        "fixed numerical setting",
        "calculated input",
        "constructed input",
        "empirical constructed input",
    }

    for name, (status, source, meaning) in PARAMETER_PROVENANCE.items():
        if status == "fitted" and fit_row is not None:
            unchanged_from_start = (
                starting_row is not None
                and name in fit_row
                and name in starting_row
                and np.isclose(float(fit_row[name]), float(starting_row[name]), rtol=1e-7, atol=1e-10)
            )
            if str(fit_row.get("stage", "")) == "starting_point" or unchanged_from_start:
                status = "selected starting value (not identified)"
                source = "assumed starting value retained by London calibration"
            else:
                status = "fitted/selected by London calibration"
            current_value = fit_row.get(name, getattr(params, name, np.nan))
        elif name in {"contact_scale", "initial_exposed_per_case", "initial_infectious_per_case"}:
            current_value = "calibration candidate; no trial audit loaded"
        elif name == "region_risk_multiplier":
            if inputs is None:
                current_value = "loaded from regional input CSV"
            else:
                values = np.asarray(inputs.region_risk_multiplier, dtype=float)
                london_value = (
                    float(values[inputs.regions.index("London")])
                    if "London" in inputs.regions
                    else float("nan")
                )
                current_value = (
                    f"London={london_value:.6g}; min={float(values.min()):.6g}; "
                    f"max={float(values.max()):.6g}"
                )
        elif name == "contact_matrix":
            if inputs is None:
                current_value = "loaded/constructed model input"
            else:
                matrix = np.asarray(inputs.contact_matrix, dtype=float)
                current_value = f"shape={matrix.shape}; mean={float(matrix.mean()):.6g}"
        else:
            current_value = getattr(params, name, np.nan)
        rows.append(
            {
                "parameter": name,
                "status": status,
                "meets_requested_provenance": status in accepted_statuses,
                "current_value": current_value,
                "source_or_calculation": source,
                "meaning": meaning,
            }
        )

    # Keep the old scalar rho visible only as a historical alias.  It is not
    # an additional active parameter and is never used by this calibration.
    rows.append(
        {
            "parameter": "rho (old simulator alias)",
            "status": "historical alias (not active)",
            "meets_requested_provenance": False,
            "current_value": "not used; do not map to psi",
            "source_or_calculation": "old SDE code used rho for I -> Q quarantine rate",
            "meaning": "different from current regional risk rho_r",
        }
    )
    return pd.DataFrame(rows)


def _target_age_weights(inputs: RegionAgeInputs, region_index: int) -> np.ndarray:
    """Distribute all-age hidden states over ages using unprotected population."""

    available = inputs.population[region_index] * (1.0 - inputs.protected_fraction[region_index])
    if available.sum() <= 0:
        available = inputs.population[region_index]
    return available / max(float(available.sum()), 1.0)


def _initial_state_arrays(
    inputs: RegionAgeInputs,
    first_week_cases: float,
    params: ModelParameters,
    region: str = "London",
    age_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create full England matrices with London E/I/Q initialised for a block."""

    region_index = inputs.regions.index(region)
    if age_weights is None:
        age_weights = _target_age_weights(inputs, region_index)
    else:
        age_weights = np.asarray(age_weights, dtype=float)
        if age_weights.shape != (len(inputs.age_groups),) or (age_weights < 0).any():
            raise ValueError("age_weights must contain one non-negative value per age group")
        age_weights = age_weights / max(float(age_weights.sum()), 1.0)

    # Only London has a dated observation aligned to the conditioning week.
    # Do not inject the latest synthetic values from other regions into older
    # calibration windows or historical validation origins.
    initial_q = np.zeros_like(inputs.latest_weekly_cases, dtype=float)
    initial_q[region_index] = float(max(first_week_cases, 0.0)) * age_weights

    initial_e = np.zeros_like(initial_q)
    initial_i = np.zeros_like(initial_q)
    initial_e[region_index] = max(params.initial_exposed_per_case, 0.0) * max(first_week_cases, 0.0) * age_weights
    initial_i[region_index] = max(params.initial_infectious_per_case, 0.0) * max(first_week_cases, 0.0) * age_weights
    return initial_e, initial_i, initial_q


def _with_vector(
    base: ModelParameters,
    vector: dict[str, float],
    force_deterministic: bool = True,
) -> ModelParameters:
    """Copy scalar model parameters and attach the two fitted state multipliers."""

    params = ModelParameters(
        **{name: getattr(base, name) for name in ModelParameters.__dataclass_fields__}
    )
    for name, value in vector.items():
        if hasattr(params, name):
            setattr(params, name, float(value))
    # These are fit to the expected path by default; independent compartment
    # noise is restored separately in the ensemble/sensitivity step.
    if force_deterministic:
        params.noise_scale = 0.0
    elif "noise_scale" in vector:
        params.noise_scale = float(vector["noise_scale"])
    params.dt = float(vector.get("dt", params.dt))
    params.initial_exposed_per_case = float(vector.get("initial_exposed_per_case", 0.0))
    params.initial_infectious_per_case = float(vector.get("initial_infectious_per_case", 0.0))
    params.contact_scale = float(vector.get("contact_scale", 1.0))
    return params


def load_london_fitted_parameters(
    path: Path | str = DEFAULT_LONDON_FITTED_PARAMETERS,
    base_parameters: ModelParameters | None = None,
) -> tuple[ModelParameters, dict[str, float]]:
    """Load the fitted London vector and attach it to model parameters.

    The rolling calibration CSV contains both fixed and fitted columns.  Only
    names in ``FIT_PARAMETER_BOUNDS`` are copied into the fitted vector; fixed
    values, including ``psi=1/3``, always come from
    ``default_calibration_parameters`` so an old CSV cannot silently restore
    the historical ``psi=0.02`` setting.
    """

    fitted_path = Path(path)
    if not fitted_path.exists():
        raise FileNotFoundError(
            f"London fitted-parameter file not found: {fitted_path}. "
            "Run London_Calibration_6Week_Rolling_Poisson_Fit.ipynb first."
        )
    table = pd.read_csv(fitted_path)
    if table.empty:
        raise ValueError(f"London fitted-parameter file is empty: {fitted_path}")
    row = table.iloc[0]
    saved_version = str(row.get("pipeline_version", ""))
    if saved_version != PIPELINE_VERSION:
        raise ValueError(
            f"London fitted parameters were produced by pipeline version "
            f"{saved_version or 'unknown'}, but this code requires {PIPELINE_VERSION}. "
            "Restart its kernel and rerun "
            "London_Calibration_6Week_Rolling_Poisson_Fit.ipynb before "
            "running the seasonal fit or forecasts."
        )
    missing = [name for name in FIT_PARAMETER_BOUNDS if name not in table.columns]
    if missing:
        raise ValueError(f"London fitted-parameter file is missing columns: {missing}")
    vector = {name: float(row[name]) for name in FIT_PARAMETER_BOUNDS}
    # Seasonal fits are stored in a separate file but use the same loader.
    # Older non-seasonal files remain valid and simply omit these columns.
    for name, (low, high) in SEASONAL_PARAMETER_BOUNDS.items():
        if name not in table.columns:
            continue
        value = float(row[name])
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"London fitted parameter {name!r}={value} is outside "
                f"the seasonal range [{low}, {high}]"
            )
        vector[name] = value
    base = base_parameters or default_calibration_parameters()
    params = _with_vector(base, vector, force_deterministic=False)
    return params, vector


def _simulate_block(
    inputs: RegionAgeInputs,
    params: ModelParameters,
    observed_block: pd.DataFrame,
    random_seed: int,
    region: str = "London",
    warmup_weeks: int = 0,
    return_q: bool = False,
    sample_weekly_counts: bool = False,
    initial_reference_cases: float | None = None,
    age_weights: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Forecast one block immediately after its observed week-0 anchor.

    ``initial_reference_cases`` is the known count at week 0. Blocks created by
    :func:`make_six_week_blocks` store it explicitly. Hidden warm-up weeks are
    rejected because they would shift the simulated dates away from the target
    dates. If ``return_q`` is true, active Q is returned with incidence.
    """

    if int(warmup_weeks) != 0:
        raise ValueError(
            "warmup_weeks must be 0: week 0 is the observed conditioning week "
            "and the block contains the immediately following forecast weeks"
        )
    if initial_reference_cases is None:
        if "initial_reference_cases" not in observed_block:
            raise ValueError(
                "A block must contain initial_reference_cases or receive an explicit "
                "initial_reference_cases argument"
            )
        references = observed_block["initial_reference_cases"].drop_duplicates()
        if len(references) != 1:
            raise ValueError("Each block must have exactly one week-0 reference value")
        first_cases = float(references.iloc[0])
    else:
        first_cases = max(float(initial_reference_cases), 0.0)
    initial_e, initial_i, initial_q = _initial_state_arrays(
        inputs,
        first_cases,
        params,
        region,
        age_weights=age_weights,
    )
    scaled_inputs = replace(
        inputs,
        contact_matrix=inputs.contact_matrix * float(getattr(params, "contact_scale", 1.0)),
    )
    weekly = _run_one_simulation(
        inputs=scaled_inputs,
        params=params,
        initial_reported_sick=initial_q,
        initial_exposed=initial_e,
        initial_infectious=initial_i,
        horizon_weeks=len(observed_block),
        random_seed=random_seed,
        return_weekly_q=return_q,
        sample_weekly_counts=sample_weekly_counts,
        seed_region=region,
        simulation_start_date=pd.to_datetime(observed_block["date"]).min(),
    )
    region_index = inputs.regions.index(region)
    if return_q:
        weekly_reported, weekly_q = weekly
        reported_series = weekly_reported[:, region_index, :].sum(axis=1)
        q_series = weekly_q[:, region_index, :].sum(axis=1)
        return reported_series, q_series
    reported_series = weekly[:, region_index, :].sum(axis=1)
    return reported_series


def condition_on_recent_history(
    cases: pd.DataFrame,
    inputs: RegionAgeInputs,
    params: ModelParameters,
    config: HistoryConditioningConfig | None = None,
    *,
    region: str = "London",
    age_weights: np.ndarray | None = None,
    random_seed: int = 0,
) -> HistoryConditioningResult:
    """Estimate the forecast-origin latent state from trailing weekly cases.

    Only observations at or before the forecast origin are used.  The first
    selected observation anchors Q at the beginning of the replay. The
    remaining ``history_weeks - 1`` observations are predictions to be
    matched. Three local
    quantities are estimated: E and I per anchor case and a multiplier on the
    globally fitted contact scale.  All clinical and London-wide parameters
    remain fixed.

    The objective is weighted log-RMSE plus a quadratic penalty toward the
    global fit. The final replay observation receives extra weight because it
    anchors the forecast origin; earlier observations identify the recent
    direction of travel. Log errors stop high-incidence weeks from completely
    dominating, and the penalty is essential because a short history cannot
    identify three local quantities without shrinkage.
    """

    cfg = config or HistoryConditioningConfig()
    required = {"date", "observed_cases"}
    if not required.issubset(cases.columns):
        raise ValueError("cases must contain date and observed_cases columns")
    ordered = cases[["date", "observed_cases"]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered["observed_cases"] = pd.to_numeric(ordered["observed_cases"], errors="raise")
    ordered = ordered.sort_values("date").reset_index(drop=True)
    if ordered["date"].duplicated().any() or (ordered["observed_cases"] < 0).any():
        raise ValueError("conditioning cases require unique dates and non-negative counts")
    if not ordered["date"].diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("conditioning case dates must be consecutive weekly dates")
    history_observations = int(cfg.history_weeks)
    if history_observations < 2:
        raise ValueError("history_weeks must be at least 2")
    if len(ordered) < history_observations:
        raise ValueError(
            f"history conditioning needs at least {history_observations} observations "
            f"(one anchor plus {history_observations - 1} fitted transitions)"
        )
    if region not in inputs.regions:
        raise ValueError(f"inputs do not contain region {region!r}")

    selected = ordered.iloc[-history_observations:].reset_index(drop=True)
    anchor_cases = float(selected.iloc[0]["observed_cases"])
    observed = selected.iloc[1:]["observed_cases"].to_numpy(dtype=float)
    replay_weeks = history_observations - 1
    region_index = inputs.regions.index(region)
    base_e = max(float(params.initial_exposed_per_case), 0.0)
    base_i = max(float(params.initial_infectious_per_case), 0.0)
    transmission_bounds = tuple(map(float, cfg.transmission_multiplier_bounds))
    e_bounds = tuple(map(float, cfg.initial_exposed_bounds))
    i_bounds = tuple(map(float, cfg.initial_infectious_bounds))
    bounds = [transmission_bounds, e_bounds, i_bounds]
    if any(low < 0 or high <= low for low, high in bounds):
        raise ValueError("history-conditioning bounds must be non-negative and increasing")
    if not np.isfinite(cfg.origin_observation_weight) or cfg.origin_observation_weight <= 0:
        raise ValueError("origin_observation_weight must be positive and finite")
    fit_weights = np.ones(replay_weeks, dtype=float)
    fit_weights[-1] = float(cfg.origin_observation_weight)

    deterministic = _with_vector(
        params,
        {
            "contact_scale": float(getattr(params, "contact_scale", 1.0)),
            "initial_exposed_per_case": base_e,
            "initial_infectious_per_case": base_i,
        },
        force_deterministic=True,
    )

    def replay(values: np.ndarray, *, return_state: bool = False):
        multiplier, initial_e_per_case, initial_i_per_case = map(float, values)
        local_params = _with_vector(
            deterministic,
            {
                "initial_exposed_per_case": initial_e_per_case,
                "initial_infectious_per_case": initial_i_per_case,
            },
            force_deterministic=True,
        )
        initial_e, initial_i, initial_q = _initial_state_arrays(
            inputs, anchor_cases, local_params, region=region, age_weights=age_weights
        )
        replay_inputs = replace(
            inputs,
            contact_matrix=(
                inputs.contact_matrix
                * float(getattr(params, "contact_scale", 1.0))
                * multiplier
            ),
        )
        output = _run_one_simulation(
            inputs=replay_inputs,
            params=local_params,
            initial_reported_sick=initial_q,
            initial_exposed=initial_e,
            initial_infectious=initial_i,
            horizon_weeks=replay_weeks,
            random_seed=random_seed,
            sample_weekly_counts=False,
            seed_region=region,
            return_final_state=return_state,
            simulation_start_date=pd.Timestamp(selected.iloc[1]["date"]),
        )
        if return_state:
            weekly, state = output
        else:
            weekly, state = output, None
        prediction = np.asarray(weekly[:, region_index, :], dtype=float).sum(axis=1)
        return prediction, state

    def objective(values: np.ndarray) -> float:
        prediction, _ = replay(values)
        log_rmse = float(
            np.sqrt(
                np.average(
                    (np.log1p(prediction) - np.log1p(observed)) ** 2,
                    weights=fit_weights,
                )
            )
        )
        multiplier, initial_e_per_case, initial_i_per_case = map(float, values)
        # Scale the E/I deviations so their much wider numerical ranges do not
        # overpower the transmission penalty merely because of their units.
        e_scale = max(e_bounds[1] - e_bounds[0], 1.0)
        i_scale = max(i_bounds[1] - i_bounds[0], 1.0)
        penalty = float(cfg.regularization_strength) * (
            np.log(max(multiplier, 1e-12)) ** 2
            + ((initial_e_per_case - base_e) / e_scale) ** 2
            + ((initial_i_per_case - base_i) / i_scale) ** 2
        )
        return log_rmse + penalty

    start = np.asarray(
        [
            1.0,
            np.clip(base_e, *e_bounds),
            np.clip(base_i, *i_bounds),
        ],
        dtype=float,
    )
    fit = minimize(
        objective,
        x0=start,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(cfg.maxiter), "xtol": 1e-4, "ftol": 1e-5},
    )
    selected_values = np.asarray(fit.x, dtype=float)
    prediction, final_state = replay(selected_values, return_state=True)
    assert final_state is not None
    history_fit = pd.DataFrame(
        {
            "date": selected.iloc[1:]["date"].to_numpy(),
            "observed_cases": observed,
            "conditioned_expected_cases": prediction,
            "residual": prediction - observed,
            "fit_weight": fit_weights,
        }
    )
    return HistoryConditioningResult(
        state=final_state,
        transmission_multiplier=float(selected_values[0]),
        initial_exposed_per_case=float(selected_values[1]),
        initial_infectious_per_case=float(selected_values[2]),
        objective=float(objective(selected_values)),
        history_fit=history_fit,
        conditioning_start=pd.Timestamp(selected.iloc[0]["date"]),
        forecast_origin=pd.Timestamp(selected.iloc[-1]["date"]),
        optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message),
        origin_exposed_total=float(final_state.E[region_index].sum()),
        origin_infectious_total=float(final_state.I[region_index].sum()),
        origin_sick_total=float(final_state.Q[region_index].sum()),
    )


def forecast_london_age_groups(
    cases: pd.DataFrame | None = None,
    inputs: RegionAgeInputs | None = None,
    fitted_parameters_path: Path | str = DEFAULT_LONDON_FITTED_PARAMETERS,
    outbreak_threshold: float = 10.0,
    horizon_weeks: int = 6,
    n_simulations: int = 1000,
    random_seed: int = 42,
    warmup_weeks: int = 0,
    current_cases: float | None = None,
    sample_weekly_counts: bool = True,
    noise_scale_override: float | None = None,
    predictive_uncertainty: MathsyPredictiveUncertainty | None = None,
    history_conditioning: HistoryConditioningConfig | None = None,
    history_conditioning_result: HistoryConditioningResult | None = None,
    age_case_history_path: Path | str | None = DEFAULT_LONDON_AGE_CASES,
    fitted_vector_override: dict[str, float] | None = None,
) -> LondonAgeForecastResult:
    """Forecast six weeks for every London age group using the fitted vector.

    Without history conditioning, the latest all-age count initializes Q/E/I
    using the calibration multipliers. With ``history_conditioning`` the recent
    observations are replayed first; with ``history_conditioning_result`` a
    previously inferred state is reused, which is useful for fair sensitivity
    experiments. The fitted contact scale is applied to the age matrix.
    Probabilities are reported for each age group, all-age London totals, and
    the union of age-group events.
    """

    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be at least 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")
    if outbreak_threshold < 0:
        raise ValueError("outbreak_threshold must be non-negative")
    if int(warmup_weeks) != 0:
        raise ValueError(
            "warmup_weeks must be 0 for the live forecast: the latest Monday is "
            "week 0 and forecast week 1 is the immediately following week"
        )
    if history_conditioning is not None and history_conditioning_result is not None:
        raise ValueError(
            "Supply history_conditioning or history_conditioning_result, not both"
        )

    observed = load_london_confirmed_cases() if cases is None else load_london_confirmed_cases_from_frame(cases)
    model_inputs = inputs or load_default_inputs()
    params, fitted_vector = load_london_fitted_parameters(fitted_parameters_path)
    if fitted_vector_override:
        unknown = set(fitted_vector_override).difference(SENSITIVITY_PARAMETER_BOUNDS)
        if unknown:
            raise ValueError(f"Unknown model-parameter override parameters: {sorted(unknown)}")
        for name, value in fitted_vector_override.items():
            low, high = SENSITIVITY_PARAMETER_BOUNDS[name]
            if not low <= float(value) <= high:
                raise ValueError(f"Override for {name} must lie in [{low}, {high}]")
        fitted_vector.update(
            {
                name: float(value)
                for name, value in fitted_vector_override.items()
                if name in FIT_PARAMETER_BOUNDS
            }
        )
        params = _with_vector(
            params,
            {**fitted_vector, **{name: float(value) for name, value in fitted_vector_override.items()}},
            force_deterministic=False,
        )
    if noise_scale_override is not None:
        if float(noise_scale_override) < 0:
            raise ValueError("noise_scale_override must be non-negative")
        # Noise is not fitted by the current London calibration.  This is an
        # explicit stochastic sensitivity override, not a fitted parameter.
        params.noise_scale = float(noise_scale_override)
    if "London" not in model_inputs.regions:
        raise ValueError("The loaded inputs do not contain London.")

    latest_cases = (
        float(observed.iloc[-1]["observed_cases"])
        if current_cases is None
        else max(float(current_cases), 0.0)
    )
    event_definition = (
        f"at least one of forecast weeks 1-{int(horizon_weeks)} has more than "
        f"{float(outbreak_threshold):g} reported cases"
    )
    age_weights, age_alpha, age_allocation = london_age_case_weights(
        model_inputs,
        forecast_origin=pd.Timestamp(observed.iloc[-1]["date"]),
        path=age_case_history_path,
        prior_strength=(
            0.0
            if predictive_uncertainty is None
            else predictive_uncertainty.age_allocation_prior_strength
        ),
    )
    initial_e, initial_i, initial_q = _initial_state_arrays(
        model_inputs, latest_cases, params, region="London", age_weights=age_weights
    )
    conditioned = history_conditioning_result
    conditioned_state = None
    recent_transmission_multiplier = 1.0
    if history_conditioning is not None:
        conditioning_cases = observed.copy()
        # ``current_cases`` represents a newly available origin observation.
        # Replace only that final value; no future observation is introduced.
        conditioning_cases.loc[conditioning_cases.index[-1], "observed_cases"] = latest_cases
        conditioned = condition_on_recent_history(
            conditioning_cases,
            model_inputs,
            params,
            history_conditioning,
            region="London",
            age_weights=age_weights,
            random_seed=random_seed,
        )
    if conditioned is not None:
        conditioned_state = conditioned.state
        recent_transmission_multiplier = conditioned.transmission_multiplier
        initial_e = conditioned_state.E
        initial_i = conditioned_state.I
        initial_q = conditioned_state.Q
    contact_scale = float(getattr(params, "contact_scale", 1.0))
    forecast_inputs = replace(
        model_inputs,
        contact_matrix=(
            model_inputs.contact_matrix * contact_scale * recent_transmission_multiplier
        ),
    )
    london_index = model_inputs.regions.index("London")
    forecast_start = pd.Timestamp(observed.iloc[-1]["date"]) + pd.Timedelta(days=7)
    forecast_dates = pd.date_range(forecast_start, periods=horizon_weeks, freq="7D")

    rng = np.random.default_rng(random_seed)
    paths: list[np.ndarray] = []
    episode_rows: list[dict[str, object]] = []
    for simulation_number in range(1, int(n_simulations) + 1):
        simulation_params = params
        simulation_inputs = forecast_inputs
        simulation_e = initial_e
        simulation_i = initial_i
        simulation_q = initial_q
        simulation_state = conditioned_state
        weekly_seed_schedule = None
        if predictive_uncertainty is not None:
            simulation_params, simulation_vector = _sample_forecast_parameters(
                params, fitted_vector, predictive_uncertainty, rng
            )
            state_multiplier = rng.lognormal(
                mean=-0.5 * predictive_uncertainty.initial_state_log_sd**2,
                sigma=predictive_uncertainty.initial_state_log_sd,
                size=3,
            )
            if conditioned_state is None:
                simulation_age_weights = rng.dirichlet(age_alpha)
                simulation_e, simulation_i, simulation_q = _initial_state_arrays(
                    model_inputs,
                    latest_cases,
                    simulation_params,
                    region="London",
                    age_weights=simulation_age_weights,
                )
                simulation_e = simulation_e * state_multiplier[0]
                simulation_i = simulation_i * state_multiplier[1]
                simulation_q = simulation_q * state_multiplier[2]
            else:
                # Preserve the dynamically consistent S/H history while
                # perturbing the uncertain latent E/I/Q stocks per path.
                perturbed_e = conditioned_state.E * state_multiplier[0]
                perturbed_i = conditioned_state.I * state_multiplier[1]
                perturbed_q = conditioned_state.Q * state_multiplier[2]
                available_unprotected = (
                    conditioned_state.S
                    + conditioned_state.E
                    + conditioned_state.I
                    + conditioned_state.Q
                )
                latent_total = perturbed_e + perturbed_i + perturbed_q
                latent_scale = np.minimum(
                    1.0,
                    np.divide(
                        available_unprotected,
                        np.maximum(latent_total, 1e-12),
                        out=np.ones_like(latent_total),
                        where=latent_total > 0,
                    ),
                )
                perturbed_e *= latent_scale
                perturbed_i *= latent_scale
                perturbed_q *= latent_scale
                simulation_state = CompartmentState(
                    S=np.maximum(
                        available_unprotected - perturbed_e - perturbed_i - perturbed_q,
                        0.0,
                    ),
                    H=conditioned_state.H,
                    E=perturbed_e,
                    I=perturbed_i,
                    Q=perturbed_q,
                    D=conditioned_state.D,
                )
            simulation_inputs = replace(
                model_inputs,
                contact_matrix=(
                    model_inputs.contact_matrix
                    * float(getattr(simulation_params, "contact_scale", contact_scale))
                    * recent_transmission_multiplier
                ),
            )
        # Treat future imported infections as discrete events. Poisson seeding
        # permits import-free weeks and therefore does not mechanically keep
        # every simulated chain alive.
        seed_mean = max(float(simulation_params.seed_infections_per_week), 0.0)
        weekly_seed_schedule = rng.poisson(
            seed_mean, size=int(horizon_weeks)
        ).astype(float)
        simulation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        simulated = _run_one_simulation(
            inputs=simulation_inputs,
            params=simulation_params,
            initial_reported_sick=simulation_q,
            initial_exposed=simulation_e,
            initial_infectious=simulation_i,
            horizon_weeks=int(horizon_weeks),
            random_seed=simulation_seed,
            sample_weekly_counts=sample_weekly_counts,
            weekly_seed_infections=weekly_seed_schedule,
            seed_region="London",
            initial_state=simulation_state,
            simulation_start_date=forecast_start,
        )
        london_simulated = np.asarray(simulated[:, london_index, :], dtype=float)
        paths.append(london_simulated)

        # Replay the chain present at the forecast origin with all future
        # imports disabled. This counterfactual separates fade-out of that
        # chain from a later, unrelated reintroduction.
        local_weekly, local_latent = _run_one_simulation(
            inputs=simulation_inputs,
            params=simulation_params,
            initial_reported_sick=simulation_q,
            initial_exposed=simulation_e,
            initial_infectious=simulation_i,
            horizon_weeks=int(horizon_weeks),
            random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            sample_weekly_counts=sample_weekly_counts,
            weekly_seed_infections=np.zeros(int(horizon_weeks), dtype=float),
            seed_region="London",
            initial_state=simulation_state,
            return_weekly_latent=True,
            simulation_start_date=forecast_start,
        )
        local_linked_cases = np.asarray(
            local_weekly[:, london_index, :], dtype=float
        ).sum(axis=1)
        local_active = sum(
            np.asarray(local_latent[name][:, london_index, :], dtype=float).sum(axis=1)
            for name in ("E", "I", "Q")
        )
        extinct_candidates = np.flatnonzero(local_active < 0.5)
        extinction_week = (
            float(extinct_candidates[0] + 1) if extinct_candidates.size else np.nan
        )
        london_imports = rng.poisson(seed_mean, size=int(horizon_weeks))
        new_import_after_extinction = bool(
            np.isfinite(extinction_week)
            and np.any(london_imports[int(extinction_week) :] > 0)
        )
        linked_case_free_46_days = np.nan
        if int(horizon_weeks) >= 7:
            linked_case_free_46_days = bool(
                any(
                    np.all(local_linked_cases[start : start + 7] == 0)
                    for start in range(int(horizon_weeks) - 7 + 1)
                )
            )
        all_age_values = london_simulated.sum(axis=1)
        week_six_index = min(6, int(horizon_weeks)) - 1
        episode_rows.append(
            {
                "simulation": simulation_number,
                "any_week_above_threshold": bool(
                    np.any(all_age_values[: min(6, int(horizon_weeks))] > outbreak_threshold)
                ),
                "at_or_below_threshold_by_week_6": bool(
                    all_age_values[week_six_index] <= outbreak_threshold
                ),
                "local_extinction_week": extinction_week,
                "local_chain_extinct_by_horizon": bool(np.isfinite(extinction_week)),
                "linked_case_free_46_days_by_horizon": linked_case_free_46_days,
                "new_import_after_extinction": new_import_after_extinction,
                "week_6_cases": float(all_age_values[week_six_index]),
                "peak_first_6_weeks": float(
                    all_age_values[: min(6, int(horizon_weeks))].max()
                ),
            }
        )

    age_paths = np.asarray(paths, dtype=float)  # simulation, week, age
    all_age_paths = age_paths.sum(axis=2)
    age_peak = age_paths.max(axis=1)
    all_age_peak = all_age_paths.max(axis=1)
    age_event_flags = exceeds_outbreak_threshold(age_peak, outbreak_threshold)
    all_age_event_flags = exceeds_outbreak_threshold(all_age_peak, outbreak_threshold)
    any_age_event_flags = age_event_flags.any(axis=1)
    all_age_probability = float(all_age_event_flags.mean())
    any_age_group_probability = float(any_age_event_flags.mean())
    all_age_mc95_low, all_age_mc95_high = _wilson_interval(all_age_event_flags)
    any_age_mc95_low, any_age_mc95_high = _wilson_interval(any_age_event_flags)

    initial_q_london = initial_q[london_index]
    initial_e_london = initial_e[london_index]
    initial_i_london = initial_i[london_index]
    age_summary_rows = []
    weekly_summary_rows = []
    trajectory_rows = []
    for age_index, age_group in enumerate(model_inputs.age_groups):
        age_values = age_paths[:, :, age_index]
        age_cumulative = np.cumsum(age_values, axis=1)
        age_flags = exceeds_outbreak_threshold(age_peak[:, age_index], outbreak_threshold)
        age_mc95_low, age_mc95_high = _wilson_interval(age_flags)
        age_summary_rows.append(
            {
                "age_group": age_group,
                "outbreak_probability": float(age_flags.mean()),
                "mc_95_low": age_mc95_low,
                "mc_95_high": age_mc95_high,
                "outbreak_threshold": float(outbreak_threshold),
                "event_definition": event_definition,
                "expected_peak_weekly_cases": float(age_values.max(axis=1).mean()),
                "expected_total_cases": float(age_values.sum(axis=1).mean()),
                "initial_q": float(initial_q_london[age_index]),
                "initial_e": float(initial_e_london[age_index]),
                "initial_i": float(initial_i_london[age_index]),
            }
        )
        for week_index, date in enumerate(forecast_dates):
            weekly_summary_rows.append(
                {
                    "age_group": age_group,
                    "week": week_index + 1,
                    "date": date,
                    "mean_cases": float(age_values[:, week_index].mean()),
                    "median_cases": float(np.percentile(age_values[:, week_index], 50)),
                    "p10_cases": float(np.percentile(age_values[:, week_index], 10)),
                    "p90_cases": float(np.percentile(age_values[:, week_index], 90)),
                }
            )
        for simulation_index, (values, cumulative) in enumerate(
            zip(age_values, age_cumulative), start=1
        ):
            peak = float(values.max())
            crosses = bool(exceeds_outbreak_threshold(peak, outbreak_threshold))
            for week_index, (date, value, cumulative_value) in enumerate(
                zip(forecast_dates, values, cumulative), start=1
            ):
                trajectory_rows.append(
                    {
                        "simulation": simulation_index,
                        "week": week_index,
                        "date": date,
                        "age_group": age_group,
                        "weekly_cases": float(value),
                        "cumulative_cases": float(cumulative_value),
                        "peak_weekly_cases": peak,
                        "crosses_outbreak_threshold": crosses,
                    }
                )

    all_age_trajectory_rows = []
    all_age_cumulative = np.cumsum(all_age_paths, axis=1)
    for simulation_index, (values, cumulative) in enumerate(
        zip(all_age_paths, all_age_cumulative), start=1
    ):
        peak = float(values.max())
        crosses = bool(exceeds_outbreak_threshold(peak, outbreak_threshold))
        for week_index, (date, value, cumulative_value) in enumerate(
            zip(forecast_dates, values, cumulative), start=1
        ):
            all_age_trajectory_rows.append(
                {
                    "simulation": simulation_index,
                    "week": week_index,
                    "date": date,
                    "weekly_cases": float(value),
                    "cumulative_cases": float(cumulative_value),
                    "peak_weekly_cases": peak,
                    "crosses_outbreak_threshold": crosses,
                }
            )

    episode_simulations = pd.DataFrame(episode_rows)
    episode_summary_rows: list[dict[str, object]] = []
    for metric, label in (
        ("any_week_above_threshold", "any of forecast weeks 1-6 above threshold"),
        ("at_or_below_threshold_by_week_6", "at or below threshold in forecast week 6"),
        ("local_chain_extinct_by_horizon", "origin local chain extinct by horizon"),
        (
            "linked_case_free_46_days_by_horizon",
            "seven consecutive linked-case-free weeks by horizon (weekly proxy for 46 days)",
        ),
        ("new_import_after_extinction", "new importation after local extinction by horizon"),
    ):
        flags = episode_simulations[metric].dropna().astype(bool).to_numpy()
        probability = float(flags.mean()) if flags.size else np.nan
        low, high = _wilson_interval(flags)
        episode_summary_rows.append(
            {
                "metric": metric,
                "definition": label,
                "week": np.nan,
                "probability": probability,
                "mc_95_low": low,
                "mc_95_high": high,
                "n_simulations": int(flags.size),
            }
        )
    for week in range(1, int(horizon_weeks) + 1):
        flags = (
            episode_simulations["local_extinction_week"].notna()
            & (episode_simulations["local_extinction_week"] <= week)
        ).to_numpy()
        low, high = _wilson_interval(flags)
        episode_summary_rows.append(
            {
                "metric": "local_chain_extinct_by_week",
                "definition": f"origin local chain extinct by forecast week {week}",
                "week": week,
                "probability": float(flags.mean()),
                "mc_95_low": low,
                "mc_95_high": high,
                "n_simulations": int(flags.size),
            }
        )

    return LondonAgeForecastResult(
        age_probabilities=pd.DataFrame(age_summary_rows),
        all_age_probability=all_age_probability,
        all_age_mc95_low=all_age_mc95_low,
        all_age_mc95_high=all_age_mc95_high,
        any_age_group_probability=any_age_group_probability,
        any_age_group_mc95_low=any_age_mc95_low,
        any_age_group_mc95_high=any_age_mc95_high,
        weekly_summary=pd.DataFrame(weekly_summary_rows),
        trajectories=pd.DataFrame(trajectory_rows),
        all_age_trajectories=pd.DataFrame(all_age_trajectory_rows),
        parameters=params,
        fitted_vector=fitted_vector,
        current_cases=latest_cases,
        outbreak_threshold=float(outbreak_threshold),
        horizon_weeks=int(horizon_weeks),
        event_definition=event_definition,
        forecast_start=forecast_start,
        uncertainty_settings=(
            {
                "enabled": False,
                "history_conditioning_enabled": conditioned is not None,
                "history_weeks": (
                    None if conditioned is None else len(conditioned.history_fit) + 1
                ),
                "recent_transmission_multiplier": recent_transmission_multiplier,
                "history_conditioning_objective": (
                    None if conditioned is None else conditioned.objective
                ),
                "history_conditioning_optimizer_success": (
                    None if conditioned is None else conditioned.optimizer_success
                ),
                "history_conditioning_start": (
                    None if conditioned is None else conditioned.conditioning_start
                ),
                "history_conditioning_origin": (
                    None if conditioned is None else conditioned.forecast_origin
                ),
                "anchor_exposed_per_case": (
                    None if conditioned is None else conditioned.initial_exposed_per_case
                ),
                "anchor_infectious_per_case": (
                    None if conditioned is None else conditioned.initial_infectious_per_case
                ),
                "origin_exposed_total": (
                    None if conditioned is None else conditioned.origin_exposed_total
                ),
                "origin_infectious_total": (
                    None if conditioned is None else conditioned.origin_infectious_total
                ),
                "origin_sick_total": (
                    None if conditioned is None else conditioned.origin_sick_total
                ),
            }
            if predictive_uncertainty is None
            else {
                "enabled": True,
                "parameter_relative_sd": predictive_uncertainty.parameter_relative_sd,
                "initial_state_log_sd": predictive_uncertainty.initial_state_log_sd,
                "seed_dispersion": predictive_uncertainty.seed_dispersion,
                "age_allocation_prior_strength": predictive_uncertainty.age_allocation_prior_strength,
                "history_conditioning_enabled": conditioned is not None,
                "history_weeks": (
                    None if conditioned is None else len(conditioned.history_fit) + 1
                ),
                "recent_transmission_multiplier": recent_transmission_multiplier,
                "history_conditioning_objective": (
                    None if conditioned is None else conditioned.objective
                ),
                "history_conditioning_optimizer_success": (
                    None if conditioned is None else conditioned.optimizer_success
                ),
                "history_conditioning_start": (
                    None if conditioned is None else conditioned.conditioning_start
                ),
                "history_conditioning_origin": (
                    None if conditioned is None else conditioned.forecast_origin
                ),
                "anchor_exposed_per_case": (
                    None if conditioned is None else conditioned.initial_exposed_per_case
                ),
                "anchor_infectious_per_case": (
                    None if conditioned is None else conditioned.initial_infectious_per_case
                ),
                "origin_exposed_total": (
                    None if conditioned is None else conditioned.origin_exposed_total
                ),
                "origin_infectious_total": (
                    None if conditioned is None else conditioned.origin_infectious_total
                ),
                "origin_sick_total": (
                    None if conditioned is None else conditioned.origin_sick_total
                ),
            }
        ),
        age_allocation=age_allocation,
        history_conditioning_summary=(
            pd.DataFrame() if conditioned is None else conditioned.history_fit.copy()
        ),
        episode_probabilities=pd.DataFrame(episode_summary_rows),
        episode_simulations=episode_simulations,
    )


def forecast_recent_london_origins(
    cases: pd.DataFrame | None = None,
    *,
    n_origins: int = 4,
    **forecast_kwargs: object,
) -> RecentLondonOriginsResult:
    """Launch separate forecasts from each of the most recent observed weeks.

    Each origin receives only the case history available on or before that
    date. Consequently four origins use four different current case counts and
    four independently conditioned E/I/Q states; later observations never leak
    into earlier forecasts.
    """

    observed = (
        load_london_confirmed_cases()
        if cases is None
        else load_london_confirmed_cases_from_frame(cases)
    )
    n_origins = int(n_origins)
    if n_origins < 1:
        raise ValueError("n_origins must be at least 1")
    history_cfg = forecast_kwargs.get("history_conditioning")
    minimum_history = 1 if history_cfg is None else int(history_cfg.history_weeks)
    if len(observed) < n_origins + minimum_history - 1:
        raise ValueError("case history is too short for the requested origins and conditioning")

    forecasts: dict[pd.Timestamp, LondonAgeForecastResult] = {}
    probability_tables = []
    origin_rows = []
    first_origin_position = len(observed) - n_origins
    for offset, position in enumerate(range(first_origin_position, len(observed))):
        truncated = observed.iloc[: position + 1].copy()
        origin_date = pd.Timestamp(truncated.iloc[-1]["date"])
        origin_cases = float(truncated.iloc[-1]["observed_cases"])
        result = forecast_london_age_groups(
            cases=truncated,
            random_seed=int(forecast_kwargs.get("random_seed", 42)) + offset * 100_003,
            **{k: v for k, v in forecast_kwargs.items() if k != "random_seed"},
        )
        forecasts[origin_date] = result
        probabilities = result.episode_probabilities.copy()
        probabilities.insert(0, "origin_cases", origin_cases)
        probabilities.insert(0, "origin_date", origin_date)
        probability_tables.append(probabilities)
        origin_rows.append(
            {
                "origin_date": origin_date,
                "origin_cases": origin_cases,
                "forecast_start": result.forecast_start,
                "origin_exposed_total": result.uncertainty_settings.get(
                    "origin_exposed_total"
                ),
                "origin_infectious_total": result.uncertainty_settings.get(
                    "origin_infectious_total"
                ),
                "origin_sick_total": result.uncertainty_settings.get("origin_sick_total"),
                "recent_transmission_multiplier": result.uncertainty_settings.get(
                    "recent_transmission_multiplier"
                ),
            }
        )
    return RecentLondonOriginsResult(
        origin_summary=pd.DataFrame(origin_rows),
        episode_probabilities=pd.concat(probability_tables, ignore_index=True),
        forecasts=forecasts,
    )


def _wilson_interval(flags: np.ndarray, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for Monte Carlo precision of a simulated probability.

    This measures finite-path numerical error only. It is not a confidence
    interval for the real future event probability or for model parameters.
    """

    binary = np.asarray(flags, dtype=float).reshape(-1)
    n = binary.size
    if n == 0:
        return np.nan, np.nan
    p = float(binary.mean())
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def _condition_baseline_for_sensitivity(
    cases: pd.DataFrame | None,
    inputs: RegionAgeInputs,
    params: ModelParameters,
    config: HistoryConditioningConfig | None,
    current_cases: float | None,
    random_seed: int,
) -> HistoryConditioningResult | None:
    """Condition the current state once, then hold it fixed across scenarios.

    This isolates the effect of changing future model parameters. Re-estimating
    the recent state separately at every parameter level would allow the local
    optimiser to compensate for the tested change and would make the OAT
    comparison both much slower and harder to interpret.
    """

    if config is None:
        return None
    observed = (
        load_london_confirmed_cases()
        if cases is None
        else load_london_confirmed_cases_from_frame(cases)
    )
    conditioning_cases = observed.copy()
    if current_cases is not None:
        conditioning_cases.loc[
            conditioning_cases.index[-1], "observed_cases"
        ] = max(float(current_cases), 0.0)
    age_weights, _, _ = london_age_case_weights(
        inputs,
        forecast_origin=pd.Timestamp(conditioning_cases.iloc[-1]["date"]),
        prior_strength=0.0,
    )
    return condition_on_recent_history(
        conditioning_cases,
        inputs,
        params,
        config,
        region="London",
        age_weights=age_weights,
        random_seed=random_seed,
    )


def forecast_parameter_sensitivity(
    cases: pd.DataFrame | None = None,
    inputs: RegionAgeInputs | None = None,
    fitted_parameters_path: Path | str = DEFAULT_LONDON_FITTED_PARAMETERS,
    parameter_ranges: dict[str, tuple[float, float]] | None = None,
    relative_range: float = 0.20,
    n_design: int = 24,
    simulations_per_design: int = 50,
    outbreak_threshold: float = 10.0,
    horizon_weeks: int = 6,
    warmup_weeks: int = 0,
    random_seed: int = 20260804,
    current_cases: float | None = None,
    sample_weekly_counts: bool = True,
    noise_scale: float | None = None,
    history_conditioning: HistoryConditioningConfig | None = None,
) -> ParameterSensitivityResult:
    """Rank parameter effects and separate parameter from stochastic variance.

    A stratified Latin-hypercube-style design changes every fitted parameter,
    fixed literature/model quantity, noise scale and numerical step listed in
    ``SENSITIVITY_PARAMETERS`` simultaneously. At each design point, repeated SDE paths
    estimate the conditional outcome distribution. The law of total variance
    is then used as ``Var(E[Y|theta]) + E[Var(Y|theta)]``.

    The ranking uses squared Spearman associations with design-point outcome
    means. Its variance contribution is an association-based allocation, not a
    formal Sobol index; correlated/non-monotone effects should be followed up
    with targeted experiments.
    """

    if n_design < 3:
        raise ValueError("n_design must be at least 3")
    if simulations_per_design < 2:
        raise ValueError("simulations_per_design must be at least 2")
    if not 0 < relative_range <= 1:
        raise ValueError("relative_range must lie in (0, 1]")

    model_inputs = inputs or load_default_inputs()
    baseline_params, fitted_baseline = load_london_fitted_parameters(fitted_parameters_path)
    conditioned_history = _condition_baseline_for_sensitivity(
        cases,
        model_inputs,
        baseline_params,
        history_conditioning,
        current_cases,
        random_seed,
    )
    baseline = {
        name: (
            float(fitted_baseline[name])
            if name in fitted_baseline
            else float(getattr(baseline_params, name))
        )
        for name in SENSITIVITY_PARAMETERS
    }
    if parameter_ranges is None:
        ranges: dict[str, tuple[float, float]] = {}
        for name, value in baseline.items():
            global_low, global_high = SENSITIVITY_PARAMETER_BOUNDS[name]
            half_width = relative_range * max(abs(value), 0.05 * (global_high - global_low))
            low = max(global_low, value - half_width)
            high = min(global_high, value + half_width)
            if high <= low:
                high = min(global_high, low + max(1e-9, 0.01 * (global_high - global_low)))
            ranges[name] = (float(low), float(high))
    else:
        ranges = {name: (float(bounds[0]), float(bounds[1])) for name, bounds in parameter_ranges.items()}
        unknown = set(ranges).difference(SENSITIVITY_PARAMETER_BOUNDS)
        if unknown:
            raise ValueError(f"Sensitivity ranges contain unknown parameters: {sorted(unknown)}")
        for name, (low, high) in ranges.items():
            bound_low, bound_high = SENSITIVITY_PARAMETER_BOUNDS[name]
            if not bound_low <= low < high <= bound_high:
                raise ValueError(
                    f"Range for {name} must satisfy {bound_low} <= low < high <= {bound_high}"
                )

    names = list(ranges)
    rng = np.random.default_rng(random_seed)
    unit_design = np.empty((n_design, len(names)), dtype=float)
    for column in range(len(names)):
        unit_design[:, column] = (rng.permutation(n_design) + rng.random(n_design)) / n_design
    design = pd.DataFrame(
        {
            name: ranges[name][0] + unit_design[:, index] * (ranges[name][1] - ranges[name][0])
            for index, name in enumerate(names)
        }
    )
    design.insert(0, "design_id", np.arange(n_design, dtype=int))

    outcome_frames: list[pd.DataFrame] = []
    for row in design.itertuples(index=False):
        design_id = int(row.design_id)
        vector = {name: float(getattr(row, name)) for name in names}
        result = forecast_london_age_groups(
            cases=cases,
            inputs=model_inputs,
            fitted_parameters_path=fitted_parameters_path,
            fitted_vector_override=vector,
            outbreak_threshold=outbreak_threshold,
            horizon_weeks=horizon_weeks,
            n_simulations=simulations_per_design,
            # Common random numbers across designs reduce Monte Carlo noise in
            # between-parameter comparisons without changing any path marginal.
            random_seed=int(random_seed),
            warmup_weeks=warmup_weeks,
            current_cases=current_cases,
            sample_weekly_counts=sample_weekly_counts,
            noise_scale_override=noise_scale,
            predictive_uncertainty=None,
            history_conditioning_result=conditioned_history,
        )
        age = result.trajectories.groupby(["age_group", "simulation"], as_index=False).agg(
            peak_weekly_cases=("weekly_cases", "max"),
            cumulative_cases=("weekly_cases", "sum"),
            final_week_cases=("weekly_cases", "last"),
        )
        all_age = result.all_age_trajectories.groupby("simulation", as_index=False).agg(
            peak_weekly_cases=("weekly_cases", "max"),
            cumulative_cases=("weekly_cases", "sum"),
            final_week_cases=("weekly_cases", "last"),
        )
        all_age.insert(0, "age_group", "all ages combined")
        outcomes = pd.concat([age, all_age], ignore_index=True)
        outcomes["outbreak_indicator"] = (
            exceeds_outbreak_threshold(outcomes["peak_weekly_cases"], outbreak_threshold)
        ).astype(float)
        outcomes.insert(0, "design_id", design_id)
        outcome_frames.append(outcomes)
    simulation_outcomes = pd.concat(outcome_frames, ignore_index=True)

    outcome_names = [
        "outbreak_indicator",
        "peak_weekly_cases",
        "cumulative_cases",
        "final_week_cases",
    ]
    variance_rows: list[dict[str, float | str]] = []
    mean_rows: list[pd.DataFrame] = []
    for age_group, group in simulation_outcomes.groupby("age_group", sort=False):
        means = group.groupby("design_id")[outcome_names].mean()
        mean_rows.append(means.assign(age_group=age_group).reset_index())
        for outcome in outcome_names:
            conditional_means = group.groupby("design_id")[outcome].mean()
            conditional_variances = group.groupby("design_id")[outcome].agg(
                lambda values: float(np.var(values.to_numpy(dtype=float), ddof=0))
            )
            raw_between = float(np.var(conditional_means.to_numpy(dtype=float), ddof=0))
            within = float(conditional_variances.mean())
            # Conditional means are themselves Monte Carlo estimates. Remove
            # their average sampling variance so stochastic simulation error is
            # not counted once as parameter variation and again as within-design
            # variation.
            monte_carlo_mean_variance = within / float(simulations_per_design)
            between = max(raw_between - monte_carlo_mean_variance, 0.0)
            total = between + within
            variance_rows.append(
                {
                    "age_group": age_group,
                    "outcome": outcome,
                    "between_parameter_variance": between,
                    "raw_variance_across_design_means": raw_between,
                    "monte_carlo_variance_of_design_mean": monte_carlo_mean_variance,
                    "mean_stochastic_variance": within,
                    "total_predictive_variance": total,
                    "parameter_variance_fraction": between / total if total > 0 else 0.0,
                    "stochastic_variance_fraction": within / total if total > 0 else 0.0,
                }
            )
    variance_decomposition = pd.DataFrame(variance_rows)

    design_means = pd.concat(mean_rows, ignore_index=True)
    ranking_rows: list[dict[str, float | str]] = []
    for age_group, group in design_means.groupby("age_group", sort=False):
        merged = design.merge(group, on="design_id", how="inner")
        for outcome in outcome_names:
            effects: list[tuple[str, float]] = []
            for name in names:
                parameter_rank = merged[name].rank()
                outcome_rank = merged[outcome].rank()
                if parameter_rank.nunique() < 2 or outcome_rank.nunique() < 2:
                    rho = 0.0
                else:
                    rho = float(parameter_rank.corr(outcome_rank))
                    if not np.isfinite(rho):
                        rho = 0.0
                effects.append((name, rho))
            denominator = sum(rho * rho for _, rho in effects)
            between = float(
                variance_decomposition.loc[
                    (variance_decomposition["age_group"] == age_group)
                    & (variance_decomposition["outcome"] == outcome),
                    "between_parameter_variance",
                ].iloc[0]
            )
            ordered = sorted(effects, key=lambda item: abs(item[1]), reverse=True)
            for rank, (name, rho) in enumerate(ordered, start=1):
                share = rho * rho / denominator if denominator > 0 else 0.0
                ranking_rows.append(
                    {
                        "age_group": age_group,
                        "outcome": outcome,
                        "parameter": name,
                        "spearman_rho": rho,
                        "association_share": share,
                        "association_allocated_variance": share * between,
                        # Clearer aliases retained alongside the original
                        # column names for backwards-compatible CSV readers.
                        "screening_importance_share": share,
                        "screening_allocated_parameter_variance": share * between,
                        "sensitivity_rank": rank,
                    }
                )
    parameter_ranking = pd.DataFrame(ranking_rows)
    return ParameterSensitivityResult(
        design=design,
        simulation_outcomes=simulation_outcomes,
        variance_decomposition=variance_decomposition,
        parameter_ranking=parameter_ranking,
    )


def forecast_parameter_spread_sensitivity(
    cases: pd.DataFrame | None = None,
    inputs: RegionAgeInputs | None = None,
    fitted_parameters_path: Path | str = DEFAULT_LONDON_FITTED_PARAMETERS,
    parameter_ranges: dict[str, tuple[float, float]] | None = None,
    relative_range: float = 0.20,
    levels_per_parameter: int = 5,
    simulations_per_level: int = 30,
    outbreak_threshold: float = 10.0,
    horizon_weeks: int = 6,
    warmup_weeks: int = 0,
    random_seed: int = 20260805,
    current_cases: float | None = None,
    sample_weekly_counts: bool = True,
    noise_scale: float | None = None,
    history_conditioning: HistoryConditioningConfig | None = None,
) -> ParameterSpreadSensitivityResult:
    """Measure an isolated numerical spread for every declared sensitivity parameter.

    One parameter is swept across its scenario range while all remaining
    fitted and fixed values stay at baseline. Repeated stochastic paths estimate
    both the change in the conditional mean and within-level path variance. If
    history conditioning is requested, it is performed once at the baseline and
    the inferred state is held fixed across all parameter levels. This isolates
    future parameter effects and prevents reconditioning from compensating for
    the parameter being tested.

    ``noise_adjusted_parameter_variance`` subtracts the average Monte Carlo
    variance of a level mean from the raw variance across level means. This is
    an exploratory one-at-a-time screening measure, not a Sobol index and not
    a confidence interval for the parameter.
    """

    if levels_per_parameter < 3:
        raise ValueError("levels_per_parameter must be at least 3")
    if simulations_per_level < 2:
        raise ValueError("simulations_per_level must be at least 2")
    if not 0 < relative_range <= 1:
        raise ValueError("relative_range must lie in (0, 1]")

    model_inputs = inputs or load_default_inputs()
    baseline_params, fitted_baseline = load_london_fitted_parameters(fitted_parameters_path)
    conditioned_history = _condition_baseline_for_sensitivity(
        cases,
        model_inputs,
        baseline_params,
        history_conditioning,
        current_cases,
        random_seed,
    )
    baseline = {
        name: (
            float(fitted_baseline[name])
            if name in fitted_baseline
            else float(getattr(baseline_params, name))
        )
        for name in SENSITIVITY_PARAMETERS
    }
    if parameter_ranges is None:
        ranges: dict[str, tuple[float, float]] = {}
        for name, value in baseline.items():
            bound_low, bound_high = SENSITIVITY_PARAMETER_BOUNDS[name]
            half_width = relative_range * max(abs(value), 0.05 * (bound_high - bound_low))
            ranges[name] = (
                float(max(bound_low, value - half_width)),
                float(min(bound_high, value + half_width)),
            )
    else:
        ranges = {name: (float(bounds[0]), float(bounds[1])) for name, bounds in parameter_ranges.items()}
        unknown = set(ranges).difference(SENSITIVITY_PARAMETER_BOUNDS)
        if unknown:
            raise ValueError(f"Sensitivity ranges contain unknown parameters: {sorted(unknown)}")
        for name, (low, high) in ranges.items():
            bound_low, bound_high = SENSITIVITY_PARAMETER_BOUNDS[name]
            if not bound_low <= low < high <= bound_high:
                raise ValueError(
                    f"Range for {name} must satisfy {bound_low} <= low < high <= {bound_high}"
                )

    level_rows: list[dict[str, object]] = []
    outcome_frames: list[pd.DataFrame] = []
    outcome_names = [
        "outbreak_indicator",
        "peak_weekly_cases",
        "cumulative_cases",
        "final_week_cases",
    ]
    for parameter_index, (name, (low, high)) in enumerate(ranges.items()):
        values = np.unique(
            np.concatenate(
                [np.linspace(low, high, levels_per_parameter), np.asarray([baseline[name]])]
            )
        )
        for level_index, value in enumerate(values):
            is_baseline = bool(np.isclose(value, baseline[name], rtol=1e-10, atol=1e-12))
            level_id = f"{name}:{level_index}"
            level_rows.append(
                {
                    "level_id": level_id,
                    "parameter": name,
                    "parameter_value": float(value),
                    "baseline_value": float(baseline[name]),
                    "is_baseline": is_baseline,
                    "tested_lower": low,
                    "tested_upper": high,
                }
            )
            # Common random numbers across all levels make each OAT difference
            # paired: changes are caused by the tested parameter rather than a
            # different set of random paths.
            simulation_seed = random_seed
            result = forecast_london_age_groups(
                cases=cases,
                inputs=model_inputs,
                fitted_parameters_path=fitted_parameters_path,
                fitted_vector_override={name: float(value)},
                outbreak_threshold=outbreak_threshold,
                horizon_weeks=horizon_weeks,
                n_simulations=simulations_per_level,
                random_seed=simulation_seed,
                warmup_weeks=warmup_weeks,
                current_cases=current_cases,
                sample_weekly_counts=sample_weekly_counts,
                noise_scale_override=noise_scale,
                predictive_uncertainty=None,
                history_conditioning_result=conditioned_history,
            )
            age = result.trajectories.groupby(["age_group", "simulation"], as_index=False).agg(
                peak_weekly_cases=("weekly_cases", "max"),
                cumulative_cases=("weekly_cases", "sum"),
                final_week_cases=("weekly_cases", "last"),
            )
            all_age = result.all_age_trajectories.groupby("simulation", as_index=False).agg(
                peak_weekly_cases=("weekly_cases", "max"),
                cumulative_cases=("weekly_cases", "sum"),
                final_week_cases=("weekly_cases", "last"),
            )
            all_age.insert(0, "age_group", "all ages combined")
            outcomes = pd.concat([age, all_age], ignore_index=True)
            outcomes["outbreak_indicator"] = (
                exceeds_outbreak_threshold(outcomes["peak_weekly_cases"], outbreak_threshold)
            ).astype(float)
            outcomes.insert(0, "parameter_value", float(value))
            outcomes.insert(0, "parameter", name)
            outcomes.insert(0, "level_id", level_id)
            outcome_frames.append(outcomes)

    parameter_levels = pd.DataFrame(level_rows)
    simulation_outcomes = pd.concat(outcome_frames, ignore_index=True)
    long_outcomes = simulation_outcomes.melt(
        id_vars=["level_id", "parameter", "parameter_value", "age_group", "simulation"],
        value_vars=outcome_names,
        var_name="outcome",
        value_name="outcome_value",
    )
    level_summary = (
        long_outcomes.groupby(
            ["level_id", "parameter", "parameter_value", "age_group", "outcome"],
            as_index=False,
        )
        .outcome_value.agg(
            conditional_mean="mean",
            conditional_variance=lambda values: float(np.var(values.to_numpy(dtype=float), ddof=0)),
            simulations="size",
        )
        .merge(parameter_levels[["level_id", "baseline_value", "is_baseline"]], on="level_id")
    )

    spread_rows: list[dict[str, object]] = []
    for (name, age_group, outcome), group in level_summary.groupby(
        ["parameter", "age_group", "outcome"], sort=False
    ):
        group = group.sort_values("parameter_value").reset_index(drop=True)
        means = group["conditional_mean"].to_numpy(dtype=float)
        raw_between = float(np.var(means, ddof=0))
        within = float(group["conditional_variance"].mean())
        n_repeats = float(group["simulations"].mean())
        monte_carlo_mean_variance = within / max(n_repeats, 1.0)
        adjusted_between = max(raw_between - monte_carlo_mean_variance, 0.0)
        baseline_rows = group.loc[group["is_baseline"]]
        baseline_mean = float(
            baseline_rows.iloc[0]["conditional_mean"]
            if not baseline_rows.empty
            else group.iloc[(group.parameter_value - group.baseline_value).abs().argmin()]["conditional_mean"]
        )
        low_level_mean = float(group.iloc[0]["conditional_mean"])
        high_level_mean = float(group.iloc[-1]["conditional_mean"])
        low_to_high_change = high_level_mean - low_level_mean
        outcome_spread = float(means.max() - means.min())
        total = adjusted_between + within
        spread_rows.append(
            {
                "parameter": name,
                "age_group": age_group,
                "outcome": outcome,
                "baseline_parameter_value": float(group["baseline_value"].iloc[0]),
                "tested_lower": float(group["parameter_value"].min()),
                "tested_upper": float(group["parameter_value"].max()),
                "baseline_outcome_mean": baseline_mean,
                "minimum_level_mean": float(means.min()),
                "maximum_level_mean": float(means.max()),
                "low_level_outcome_mean": low_level_mean,
                "high_level_outcome_mean": high_level_mean,
                "low_to_high_outcome_change": low_to_high_change,
                "outcome_mean_spread": outcome_spread,
                "relative_outcome_spread": outcome_spread / max(abs(baseline_mean), 1e-12),
                "outbreak_probability_spread_percentage_points": (
                    100.0 * outcome_spread if outcome == "outbreak_indicator" else np.nan
                ),
                "outbreak_probability_low_to_high_change_percentage_points": (
                    100.0 * low_to_high_change
                    if outcome == "outbreak_indicator"
                    else np.nan
                ),
                "raw_variance_across_level_means": raw_between,
                "monte_carlo_variance_of_level_mean": monte_carlo_mean_variance,
                "noise_adjusted_parameter_variance": adjusted_between,
                "mean_within_level_stochastic_variance": within,
                "oat_total_predictive_variance": total,
                "oat_parameter_variance_fraction": adjusted_between / total if total > 0 else 0.0,
                "oat_stochastic_variance_fraction": within / total if total > 0 else 0.0,
                "parameter_signal_to_mc_noise": (
                    adjusted_between / monte_carlo_mean_variance
                    if monte_carlo_mean_variance > 0
                    else np.inf if adjusted_between > 0 else 0.0
                ),
            }
        )
    parameter_spread = pd.DataFrame(spread_rows)
    group_total = parameter_spread.groupby(["age_group", "outcome"])[
        "noise_adjusted_parameter_variance"
    ].transform("sum")
    parameter_spread["isolated_variance_screening_share"] = np.where(
        group_total > 0,
        parameter_spread["noise_adjusted_parameter_variance"] / group_total,
        0.0,
    )
    # Presentation-friendly alias. Across parameters this sums to 100% for a
    # given age group and outcome. It is a relative OAT screening allocation,
    # not a joint Sobol index and not a universal causal percentage.
    parameter_spread["relative_parameter_variance_share_percent"] = (
        100.0 * parameter_spread["isolated_variance_screening_share"]
    )
    parameter_spread["sensitivity_rank"] = parameter_spread.groupby(
        ["age_group", "outcome"]
    )["noise_adjusted_parameter_variance"].rank(method="dense", ascending=False)
    return ParameterSpreadSensitivityResult(
        parameter_levels=parameter_levels,
        simulation_outcomes=simulation_outcomes,
        level_summary=level_summary,
        parameter_spread=parameter_spread,
    )


def london_age_case_weights(
    inputs: RegionAgeInputs,
    forecast_origin: pd.Timestamp,
    path: Path | str | None = DEFAULT_LONDON_AGE_CASES,
    prior_strength: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return leakage-safe London age-case proportions and Dirichlet alpha.

    The latest period whose ``period_end`` is not after the forecast origin is
    selected. Observed counts are shrunk toward unprotected-population weights
    with ``prior_strength`` pseudo-cases. The resulting alpha vector is also
    used to sample age-allocation uncertainty per stochastic path.
    """

    london_index = inputs.regions.index("London")
    fallback = _target_age_weights(inputs, london_index)
    if path is None or not Path(path).exists():
        alpha = np.maximum(fallback * max(float(prior_strength), 1.0), 1e-6)
        table = pd.DataFrame(
            {
                "age_group": inputs.age_groups,
                "observed_age_cases": np.nan,
                "allocation_weight": fallback,
                "source_period_end": pd.NaT,
                "allocation_source": "unprotected-population fallback",
            }
        )
        return fallback, alpha, table

    history = pd.read_csv(path)
    required = {"period_end", "age_group", "london_cases"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"London age-case file is missing columns: {sorted(missing)}")
    history["period_end"] = pd.to_datetime(history["period_end"])
    eligible = history.loc[history["period_end"].le(pd.Timestamp(forecast_origin))].copy()
    if eligible.empty:
        return london_age_case_weights(inputs, forecast_origin, path=None, prior_strength=prior_strength)
    selected_end = eligible["period_end"].max()
    selected = eligible.loc[eligible["period_end"].eq(selected_end)]
    counts = (
        selected.set_index("age_group")["london_cases"]
        .reindex(inputs.age_groups)
        .astype(float)
    )
    if counts.isna().any() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Selected London age-case period is incomplete or invalid")
    alpha = counts.to_numpy(dtype=float) + max(float(prior_strength), 0.0) * fallback
    alpha = np.maximum(alpha, 1e-6)
    weights = alpha / alpha.sum()
    table = pd.DataFrame(
        {
            "age_group": inputs.age_groups,
            "observed_age_cases": counts.to_numpy(dtype=float),
            "allocation_weight": weights,
            "source_period_end": selected_end,
            "allocation_source": "UKHSA London age cases + unprotected-population shrinkage",
        }
    )
    return weights, alpha, table


def _sample_forecast_parameters(
    base: ModelParameters,
    fitted_vector: dict[str, float],
    uncertainty: MathsyPredictiveUncertainty,
    rng: np.random.Generator,
) -> tuple[ModelParameters, dict[str, float]]:
    """Draw one bounded fitted-parameter vector for a future SDE path."""

    if uncertainty.parameter_relative_sd < 0 or uncertainty.initial_state_log_sd < 0:
        raise ValueError("uncertainty standard deviations must be non-negative")
    if uncertainty.seed_dispersion <= 0:
        raise ValueError("seed_dispersion must be positive")
    drawn = dict(fitted_vector)
    for name, value in fitted_vector.items():
        low, high = FIT_PARAMETER_BOUNDS[name]
        standard_deviation = uncertainty.parameter_relative_sd * max(abs(value), 0.02 * (high - low))
        drawn[name] = float(np.clip(rng.normal(value, standard_deviation), low, high))

    # Copy only declared scalar fields, then attach calibration-only values.
    simulation_params = ModelParameters(
        **{name: getattr(base, name) for name in ModelParameters.__dataclass_fields__}
    )
    for name, value in drawn.items():
        if hasattr(simulation_params, name):
            setattr(simulation_params, name, float(value))
    simulation_params.contact_scale = float(drawn["contact_scale"])
    simulation_params.initial_exposed_per_case = float(drawn["initial_exposed_per_case"])
    simulation_params.initial_infectious_per_case = float(drawn["initial_infectious_per_case"])
    return simulation_params, drawn


def load_london_confirmed_cases_from_frame(cases: pd.DataFrame) -> pd.DataFrame:
    """Validate an already-loaded London case frame using the same rules."""

    required = {"date", "observed_cases"}
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"London case frame is missing columns: {sorted(missing)}")
    validated = cases[["date", "observed_cases"]].copy()
    validated["date"] = pd.to_datetime(validated["date"])
    validated["observed_cases"] = pd.to_numeric(validated["observed_cases"], errors="raise")
    validated = validated.sort_values("date").reset_index(drop=True)
    if validated.empty or (validated["observed_cases"] < 0).any():
        raise ValueError("London confirmed cases must be non-empty and non-negative.")
    if validated["date"].duplicated().any():
        raise ValueError("London confirmed cases contain duplicate weekly dates.")
    gaps = validated["date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=7)).all():
        raise ValueError("London confirmed case dates must be consecutive seven-day intervals.")
    return validated


def make_balanced_blocked_validation_design(
    cases: pd.DataFrame,
    *,
    training_weeks: int = 40,
    horizon_weeks: int = 6,
    outbreak_threshold: float = 10.0,
    random_seed: int = 20260890,
) -> pd.DataFrame:
    """Create a leakage-safe balanced design of non-overlapping time blocks.

    The first ``training_weeks`` observations are reserved for the one frozen
    global calibration.  Starting immediately afterwards, the remaining
    observations are divided into disjoint ``horizon_weeks`` outcome blocks.
    A block is positive when any observed outcome is strictly above the
    outbreak threshold.  Every block in the minority class is retained and a
    time-stratified seeded sample of equal size is taken from the majority
    class *before any model forecasts are inspected*.  Time stratification
    prevents an unlucky random draw from putting one outcome class mostly at
    the beginning or end of the held-out calendar.

    The resulting selected blocks therefore have equal positive and negative
    counts and no shared outcome weeks.  Later origins may use observations
    from earlier test blocks for four-week state conditioning, as they would
    in a genuine real-time rolling forecast, but those observations never
    update the frozen global parameter fit.
    """

    ordered = load_london_confirmed_cases_from_frame(cases)
    training_weeks = int(training_weeks)
    horizon_weeks = int(horizon_weeks)
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be at least 1")
    if training_weeks < 4:
        raise ValueError("training_weeks must leave at least four conditioning weeks")
    if training_weeks + horizon_weeks > len(ordered):
        raise ValueError("case history is too short for training plus one outcome block")

    rows: list[dict[str, object]] = []
    for block_id, forecast_start_index in enumerate(
        range(training_weeks, len(ordered) - horizon_weeks + 1, horizon_weeks)
    ):
        origin_index = forecast_start_index - 1
        future = ordered.iloc[
            forecast_start_index : forecast_start_index + horizon_weeks
        ]
        observed_event = bool(
            exceeds_outbreak_threshold(
                future["observed_cases"].to_numpy(dtype=float).max(),
                outbreak_threshold,
            )
        )
        rows.append(
            {
                "block_id": block_id,
                "cutoff_index": forecast_start_index,
                "origin_index": origin_index,
                "origin_date": pd.Timestamp(ordered.iloc[origin_index]["date"]),
                "forecast_start": pd.Timestamp(future.iloc[0]["date"]),
                "forecast_end": pd.Timestamp(future.iloc[-1]["date"]),
                "origin_cases": float(ordered.iloc[origin_index]["observed_cases"]),
                "future_maximum": float(future["observed_cases"].max()),
                "observed_event": observed_event,
                "class_label": "outbreak" if observed_event else "no_outbreak",
                "future_cases": " -> ".join(
                    f"{value:g}" for value in future["observed_cases"]
                ),
            }
        )
    design = pd.DataFrame(rows)
    class_counts = design["observed_event"].value_counts()
    if len(class_counts) != 2:
        raise ValueError(
            "post-training blocks must contain at least one outbreak and one "
            "non-outbreak block to construct a balanced test"
        )
    per_class = int(class_counts.min())
    rng = np.random.default_rng(int(random_seed))
    selected_ids: list[int] = []
    for observed_event in (False, True):
        candidates = design.loc[
            design["observed_event"].eq(observed_event), "block_id"
        ].sort_values()
        if len(candidates) == per_class:
            chosen = candidates.to_numpy(dtype=int)
        else:
            calendar_strata = np.array_split(candidates.to_numpy(dtype=int), per_class)
            chosen = np.asarray(
                [rng.choice(stratum) for stratum in calendar_strata], dtype=int
            )
        selected_ids.extend(map(int, chosen))
    design["selected_for_balanced_test"] = design["block_id"].isin(selected_ids)
    design["training_end_date"] = pd.Timestamp(ordered.iloc[training_weeks - 1]["date"])
    design["training_weeks"] = training_weeks
    design["horizon_weeks"] = horizon_weeks
    design["outbreak_threshold"] = float(outbreak_threshold)
    design["selection_random_seed"] = int(random_seed)
    return design


def binary_classification_metrics_across_cutoffs(
    forecasts: pd.DataFrame,
    *,
    probability_column: str = "predicted_probability",
    outcome_column: str = "observed_event",
    probability_cutoffs: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Calculate confusion counts and diagnostic rates at many alarm cutoffs."""

    required = {probability_column, outcome_column}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"forecast table is missing columns: {sorted(missing)}")
    probability = pd.to_numeric(forecasts[probability_column], errors="raise").to_numpy()
    actual = forecasts[outcome_column].astype(bool).to_numpy()
    if probability.size == 0 or not np.isfinite(probability).all():
        raise ValueError("forecast probabilities must be non-empty and finite")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("forecast probabilities must lie between zero and one")
    cutoffs = (
        np.linspace(0.0, 1.0, 21)
        if probability_cutoffs is None
        else np.asarray(list(probability_cutoffs), dtype=float)
    )
    if cutoffs.size == 0 or not np.isfinite(cutoffs).all():
        raise ValueError("probability_cutoffs must be non-empty and finite")

    def safe_rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else np.nan

    rows: list[dict[str, float | int]] = []
    for cutoff in cutoffs:
        if cutoff < 0 or cutoff > 1:
            raise ValueError("probability cutoffs must lie between zero and one")
        predicted = probability >= cutoff
        tp = int((actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        fp = int((~actual & predicted).sum())
        tn = int((~actual & ~predicted).sum())
        sensitivity = safe_rate(tp, tp + fn)
        specificity = safe_rate(tn, tn + fp)
        rows.append(
            {
                "probability_cutoff": float(cutoff),
                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "true_negative": tn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
                "precision": safe_rate(tp, tp + fp),
                "negative_predictive_value": safe_rate(tn, tn + fn),
                "false_alarm_rate": safe_rate(fp, fp + tn),
                "miss_rate": safe_rate(fn, fn + tp),
                "accuracy": safe_rate(tp + tn, len(actual)),
            }
        )
    return pd.DataFrame(rows)


def strict_forecast_validation(
    cases: pd.DataFrame,
    inputs: RegionAgeInputs | None = None,
    config: CalibrationConfig | None = None,
    cutoff_indices: Iterable[int] | None = None,
    horizon_weeks: int | None = None,
    n_stochastic_simulations: int = 10,
    outbreak_threshold: float = 10.0,
    base_parameters: ModelParameters | None = None,
    history_conditioning: HistoryConditioningConfig | None = None,
    refit_parameters_each_cutoff: bool = False,
    predictive_uncertainty: MathsyPredictiveUncertainty | None = None,
    objective_metric: str = "composite",
    fitted_vector_override: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    starting_vector: dict[str, float] | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate London-outcome-held-out forecasts at selected cutoffs.

    By default, global parameters are calibrated once using only observations
    before the earliest cutoff and then held fixed across all origins. Set
    ``refit_parameters_each_cutoff=True`` only for comparison with the legacy
    expanding-window procedure. The final observed training week supplies the initial reported
    reference; the following observations are passed only as dates and are
    never used to initialize or fit the simulation. Static population,
    protection, contact, mobility and regional-risk inputs are shared across
    cutoffs and must not be described as historically vintaged covariates.
    ``fitted_vector_override`` skips that calibration and holds a supplied
    fitted mean-dynamics vector fixed.  This is intended for stochastic-setting
    sensitivity checks in which refitting the deterministic mean is neither
    necessary nor desirable. ``parameter_bounds`` and ``starting_vector`` allow
    the same leakage-safe workflow to fit either the non-seasonal eight-parameter
    model or the ten-parameter seasonal extension.
    """

    if not isinstance(cases, pd.DataFrame) or not {"date", "observed_cases"}.issubset(cases.columns):
        raise ValueError("cases must contain date and observed_cases columns")
    ordered_cases = cases[["date", "observed_cases"]].copy().reset_index(drop=True)
    ordered_cases["date"] = pd.to_datetime(ordered_cases["date"])
    cfg = config or CalibrationConfig()
    fit_bounds = parameter_bounds or FIT_PARAMETER_BOUNDS
    if int(cfg.warmup_weeks) != 0:
        raise ValueError("Strict validation requires warmup_weeks=0 for date alignment")
    horizon = int(horizon_weeks or cfg.block_weeks)
    if horizon < 2:
        raise ValueError("horizon_weeks must be at least 2")
    if cutoff_indices is None:
        cutoff_indices = range(cfg.block_weeks + 1, len(ordered_cases) - horizon + 1, horizon)
    cutoff_indices = [int(cutoff) for cutoff in cutoff_indices]
    if not cutoff_indices:
        raise ValueError("At least one cutoff index is required")

    model_inputs = inputs or load_default_inputs()
    base = base_parameters or default_calibration_parameters()
    summary_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    shared_fit: tuple[dict[str, float], float, int] | None = None
    if fitted_vector_override is not None:
        missing = set(fit_bounds).difference(fitted_vector_override)
        if missing:
            raise ValueError(
                "fitted_vector_override is missing fitted parameters: "
                f"{sorted(missing)}"
            )
        supplied = {
            name: float(fitted_vector_override[name]) for name in fit_bounds
        }
        for name, value in supplied.items():
            low, high = fit_bounds[name]
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(
                    f"fitted_vector_override[{name!r}]={value} is outside "
                    f"the fitted range [{low}, {high}]"
                )
        shared_fit = (supplied, np.nan, 0)
    elif not refit_parameters_each_cutoff:
        earliest_cutoff = min(cutoff_indices)
        shared_training = ordered_cases.iloc[:earliest_cutoff].copy().reset_index(drop=True)
        # Use every rolling origin. ``overlap_weighted=True`` below corrects
        # for repeated calendar weeks, while the additional origins constrain
        # the dynamics much better than a small set of disjoint blocks.
        shared_blocks = make_six_week_blocks(
            shared_training, block_weeks=cfg.block_weeks, step_weeks=1
        )
        shared_seed = cfg.random_seed + earliest_cutoff * 1009
        shared_vector, _, shared_diagnostics = fit_shared_parameters(
            shared_blocks,
            inputs=model_inputs,
            config=replace(cfg, random_seed=shared_seed),
            base_parameters=base,
            progress=progress,
            overlap_weighted=True,
            objective_metric=objective_metric,
            starting_vector=starting_vector,
            parameter_bounds=fit_bounds,
        )
        shared_objective = float(
            np.average(
                shared_diagnostics["objective"],
                weights=shared_diagnostics["window_weight"],
            )
        )
        shared_fit = (shared_vector, shared_objective, int(shared_blocks.block_id.nunique()))

    for cutoff_id, cutoff in enumerate(cutoff_indices):
        if cutoff < cfg.block_weeks + 1 or cutoff + horizon > len(ordered_cases):
            raise ValueError(
                f"cutoff index {cutoff} must leave at least {horizon} future weeks "
                "and at least one training window"
            )
        training = ordered_cases.iloc[:cutoff].copy().reset_index(drop=True)
        future = ordered_cases.iloc[cutoff : cutoff + horizon].copy().reset_index(drop=True)
        # Use all historical rolling origins. Overlap weighting ensures that a
        # calendar week does not receive extra objective weight merely because
        # it appears in several windows.
        training_blocks = make_six_week_blocks(
            training, block_weeks=cfg.block_weeks, step_weeks=1
        )
        fit_seed = cfg.random_seed + cutoff * 1009
        if shared_fit is None:
            best_vector, _, diagnostics = fit_shared_parameters(
                training_blocks,
                inputs=model_inputs,
                config=replace(cfg, random_seed=fit_seed),
                base_parameters=base,
                progress=progress,
                overlap_weighted=True,
                objective_metric=objective_metric,
                starting_vector=starting_vector,
                parameter_bounds=fit_bounds,
            )
            fit_objective = float(
                np.average(diagnostics["objective"], weights=diagnostics["window_weight"])
            )
            fitted_training_windows = int(training_blocks.block_id.nunique())
        else:
            best_vector, fit_objective, fitted_training_windows = shared_fit
        reference_cases = float(training.iloc[-1]["observed_cases"])
        deterministic_params = _with_vector(base, best_vector, force_deterministic=True)
        conditioned = None
        validation_inputs = replace(
            model_inputs,
            contact_matrix=(
                model_inputs.contact_matrix
                * float(getattr(deterministic_params, "contact_scale", 1.0))
            ),
        )
        initial_e, initial_i, initial_q = _initial_state_arrays(
            model_inputs, reference_cases, deterministic_params, region="London"
        )
        initial_state = None
        if history_conditioning is not None:
            conditioned = condition_on_recent_history(
                training,
                model_inputs,
                deterministic_params,
                history_conditioning,
                region="London",
                random_seed=fit_seed + 1,
            )
            initial_state = conditioned.state
            validation_inputs = replace(
                model_inputs,
                contact_matrix=(
                    model_inputs.contact_matrix
                    * float(getattr(deterministic_params, "contact_scale", 1.0))
                    * conditioned.transmission_multiplier
                ),
            )
        expected_full = _run_one_simulation(
            inputs=validation_inputs,
            params=deterministic_params,
            initial_reported_sick=initial_q,
            initial_exposed=initial_e,
            initial_infectious=initial_i,
            initial_state=initial_state,
            horizon_weeks=horizon,
            random_seed=fit_seed + 1,
            sample_weekly_counts=False,
            seed_region="London",
            simulation_start_date=pd.Timestamp(future.iloc[0]["date"]),
        )
        london_index = model_inputs.regions.index("London")
        expected = np.asarray(expected_full[:, london_index, :], dtype=float).sum(axis=1)
        stochastic_params = _with_vector(base, best_vector, force_deterministic=False)
        stochastic_paths = []
        uncertainty = predictive_uncertainty or MathsyPredictiveUncertainty(
            parameter_relative_sd=0.0,
            initial_state_log_sd=0.0,
        )
        stochastic_rng = np.random.default_rng(fit_seed + 9999)
        for simulation in range(max(1, int(n_stochastic_simulations))):
            seed_mean = max(float(stochastic_params.seed_infections_per_week), 0.0)
            seed_shape = float(uncertainty.seed_dispersion)
            weekly_seed_schedule = stochastic_rng.gamma(
                shape=seed_shape,
                scale=(seed_mean / seed_shape if seed_shape > 0 else 0.0),
                size=horizon,
            )
            stochastic_full = _run_one_simulation(
                inputs=validation_inputs,
                params=stochastic_params,
                initial_reported_sick=initial_q,
                initial_exposed=initial_e,
                initial_infectious=initial_i,
                initial_state=initial_state,
                horizon_weeks=horizon,
                random_seed=fit_seed + 10000 + simulation,
                sample_weekly_counts=True,
                weekly_seed_infections=weekly_seed_schedule,
                seed_region="London",
                simulation_start_date=pd.Timestamp(future.iloc[0]["date"]),
            )
            stochastic_paths.append(
                np.asarray(stochastic_full[:, london_index, :], dtype=float).sum(axis=1)
            )
        stochastic_array = np.asarray(stochastic_paths, dtype=float)
        observed_event = bool(
            exceeds_outbreak_threshold(future["observed_cases"].max(), outbreak_threshold)
        )
        path_events = exceeds_outbreak_threshold(
            stochastic_array.max(axis=1), outbreak_threshold
        )
        predicted_event_probability = float(path_events.mean())
        persistence_event_probability = float(
            exceeds_outbreak_threshold(reference_cases, outbreak_threshold)
        )
        event_brier_score = float(
            (predicted_event_probability - float(observed_event)) ** 2
        )
        persistence_event_brier_score = float(
            (persistence_event_probability - float(observed_event)) ** 2
        )
        for week, (date, observed, expected_value) in enumerate(
            zip(future["date"], future["observed_cases"], expected), start=1
        ):
            values = stochastic_array[:, week - 1]
            summary_rows.append(
                {
                    "cutoff_id": cutoff_id,
                    "cutoff_index": cutoff,
                    "training_end_date": training.iloc[-1]["date"],
                    "forecast_week": week,
                    "date": date,
                    "reference_cases": reference_cases,
                    "history_conditioning_enabled": history_conditioning is not None,
                    "history_weeks": (
                        0 if history_conditioning is None else history_conditioning.history_weeks
                    ),
                    "recent_transmission_multiplier": (
                        1.0 if conditioned is None else conditioned.transmission_multiplier
                    ),
                    "history_conditioning_objective": (
                        np.nan if conditioned is None else conditioned.objective
                    ),
                    "history_conditioning_optimizer_success": (
                        False if conditioned is None else conditioned.optimizer_success
                    ),
                    "anchor_exposed_per_case": (
                        np.nan if conditioned is None else conditioned.initial_exposed_per_case
                    ),
                    "anchor_infectious_per_case": (
                        np.nan if conditioned is None else conditioned.initial_infectious_per_case
                    ),
                    "origin_exposed_total": (
                        np.nan if conditioned is None else conditioned.origin_exposed_total
                    ),
                    "origin_infectious_total": (
                        np.nan if conditioned is None else conditioned.origin_infectious_total
                    ),
                    "origin_sick_total": (
                        np.nan if conditioned is None else conditioned.origin_sick_total
                    ),
                    "observed_cases": float(observed),
                    "fitted_expected_cases": float(expected_value),
                    "p10_cases": float(np.percentile(values, 10)),
                    "median_cases": float(np.percentile(values, 50)),
                    "p90_cases": float(np.percentile(values, 90)),
                    "training_windows": int(training_blocks.block_id.nunique()),
                    "global_fit_training_windows": fitted_training_windows,
                    "global_parameters_refitted_each_cutoff": refit_parameters_each_cutoff,
                    "training_objective": fit_objective,
                    "outbreak_threshold": float(outbreak_threshold),
                    "observed_six_week_event": observed_event,
                    "predicted_six_week_event_probability": predicted_event_probability,
                    "event_brier_score": event_brier_score,
                    "persistence_event_probability": persistence_event_probability,
                    "persistence_event_brier_score": persistence_event_brier_score,
                }
            )
            for simulation, path in enumerate(stochastic_array, start=1):
                path_rows.append(
                    {
                        "cutoff_id": cutoff_id,
                        "cutoff_index": cutoff,
                        "training_end_date": training.iloc[-1]["date"],
                        "simulation": simulation,
                        "forecast_week": week,
                        "date": date,
                        "weekly_cases": float(path[week - 1]),
                        "observed_cases": float(observed),
                    }
                )
        fit_rows.append(
            {
                "cutoff_id": cutoff_id,
                "cutoff_index": cutoff,
                "training_end_date": training.iloc[-1]["date"],
                "training_weeks": len(training),
                "training_windows": int(training_blocks.block_id.nunique()),
                "global_fit_training_windows": fitted_training_windows,
                "global_parameters_refitted_each_cutoff": refit_parameters_each_cutoff,
                "training_objective": fit_objective,
                "outbreak_threshold": float(outbreak_threshold),
                "observed_six_week_event": observed_event,
                "predicted_six_week_event_probability": predicted_event_probability,
                "event_brier_score": event_brier_score,
                "persistence_event_probability": persistence_event_probability,
                "persistence_event_brier_score": persistence_event_brier_score,
                "history_conditioning_enabled": history_conditioning is not None,
                "history_weeks": (
                    0 if history_conditioning is None else history_conditioning.history_weeks
                ),
                "recent_transmission_multiplier": (
                    1.0 if conditioned is None else conditioned.transmission_multiplier
                ),
                "history_conditioning_objective": (
                    np.nan if conditioned is None else conditioned.objective
                ),
                "history_conditioning_optimizer_success": (
                    False if conditioned is None else conditioned.optimizer_success
                ),
                "anchor_exposed_per_case": (
                    np.nan if conditioned is None else conditioned.initial_exposed_per_case
                ),
                "anchor_infectious_per_case": (
                    np.nan if conditioned is None else conditioned.initial_infectious_per_case
                ),
                "origin_exposed_total": (
                    np.nan if conditioned is None else conditioned.origin_exposed_total
                ),
                "origin_infectious_total": (
                    np.nan if conditioned is None else conditioned.origin_infectious_total
                ),
                "origin_sick_total": (
                    np.nan if conditioned is None else conditioned.origin_sick_total
                ),
                **best_vector,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(path_rows), pd.DataFrame(fit_rows)


def evaluate_vector(
    vector: dict[str, float],
    blocks: pd.DataFrame,
    inputs: RegionAgeInputs,
    base_parameters: ModelParameters | None = None,
    stochastic_repeats: int = 1,
    seed: int = 0,
    force_deterministic: bool = True,
    warmup_weeks: int = 0,
    overlap_weighted: bool = False,
    prediction_summary: str = "mean",
    objective_metric: str = "composite",
    sample_weekly_counts: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Score a candidate against every six-week block.

    Each block contributes equally after normalising by its mean case count.
    For rolling windows, ``overlap_weighted=True`` gives calendar weeks equal
    total weight instead of counting a week once for every window containing
    it. The returned table is useful for diagnosing which historical periods
    are difficult to fit.
    """

    if prediction_summary not in {"mean", "median"}:
        raise ValueError("prediction_summary must be 'mean' or 'median'")
    if objective_metric not in {
        "mean_squared_error",
        "composite",
        "normalised_rmse",
        "poisson_nll",
    }:
        raise ValueError(
            "objective_metric must be 'mean_squared_error', 'composite', "
            "'normalised_rmse', or 'poisson_nll'"
        )

    base = base_parameters or default_calibration_parameters()
    params = _with_vector(base, vector, force_deterministic=force_deterministic)
    rows = []
    for block_id, block in blocks.groupby("block_id", sort=True):
        # Match the live forecast's age allocation without using information
        # that was unavailable at the historical origin. For early windows
        # with no completed age table, the helper returns the documented
        # unprotected-population fallback.
        historical_age_weights = None
        if set(inputs.age_groups).issubset(set(AGE_GROUPS)):
            origin = pd.Timestamp(block.iloc[0]["conditioning_date"])
            historical_age_weights, _, _ = london_age_case_weights(
                inputs,
                forecast_origin=origin,
                prior_strength=0.0,
            )
        observed = block["observed_cases"].to_numpy(dtype=float)
        predictions = []
        for repeat in range(max(1, stochastic_repeats)):
            predictions.append(
                _simulate_block(
                    inputs,
                    params,
                    block,
                    seed + int(block_id) * 1009 + repeat,
                    warmup_weeks=warmup_weeks,
                    sample_weekly_counts=sample_weekly_counts,
                    age_weights=historical_age_weights,
                )
            )
        prediction_array = np.asarray(predictions)
        if prediction_summary == "median":
            predicted = np.median(prediction_array, axis=0)
        else:
            predicted = np.mean(prediction_array, axis=0)
        # All target weeks follow the explicit observed week-0 anchor and are
        # scored against confirmed incidence.
        observed_for_score = observed
        predicted_for_score = predicted
        week_weights = (
            1.0 / block["week_coverage"].to_numpy(dtype=float)
            if overlap_weighted and "week_coverage" in block.columns
            else np.ones_like(observed_for_score, dtype=float)
        )
        scale = max(float(np.average(observed_for_score, weights=week_weights)), 1.0)
        residual = predicted_for_score - observed_for_score
        mse = float(np.average(residual**2, weights=week_weights))
        rmse = float(np.sqrt(np.average(residual**2, weights=week_weights)))
        mae = float(np.average(np.abs(residual), weights=week_weights))
        log_rmse = float(
            np.sqrt(
                np.average(
                    (np.log1p(predicted_for_score) - np.log1p(observed_for_score)) ** 2,
                    weights=week_weights,
                )
            )
        )
        # Poisson negative log-likelihood for weekly counts. ``gammaln`` adds
        # log(y!), which is constant with respect to the fitted parameters but
        # keeps the reported score non-negative and interpretable. Omitting
        # that constant gives the same selected parameter vector.
        poisson_mean = np.maximum(predicted_for_score, 1e-12)
        poisson_nll = float(
            np.average(
                poisson_mean
                - observed_for_score * np.log(poisson_mean)
                + gammaln(observed_for_score + 1.0),
                weights=week_weights,
            )
        )
        if objective_metric == "mean_squared_error":
            # Teaching-first objective: average the ordinary squared vertical
            # distances between the deterministic Mathsy curve and the data.
            # There is no count distribution, likelihood or log transform.
            objective = mse
        elif objective_metric == "normalised_rmse":
            objective = rmse / scale
        elif objective_metric == "poisson_nll":
            objective = poisson_nll
        else:
            objective = rmse / scale + 0.5 * log_rmse
        rows.append(
            {
                "block_id": int(block_id),
                "start_date": block.iloc[0]["date"],
                "end_date": block.iloc[-1]["date"],
                "observed_total": float(observed.sum()),
                "predicted_total": float(predicted.sum()),
                "mae": mae,
                "mean_squared_error": mse,
                "rmse": rmse,
                "normalised_rmse": rmse / scale,
                "log_rmse": log_rmse,
                "poisson_nll": poisson_nll,
                "objective": objective,
                "prediction_summary": prediction_summary,
                "objective_metric": objective_metric,
                "window_weight": (
                    float(week_weights.sum())
                    if overlap_weighted and "week_coverage" in block.columns
                    else 1.0
                ),
                "observed": observed,
                "predicted": predicted,
            }
        )
    diagnostics = pd.DataFrame(rows)
    if overlap_weighted:
        score = float(
            np.average(
                diagnostics["objective"].to_numpy(dtype=float),
                weights=diagnostics["window_weight"].to_numpy(dtype=float),
            )
        )
    else:
        score = float(diagnostics["objective"].mean())
    return score, diagnostics


def _sample_vector(
    rng: np.random.Generator,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    bounds = parameter_bounds or FIT_PARAMETER_BOUNDS
    values = {}
    for name, (low, high) in bounds.items():
        if name == "contact_scale":
            values[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            values[name] = float(rng.uniform(low, high))
    return values


def calibration_starting_vector(
    base_parameters: ModelParameters | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Return the exact starting values used by ``fit_shared_parameters``."""

    base = base_parameters or default_calibration_parameters()
    bounds = parameter_bounds or FIT_PARAMETER_BOUNDS
    vector = {
        name: float(getattr(base, name, 1.0))
        for name in bounds
        if name not in {"initial_exposed_per_case", "initial_infectious_per_case"}
    }
    vector["initial_exposed_per_case"] = 1.0
    vector["initial_infectious_per_case"] = 1.0
    vector["contact_scale"] = 0.05
    for name, (low, high) in bounds.items():
        vector[name] = float(np.clip(vector[name], low, high))
    return vector


def fitted_parameter_table(
    base_parameters: ModelParameters | None = None,
    selected_vector: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """List every fitted parameter, its bounds, and optional selected value."""

    bounds = parameter_bounds or FIT_PARAMETER_BOUNDS
    starting = calibration_starting_vector(base_parameters, bounds)
    rows: list[dict[str, object]] = []
    for name, (low, high) in bounds.items():
        selected = np.nan if selected_vector is None else float(selected_vector[name])
        row: dict[str, object] = {
            "parameter": name,
            "parameter_type": (
                "hidden initial state"
                if name in {"initial_exposed_per_case", "initial_infectious_per_case"}
                else "dynamic model parameter"
            ),
            "starting_value": starting[name],
            "lower_bound": float(low),
            "upper_bound": float(high),
            "meaning": PARAMETER_PROVENANCE[name][2],
        }
        if selected_vector is not None:
            row.update(
                {
                    "selected_value": selected,
                    "change_from_start": selected - starting[name],
                    "moved_from_start": not np.isclose(
                        selected, starting[name], rtol=1e-7, atol=1e-10
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_shared_parameters(
    blocks: pd.DataFrame,
    inputs: RegionAgeInputs | None = None,
    config: CalibrationConfig | None = None,
    base_parameters: ModelParameters | None = None,
    progress: bool = True,
    overlap_weighted: bool = False,
    stochastic_repeats: int = 1,
    force_deterministic: bool = True,
    prediction_summary: str = "mean",
    objective_metric: str = "composite",
    sample_weekly_counts: bool = False,
    starting_vector: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Random-search shared parameters, followed by local random refinement.

    By default this fits the deterministic mean path with the original
    composite objective. Set ``force_deterministic=False``,
    ``prediction_summary="median"``, and ``objective_metric="normalised_rmse"``
    to fit an ensemble median using normalised RMSE only.
    """

    cfg = config or CalibrationConfig()
    model_inputs = inputs or load_default_inputs()
    base = base_parameters or default_calibration_parameters()
    bounds = parameter_bounds or FIT_PARAMETER_BOUNDS
    rng = np.random.default_rng(cfg.random_seed)
    trials: list[dict[str, float]] = []
    best_vector: dict[str, float] | None = None
    best_score = np.inf

    # Always evaluate a transparent starting point. This gives the search a
    # scientifically readable reference and avoids losing a good existing
    # parameter scale when the random global trials are wide.
    selected_start = (
        calibration_starting_vector(base, bounds)
        if starting_vector is None
        else {name: float(starting_vector[name]) for name in bounds}
    )
    for name, (low, high) in bounds.items():
        if not low <= selected_start[name] <= high:
            raise ValueError(
                f"starting_vector[{name!r}]={selected_start[name]} is outside "
                f"the fitted range [{low}, {high}]"
            )
    starting_score, _ = evaluate_vector(
        selected_start,
        blocks,
        model_inputs,
        base,
        stochastic_repeats=stochastic_repeats,
        seed=cfg.random_seed,
        force_deterministic=force_deterministic,
        warmup_weeks=cfg.warmup_weeks,
        overlap_weighted=overlap_weighted,
        prediction_summary=prediction_summary,
        objective_metric=objective_metric,
        sample_weekly_counts=sample_weekly_counts,
    )
    trials.append({"stage": "starting_point", "trial": -1, "objective": starting_score, **selected_start})
    best_vector, best_score = selected_start, starting_score

    candidates: Iterable[dict[str, float]] = (
        _sample_vector(rng, bounds) for _ in range(cfg.n_trials)
    )
    for trial_id, vector in enumerate(candidates):
        score, _ = evaluate_vector(
            vector,
            blocks,
            model_inputs,
            base,
            stochastic_repeats=stochastic_repeats,
            seed=cfg.random_seed,
            force_deterministic=force_deterministic,
            warmup_weeks=cfg.warmup_weeks,
            overlap_weighted=overlap_weighted,
            prediction_summary=prediction_summary,
            objective_metric=objective_metric,
            sample_weekly_counts=sample_weekly_counts,
        )
        trials.append({"stage": "global", "trial": trial_id, "objective": score, **vector})
        if score < best_score:
            best_score, best_vector = score, vector
        if progress and (trial_id + 1) % max(1, cfg.n_trials // 10) == 0:
            print(f"global trial {trial_id + 1}/{cfg.n_trials}: objective={score:.4f}, best={best_score:.4f}")

    assert best_vector is not None
    centre = dict(best_vector)
    for refinement_id in range(cfg.n_refinement_trials):
        vector = {}
        for name, (low, high) in bounds.items():
            if name == "contact_scale":
                value = np.exp(rng.normal(np.log(max(centre[name], low)), 0.45))
            elif name == "seasonal_peak_week":
                width = max(0.15 * abs(centre[name]), 0.02 * (high - low))
                value = low + (rng.normal(centre[name], width) - low) % (high - low)
            else:
                width = max(0.15 * abs(centre[name]), 0.02 * (high - low))
                value = rng.normal(centre[name], width)
            vector[name] = float(np.clip(value, low, high))
        score, _ = evaluate_vector(
            vector,
            blocks,
            model_inputs,
            base,
            stochastic_repeats=stochastic_repeats,
            seed=cfg.random_seed,
            force_deterministic=force_deterministic,
            warmup_weeks=cfg.warmup_weeks,
            overlap_weighted=overlap_weighted,
            prediction_summary=prediction_summary,
            objective_metric=objective_metric,
            sample_weekly_counts=sample_weekly_counts,
        )
        trials.append({"stage": "refinement", "trial": refinement_id, "objective": score, **vector})
        if score < best_score:
            best_score, best_vector = score, vector
            centre = dict(vector)
        if progress and (refinement_id + 1) % max(1, cfg.n_refinement_trials // 5) == 0:
            print(f"refinement {refinement_id + 1}/{cfg.n_refinement_trials}: objective={score:.4f}, best={best_score:.4f}")

    # The coarse sensitivity analysis previously selected both the largest
    # tested amplitude and the earliest tested peak.  When seasonality is part
    # of the fitted vector, refine these two quantities over the widened bounds
    # rather than reporting the old grid edge as an estimate.  All parameters
    # have already participated in the joint random search above; this is a
    # conditional, derivative-free final adjustment of the two seasonal terms.
    seasonal_names = ["seasonal_amplitude", "seasonal_peak_week"]
    if (
        cfg.seasonal_refinement_maxiter > 0
        and all(name in bounds for name in seasonal_names)
    ):
        seasonal_bounds = [bounds[name] for name in seasonal_names]
        seasonal_trial = 0

        def seasonal_objective(values: np.ndarray) -> float:
            nonlocal seasonal_trial, best_score, best_vector
            vector = dict(best_vector)
            for name, value, (low, high) in zip(
                seasonal_names, values, seasonal_bounds
            ):
                vector[name] = float(np.clip(value, low, high))
            score, _ = evaluate_vector(
                vector,
                blocks,
                model_inputs,
                base,
                stochastic_repeats=stochastic_repeats,
                seed=cfg.random_seed,
                force_deterministic=force_deterministic,
                warmup_weeks=cfg.warmup_weeks,
                overlap_weighted=overlap_weighted,
                prediction_summary=prediction_summary,
                objective_metric=objective_metric,
                sample_weekly_counts=sample_weekly_counts,
            )
            trials.append(
                {
                    "stage": "seasonal_optimization",
                    "trial": seasonal_trial,
                    "objective": score,
                    **vector,
                }
            )
            seasonal_trial += 1
            if score < best_score:
                best_score, best_vector = score, vector
            return score

        seasonal_guess = np.asarray(
            [best_vector[name] for name in seasonal_names], dtype=float
        )
        minimize(
            seasonal_objective,
            x0=seasonal_guess,
            method="Powell",
            bounds=seasonal_bounds,
            options={
                "maxiter": int(cfg.seasonal_refinement_maxiter),
                "xtol": 1e-3,
                "ftol": 1e-4,
            },
        )
        if progress:
            print(
                "seasonal optimisation: "
                f"amplitude={best_vector['seasonal_amplitude']:.4g}, "
                f"peak week={best_vector['seasonal_peak_week']:.4g}, "
                f"best={best_score:.4f}"
            )

    # E and I are unobserved initial states. A small random search in a large
    # joint parameter space can easily leave their starting multipliers at 1.0
    # without examining the local two-dimensional surface. Optimise those two
    # multipliers explicitly while holding the other currently-best values
    # fixed. This estimates them conditionally; it does not by itself prove
    # that E(0) and I(0) are separately identifiable from confirmed cases.
    if cfg.initial_state_refinement_maxiter > 0:
        state_names = ["initial_exposed_per_case", "initial_infectious_per_case"]
        state_bounds = [bounds[name] for name in state_names]
        state_trial = 0

        def state_objective(values: np.ndarray) -> float:
            nonlocal state_trial, best_score, best_vector
            vector = dict(best_vector)
            for name, value, (low, high) in zip(state_names, values, state_bounds):
                vector[name] = float(np.clip(value, low, high))
            score, _ = evaluate_vector(
                vector,
                blocks,
                model_inputs,
                base,
                stochastic_repeats=stochastic_repeats,
                seed=cfg.random_seed,
                force_deterministic=force_deterministic,
                warmup_weeks=cfg.warmup_weeks,
                overlap_weighted=overlap_weighted,
                prediction_summary=prediction_summary,
                objective_metric=objective_metric,
                sample_weekly_counts=sample_weekly_counts,
            )
            trials.append(
                {
                    "stage": "initial_state_optimization",
                    "trial": state_trial,
                    "objective": score,
                    **vector,
                }
            )
            state_trial += 1
            if score < best_score:
                best_score, best_vector = score, vector
            return score

        initial_guess = np.asarray([best_vector[name] for name in state_names], dtype=float)
        minimize(
            state_objective,
            x0=initial_guess,
            method="Powell",
            bounds=state_bounds,
            options={
                "maxiter": int(cfg.initial_state_refinement_maxiter),
                "xtol": 1e-3,
                "ftol": 1e-4,
            },
        )
        if progress:
            print(
                "initial-state optimisation: "
                f"E/case={best_vector['initial_exposed_per_case']:.4g}, "
                f"I/case={best_vector['initial_infectious_per_case']:.4g}, "
                f"best={best_score:.4f}"
            )

    _, diagnostics = evaluate_vector(
        best_vector,
        blocks,
        model_inputs,
        base,
        stochastic_repeats=stochastic_repeats,
        seed=cfg.random_seed,
        force_deterministic=force_deterministic,
        warmup_weeks=cfg.warmup_weeks,
        overlap_weighted=overlap_weighted,
        prediction_summary=prediction_summary,
        objective_metric=objective_metric,
        sample_weekly_counts=sample_weekly_counts,
    )
    return best_vector, pd.DataFrame(trials).sort_values("objective").reset_index(drop=True), diagnostics


def fit_simple_grid_parameters(
    blocks: pd.DataFrame,
    inputs: RegionAgeInputs | None = None,
    base_parameters: ModelParameters | None = None,
    parameter_grids: dict[str, Iterable[float]] | None = None,
    passes: int = 2,
    progress: bool = True,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Fit a small parameter subset with a visible deterministic grid search.

    This is the deliberately simple teaching fitter:

    1. Start from the printed Mathsy parameter vector.
    2. Take one parameter at a time.
    3. Try every printed candidate value while holding the others fixed.
    4. Keep the value with the smallest ordinary mean squared error (MSE).
    5. Repeat the cycle so later choices can update earlier ones.

    The fitted curve is the deterministic Mathsy trajectory.  SDE noise and
    weekly count sampling are both disabled.  Consequently this function uses
    no Poisson model, likelihood, log-error term, gradients, random search or
    black-box optimiser.  It is a coordinate grid search and can miss joint
    interactions; that limitation is intentional and should be checked later.
    """

    if passes < 1:
        raise ValueError("passes must be at least 1")
    model_inputs = inputs or load_default_inputs()
    base = base_parameters or default_calibration_parameters()
    grids = SIMPLE_FIT_PARAMETER_GRIDS if parameter_grids is None else parameter_grids
    if not grids:
        raise ValueError("parameter_grids must contain at least one parameter")
    unknown = set(grids).difference(FIT_PARAMETER_BOUNDS)
    if unknown:
        raise ValueError(f"Unknown simple-fit parameters: {sorted(unknown)}")

    selected = calibration_starting_vector(base)
    trials: list[dict[str, object]] = []

    def score_vector(
        vector: dict[str, float],
        *,
        stage: str,
        varied_parameter: str,
        tested_value: float,
    ) -> float:
        score, _ = evaluate_vector(
            vector,
            blocks,
            model_inputs,
            base,
            stochastic_repeats=1,
            seed=0,
            force_deterministic=True,
            warmup_weeks=0,
            overlap_weighted=False,
            prediction_summary="mean",
            objective_metric="mean_squared_error",
            sample_weekly_counts=False,
        )
        trials.append(
            {
                "stage": stage,
                "varied_parameter": varied_parameter,
                "tested_value": float(tested_value),
                "objective_mse": float(score),
                **vector,
            }
        )
        return float(score)

    best_score = score_vector(
        selected,
        stage="starting_point",
        varied_parameter="none",
        tested_value=np.nan,
    )
    for pass_number in range(1, int(passes) + 1):
        for name, raw_values in grids.items():
            low, high = FIT_PARAMETER_BOUNDS[name]
            values = sorted(
                {
                    float(np.clip(float(value), low, high))
                    for value in raw_values
                }
                | {float(selected[name])}
            )
            parameter_best = float(selected[name])
            parameter_best_score = best_score
            for value in values:
                candidate = dict(selected)
                candidate[name] = value
                score = score_vector(
                    candidate,
                    stage=f"pass_{pass_number}",
                    varied_parameter=name,
                    tested_value=value,
                )
                if score < parameter_best_score:
                    parameter_best_score = score
                    parameter_best = value
            selected[name] = parameter_best
            best_score = parameter_best_score
            if progress:
                print(
                    f"pass {pass_number}: {name} = {parameter_best:.6g}; "
                    f"MSE = {best_score:.6g}"
                )

    _, diagnostics = evaluate_vector(
        selected,
        blocks,
        model_inputs,
        base,
        stochastic_repeats=1,
        seed=0,
        force_deterministic=True,
        warmup_weeks=0,
        overlap_weighted=False,
        prediction_summary="mean",
        objective_metric="mean_squared_error",
        sample_weekly_counts=False,
    )
    trial_table = pd.DataFrame(trials).sort_values(
        "objective_mse", kind="stable"
    ).reset_index(drop=True)
    return selected, trial_table, diagnostics


def stochastic_trajectories(
    best_vector: dict[str, float],
    blocks: pd.DataFrame,
    inputs: RegionAgeInputs,
    base_parameters: ModelParameters | None = None,
    n_simulations: int = 100,
    seed: int = 0,
    warmup_weeks: int = 0,
    noise_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate stochastic London paths and per-week percentile summaries.

    ``weekly_cases`` is new reported incidence, while ``active_q`` is the
    corresponding active sick/reported stock. ``noise_scale`` controls the
    Euler-Maruyama compartment diffusion; when omitted, the model default is
    used (currently 0.03).
    """

    base = base_parameters or default_calibration_parameters()
    params = _with_vector(base, best_vector)
    params.noise_scale = base.noise_scale if noise_scale is None else float(noise_scale)
    path_rows = []
    summary_rows = []
    for block_id, block in blocks.groupby("block_id", sort=True):
        historical_age_weights = None
        if set(inputs.age_groups).issubset(set(AGE_GROUPS)):
            historical_age_weights, _, _ = london_age_case_weights(
                inputs,
                forecast_origin=pd.Timestamp(block.iloc[0]["conditioning_date"]),
                prior_strength=0.0,
            )
        observed = block["observed_cases"].to_numpy(dtype=float)
        simulated_cases = []
        simulated_q = []
        for simulation in range(n_simulations):
            values, q_values = _simulate_block(
                inputs,
                params,
                block,
                seed + int(block_id) * 1009 + simulation,
                warmup_weeks=warmup_weeks,
                return_q=True,
                sample_weekly_counts=True,
                age_weights=historical_age_weights,
            )
            simulated_cases.append(values)
            simulated_q.append(q_values)
        simulated_cases = np.asarray(simulated_cases)
        simulated_q = np.asarray(simulated_q)
        for simulation, (values, q_values) in enumerate(zip(simulated_cases, simulated_q), start=1):
            for week, (date, value, q_value, truth) in enumerate(
                zip(block["date"], values, q_values, observed), start=1
            ):
                path_rows.append(
                    {
                        "block_id": int(block_id),
                        "simulation": simulation,
                        "week_in_block": week,
                        "date": date,
                        "weekly_cases": float(value),
                        "active_q": float(q_value),
                        "observed_cases": float(truth),
                    }
                )
        for week, (date, truth) in enumerate(zip(block["date"], observed), start=1):
            summary_rows.append(
                {
                    "block_id": int(block_id),
                    "week_in_block": week,
                    "date": date,
                    "observed_cases": float(truth),
                    "p10_cases": float(np.percentile(simulated_cases[:, week - 1], 10)),
                    "median_cases": float(np.percentile(simulated_cases[:, week - 1], 50)),
                    "p90_cases": float(np.percentile(simulated_cases[:, week - 1], 90)),
                    "p10_active_q": float(np.percentile(simulated_q[:, week - 1], 10)),
                    "median_active_q": float(np.percentile(simulated_q[:, week - 1], 50)),
                    "p90_active_q": float(np.percentile(simulated_q[:, week - 1], 90)),
                }
            )
    return pd.DataFrame(path_rows), pd.DataFrame(summary_rows)


def sensitivity_analysis(
    best_vector: dict[str, float],
    blocks: pd.DataFrame,
    inputs: RegionAgeInputs,
    base_parameters: ModelParameters | None = None,
    stochastic_repeats: int = 1,
    seed: int = 0,
    warmup_weeks: int = 0,
    overlap_weighted: bool = False,
) -> pd.DataFrame:
    """Vary every fitted and fixed parameter one at a time around the fit."""

    base = base_parameters or default_calibration_parameters()
    baseline_vector = dict(best_vector)
    baseline_vector.update({name: getattr(base, name) for name in FIXED_CALIBRATION_PARAMETERS})
    baseline_vector.update({"noise_scale": base.noise_scale, "dt": base.dt})
    baseline, _ = evaluate_vector(
        best_vector,
        blocks,
        inputs,
        base,
        stochastic_repeats=stochastic_repeats,
        seed=seed,
        force_deterministic=stochastic_repeats == 1,
        warmup_weeks=warmup_weeks,
        overlap_weighted=overlap_weighted,
    )
    rows = [{"parameter": "baseline", "value": np.nan, "objective": baseline, "delta_objective": 0.0}]

    for name in SENSITIVITY_PARAMETERS:
        centre = float(baseline_vector[name])
        if name in FIT_PARAMETER_BOUNDS:
            low, high = FIT_PARAMETER_BOUNDS[name]
            values = np.linspace(low, high, 5) if centre <= low + 1e-12 or centre >= high - 1e-12 else [
                max(low, centre * 0.8), max(low, centre * 0.9), centre, min(high, centre * 1.1), min(high, centre * 1.2)
            ]
        elif name == "dt":
            values = [0.01, 0.02, 0.04]
        elif name == "noise_scale":
            values = [0.0, 0.01, 0.02, 0.05, 0.10]
        elif name == "eta":
            values = [centre * 0.5, centre, centre * 1.5]
        elif name == "beta_0":
            values = [centre * 0.5, centre, centre * 1.5]
        else:
            values = [centre * 0.5, centre, centre * 1.5]
        for value in values:
            candidate = dict(best_vector)
            candidate[name] = float(value)
            score, _ = evaluate_vector(
                candidate,
                blocks,
                inputs,
                base,
                stochastic_repeats=stochastic_repeats,
                seed=seed,
                force_deterministic=stochastic_repeats == 1 and name != "noise_scale",
                warmup_weeks=warmup_weeks,
                overlap_weighted=overlap_weighted,
            )
            rows.append(
                {
                    "parameter": name,
                    "value": float(value),
                    "objective": score,
                    "delta_objective": score - baseline,
                }
            )
    result = pd.DataFrame(rows)
    magnitude = result.loc[result["parameter"] != "baseline"].groupby("parameter")["delta_objective"].apply(lambda x: float(np.max(np.abs(x))))
    result["max_abs_delta_by_parameter"] = result["parameter"].map(magnitude)
    result["sensitivity_rank"] = result["max_abs_delta_by_parameter"].rank(method="dense", ascending=False)
    return result
