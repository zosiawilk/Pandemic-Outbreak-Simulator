"""Step-by-step outbreak probability model for region-age outbreak risk.


1. Split each region-age population into S, H, E, I, Q, D compartments.
2. Compute separate infectious and sick prevalence: I / L and Q / L.
3. Use a normal contact matrix for I and a reduced sick contact matrix for Q.
   Sick people can also have reduced or zero commuting.
4. Add an independent Wiener increment to every compartment trajectory.
5. Repeat the forecast many times and estimate operational exceedance
   probability as the fraction whose weekly cases are strictly above a threshold.

"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "experiments" / "measles_local_age" / "inputs"
DEFAULT_PARAMETER_SUMMARY = (
    REPO_ROOT
    / "experiments"
    / "measles_local_age"
    / "output"
    / "local_age_parameter_summary.csv"
)
DEFAULT_MOBILITY_MATRIX = (
    REPO_ROOT
    / "outbreak_risk_model"
    / "graph_outputs"
    / "odwp_england_region_mobility_adjacency.csv"
)
DEFAULT_STUDENT_MOBILITY_MATRIX = (
    REPO_ROOT
    / "outbreak_risk_model"
    / "graph_outputs"
    / "odst_england_region_student_mobility_adjacency.csv"
)
DEFAULT_CONTACT_MATRIX_PATH = (
    REPO_ROOT
    / "experiments"
    / "measles_age"
    / "inputs"
    / "age_contact_matrix_reconnect_figure_normalized.csv"
)
DEFAULT_POPULATION_DENSITY_PATH = (
    DEFAULT_INPUT_DIR / "region_population_density_ons_2024.csv"
)


AGE_GROUPS = [
    "under_1",
    "1_to_4",
    "5_to_10",
    "11_to_14",
    "15_to_24",
    "25_to_34",
    "35_and_over",
]


# Rows are receiving age groups and columns are infectious/source age groups.
# Use the repository's Reconnect-derived matrix rather than an invented
# seven-by-seven teaching matrix. It is normalised again by its mean on load.
_contact_frame = pd.read_csv(DEFAULT_CONTACT_MATRIX_PATH, index_col=0)
DEFAULT_CONTACT_MATRIX = _contact_frame.loc[AGE_GROUPS, AGE_GROUPS].to_numpy(dtype=float)

CONTACT_MATRIX_AGE_GROUPS = AGE_GROUPS


@dataclass
class ModelParameters:
    """All scalar settings used by the simulator.

    The names are matched to the maths in the PDF where possible.
    """

    # Overall contact/interactions multiplier.  This is dimensionless and
    # scales the force of infection; it is not a probability.
    mu: float = 4.34
    # Global scale applied to the empirical age-contact matrix.  ``mu`` is
    # held fixed in the London pipeline, so this is the one fitted overall
    # contact/transmission-strength adjustment.
    contact_scale: float = 1.0
    # Transmission coefficient for susceptible people in S.  In the current
    # implementation it multiplies the age/region contact pressure per day.
    sigma: float = 0.090
    # Transmission coefficient for protected/recovered people in H.  It is
    # smaller than sigma because H has reduced susceptibility.
    delta: float = 0.0009
    # E -> I progression rate per day.  1/gamma is the mean incubation period
    # (7 days for the default value).
    gamma: float = 1.0 / 7.0
    # I -> Q progression rate per day.  Q is the reported/hospitalised sick
    # stock, so 1/psi is the mean time in I before becoming sick.
    psi: float = 1.0 / 3.0
    # Recovery rate from Q per day.  1/phi is the mean Q duration if no other
    # removal process acts first.
    phi: float = 0.12
    # Disease-death rate from Q per day.
    nu: float = 0.003
    # Natural mortality rate per day, applied to S, H, E, I and Q.
    eta: float = 0.000027
    # Recruitment/birth rate per day.  Births are proportional to the living
    # population in the current implementation.
    beta_0: float = 0.0000466
    # Fraction of the force of infection coming from local contacts.  1.0
    # means entirely local mixing; 0.0 means entirely commuting pressure.
    local_mixing: float = 0.81
    # Relative contact level of people in Q compared with normal contacts.
    # For example, 0.2 means Q contributes 20% of the normal contact pressure.
    sick_contact_multiplier: float = 0.2
    # Relative commuting contribution of Q.  This affects only the Q mobility
    # term; local Q contacts are still controlled by sick_contact_multiplier.
    sick_mobility_multiplier: float = 0.0
    # Exogenous infections seeded per week.  A caller can allocate this total
    # to one explicit target region; otherwise the backwards-compatible public
    # API distributes it over the complete model population.
    seed_infections_per_week: float = 3
    # Hidden-state multipliers used only when constructing a forecast origin:
    # E(0) = this value * known week-0 cases, and likewise for I(0).
    initial_exposed_per_case: float = 0.0
    initial_infectious_per_case: float = 0.0
    # Fraction of I -> Q sick onsets observed as confirmed cases. This is an
    # observation parameter only: every sick onset enters Q whether reported.
    reporting_rate: float = 0.9
    # Dimensionless diffusion scale for the independent compartment noises.
    noise_scale: float = 0.03
    # Optional multiplier used only by the S/H noise sensitivity experiment.
    # The primary model is unscaled (1.0), giving S and H the same relative
    # diffusion form as the other living compartments.
    sh_noise_multiplier: float = 1.0
    # Euler time step in days (0.02 days is approximately 28.8 minutes).
    dt: float = 0.02
    # Optional annual multiplier on transmission pressure.  Zero preserves the
    # historical, non-seasonal model exactly.  ``seasonal_peak_week`` is the
    # number of weeks after 2024-01-01 at which the multiplier is maximal.
    seasonal_amplitude: float = 0.0
    seasonal_peak_week: float = 20.0
    seasonal_period_weeks: float = 52.18


@dataclass(frozen=True)
class RegionAgeInputs:
    """Matrix-shaped data used to initialise the model."""

    regions: list[str]
    age_groups: list[str]
    population: np.ndarray
    protected_fraction: np.ndarray
    region_risk_multiplier: np.ndarray
    population_density_per_km2: np.ndarray
    citiness_contact_multiplier: np.ndarray
    latest_weekly_cases: np.ndarray
    latest_case_date: pd.Timestamp
    mobility_matrix: np.ndarray
    mobility_layer_weights: dict[str, float]
    contact_matrix: np.ndarray
    data_sources: dict[str, str]


@dataclass(frozen=True)
class CompartmentState:
    """Complete model state at one forecast origin.

    Keeping the six arrays together lets a history-conditioning calculation
    hand its terminal latent state directly to the future simulator.  This is
    preferable to reconstructing E, I and Q from only the latest case count.
    """

    S: np.ndarray
    H: np.ndarray
    E: np.ndarray
    I: np.ndarray
    Q: np.ndarray
    D: np.ndarray


@dataclass(frozen=True)
class OutbreakResult:
    """Outputs for the chosen region-age forecast."""

    region: str
    age_group: str
    outbreak_threshold: float
    outbreak_metric: str
    horizon_weeks: int
    n_simulations: int
    outbreak_probability: float
    current_cases: float
    expected_future_cases: float
    expected_peak_weekly_cases: float
    weekly_summary: pd.DataFrame
    trajectories: pd.DataFrame
    inputs: RegionAgeInputs
    parameters: ModelParameters


def available_region_age_groups(input_dir: Path | str = DEFAULT_INPUT_DIR) -> pd.DataFrame:
    """List the region-age combinations available in the input CSV files."""

    profile = pd.read_csv(Path(input_dir) / "region_age_population_protection.csv")
    return (
        profile[["region", "age_group"]]
        .drop_duplicates()
        .sort_values(["region", "age_group"])
        .reset_index(drop=True)
    )


def load_default_inputs(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    mobility_matrix_path: Path | str | None = DEFAULT_MOBILITY_MATRIX,
    student_mobility_matrix_path: Path | str | None = DEFAULT_STUDENT_MOBILITY_MATRIX,
    work_mobility_weight: float = 1.0,
    student_mobility_weight: float = 1.0,
    contact_matrix: np.ndarray = DEFAULT_CONTACT_MATRIX,
    use_outcome_derived_region_risk: bool = False,
    population_density_path: Path | str = DEFAULT_POPULATION_DENSITY_PATH,
    population_density_period: str = "2024-06-30",
    citiness_reference_density_per_km2: float = 5782.0,
    citiness_exponent: float = 0.20,
    minimum_citiness_multiplier: float = 0.45,
    maximum_citiness_multiplier: float = 1.0,
) -> RegionAgeInputs:
    """Load population, protection, latest cases, risk, contacts, and mobility.

    Data files used:
    - ``region_age_population_protection.csv``: population, protected fraction,
      and regional risk multiplier.
    - ``synthetic_region_age_weekly_cases.csv``: latest synthetic weekly
      reported cases, used as an initial sick/reported-state approximation.
    - ``odwp_england_region_mobility_adjacency.csv``: optional work commuting matrix.
    - ``odst_england_region_student_mobility_adjacency.csv``: optional student-origin matrix.
    """

    input_path = Path(input_dir)
    profile = pd.read_csv(input_path / "region_age_population_protection.csv")
    weekly_cases = pd.read_csv(input_path / "synthetic_region_age_weekly_cases.csv")
    weekly_cases["date"] = pd.to_datetime(weekly_cases["date"])

    regions = list(profile["region"].drop_duplicates())
    age_groups = [age for age in AGE_GROUPS if age in set(profile["age_group"])]
    latest_date = weekly_cases["date"].max()
    latest_week = weekly_cases.loc[weekly_cases["date"].eq(latest_date)]

    population = _pivot(profile, regions, age_groups, "population")
    protected = _pivot(profile, regions, age_groups, "protected_fraction")
    latest_cases = _pivot(latest_week, regions, age_groups, "synthetic_cases", fill_value=0.0)
    loaded_region_risk = (
        profile.groupby("region")["region_risk_multiplier"]
        .first()
        .reindex(regions)
        .fillna(1.0)
        .to_numpy(dtype=float)
    )
    # The CSV multiplier was constructed from annual measles case outcomes.
    # Using it by default in historical fitting would leak later outcome data
    # into earlier forecast origins and would also double-use the London case
    # signal already used for calibration.  The canonical pipeline therefore
    # uses the neutral value rho_r=1.  The empirical vector remains available
    # only as an explicitly requested scenario.
    region_risk = (
        loaded_region_risk
        if use_outcome_derived_region_risk
        else np.ones_like(loaded_region_risk, dtype=float)
    )

    density = _load_population_density(
        population_density_path,
        regions,
        period=population_density_period,
    )
    if not np.isfinite(density).all() or (density <= 0).any():
        raise ValueError("Every model region needs a positive finite population density")
    if citiness_exponent < 0:
        raise ValueError("citiness_exponent must be non-negative")
    if not np.isfinite(citiness_reference_density_per_km2) or citiness_reference_density_per_km2 <= 0:
        raise ValueError("citiness reference density must be positive and finite")
    if not 0 < minimum_citiness_multiplier <= maximum_citiness_multiplier:
        raise ValueError("citiness multiplier bounds must be positive and increasing")
    citiness = np.clip(
        (density / float(citiness_reference_density_per_km2)) ** float(citiness_exponent),
        float(minimum_citiness_multiplier),
        float(maximum_citiness_multiplier),
    )

    mobility, mobility_weights = _load_combined_mobility_matrix(
        regions=regions,
        work_mobility_matrix_path=mobility_matrix_path,
        student_mobility_matrix_path=student_mobility_matrix_path,
        work_mobility_weight=work_mobility_weight,
        student_mobility_weight=student_mobility_weight,
    )
    using_default_contacts = contact_matrix is DEFAULT_CONTACT_MATRIX
    normalised_contacts = _align_contact_matrix(contact_matrix, age_groups)
    normalised_contacts = normalised_contacts / normalised_contacts.mean()

    return RegionAgeInputs(
        regions=regions,
        age_groups=age_groups,
        population=population,
        protected_fraction=protected,
        region_risk_multiplier=region_risk,
        population_density_per_km2=density,
        citiness_contact_multiplier=citiness,
        latest_weekly_cases=latest_cases,
        latest_case_date=latest_date,
        mobility_matrix=mobility,
        mobility_layer_weights=mobility_weights,
        contact_matrix=normalised_contacts,
        data_sources={
            "population_and_protection": str(input_path / "region_age_population_protection.csv"),
            "latest_cases": str(input_path / "synthetic_region_age_weekly_cases.csv"),
            "work_mobility_matrix": str(mobility_matrix_path) if mobility_matrix_path else "not used",
            "student_mobility_matrix": (
                str(student_mobility_matrix_path)
                if student_mobility_matrix_path
                else "not used"
            ),
            "mobility_matrix": "weighted combination of enabled mobility layers",
            "region_risk_multiplier": (
                str(input_path / "region_age_population_protection.csv")
                if use_outcome_derived_region_risk
                else "neutral rho_r=1 (outcome-derived multiplier disabled)"
            ),
            "population_density": (
                f"{population_density_path}; selected period={population_density_period}"
            ),
            "citiness_contact_multiplier": (
                f"(density / {citiness_reference_density_per_km2:g}) ** "
                f"{citiness_exponent:g}, clipped to "
                f"[{minimum_citiness_multiplier:g}, {maximum_citiness_multiplier:g}]"
            ),
            "contact_matrix": (
                str(DEFAULT_CONTACT_MATRIX_PATH)
                if using_default_contacts
                else "custom matrix supplied to load_default_inputs"
            ),
            "fitted_parameters": str(DEFAULT_PARAMETER_SUMMARY),
        },
    )


def _load_population_density(
    path: Path | str,
    regions: list[str],
    *,
    period: str = "2024-06-30",
) -> np.ndarray:
    """Load density from either a processed model table or raw ONS ELS data.

    Processed input uses ``region,population_density_per_km2``. Raw ONS
    Explore Local Statistics extracts use ``areacd,areanm,period,value``.
    Alignment is deliberately by the model's geography names; future upper-
    tier input profiles should use the exact ONS ``areanm`` spelling.
    """

    table = pd.read_csv(path)
    if {"region", "population_density_per_km2"}.issubset(table.columns):
        selected = table[["region", "population_density_per_km2"]].copy()
        name_column = "region"
        value_column = "population_density_per_km2"
    elif {"areacd", "areanm", "period", "value"}.issubset(table.columns):
        selected = table.loc[
            table["period"].astype(str).eq(str(period)),
            ["areacd", "areanm", "value"],
        ].copy()
        name_column = "areanm"
        value_column = "value"
    else:
        raise ValueError(
            "population-density input must be either a processed "
            "region/population_density_per_km2 table or an ONS "
            "areacd/areanm/period/value extract"
        )
    if selected.empty:
        raise ValueError(f"population-density input has no rows for period {period}")
    if selected[name_column].duplicated().any():
        duplicates = sorted(selected.loc[selected[name_column].duplicated(), name_column].unique())
        raise ValueError(f"population-density input has duplicate areas: {duplicates[:5]}")
    aligned = pd.to_numeric(
        selected.set_index(name_column)[value_column].reindex(regions), errors="coerce"
    )
    missing = [region for region, value in aligned.items() if not np.isfinite(value)]
    if missing:
        raise ValueError(
            "population-density input does not match these model geographies: "
            + ", ".join(missing[:10])
        )
    return aligned.to_numpy(dtype=float)


def input_sources_table(inputs: RegionAgeInputs) -> pd.DataFrame:
    """Return a notebook-friendly table explaining every loaded input source."""

    rows = [
        {
            "input": "population",
            "symbol": "N_r,a",
            "where_taken_from": inputs.data_sources["population_and_protection"],
            "how_used": "Total people in each region-age group.",
        },
        {
            "input": "protected_fraction",
            "symbol": "p_r,a",
            "where_taken_from": inputs.data_sources["population_and_protection"],
            "how_used": "Initial protected/immune people: H(0) = N * p.",
        },
        {
            "input": "latest_weekly_cases",
            "symbol": "Q(0)",
            "where_taken_from": inputs.data_sources["latest_cases"],
            "how_used": (
                "Latest synthetic weekly reported cases approximate the initial "
                "currently sick/reported state."
            ),
        },
        {
            "input": "region_risk_multiplier",
            "symbol": "rho_r",
            "where_taken_from": inputs.data_sources["region_risk_multiplier"],
            "how_used": (
                "Multiplies infection pressure for each region; the canonical "
                "pipeline uses 1 to avoid outcome leakage."
            ),
        },
        {
            "input": "population_density_per_km2",
            "symbol": "d_r",
            "where_taken_from": inputs.data_sources["population_density"],
            "how_used": "Observed density used to construct the citiness multiplier.",
        },
        {
            "input": "citiness_contact_multiplier",
            "symbol": "c_r",
            "where_taken_from": inputs.data_sources["citiness_contact_multiplier"],
            "how_used": "Scales local contact pressure; London is the reference c_r=1.",
        },
        {
            "input": "mobility_matrix",
            "symbol": "W_q,r",
            "where_taken_from": inputs.data_sources["mobility_matrix"],
            "how_used": "Weighted combination of mobility layers used for between-region pressure.",
        },
        {
            "input": "work_mobility_matrix",
            "symbol": "W_work",
            "where_taken_from": inputs.data_sources["work_mobility_matrix"],
            "how_used": f"ODWP commuting layer weight = {inputs.mobility_layer_weights.get('work', 0):.3g}.",
        },
        {
            "input": "student_mobility_matrix",
            "symbol": "W_student",
            "where_taken_from": inputs.data_sources["student_mobility_matrix"],
            "how_used": f"ODST student-origin layer weight = {inputs.mobility_layer_weights.get('student', 0):.3g}.",
        },
        {
            "input": "contact_matrix",
            "symbol": "C_a,b",
            "where_taken_from": inputs.data_sources["contact_matrix"],
            "how_used": "Age-group mixing weights used in the force of infection.",
        },
        {
            "input": "model_parameters",
            "symbol": "mu, sigma, delta, gamma, psi, ...",
            "where_taken_from": inputs.data_sources["fitted_parameters"],
            "how_used": "Fitted medians override readable defaults when available.",
        },
    ]
    return pd.DataFrame(rows)


def selected_initial_state(
    inputs: RegionAgeInputs,
    region: str,
    age_group: str,
    current_cases: float | None = None,
) -> pd.DataFrame:
    """Show the initial S, H, E, I, Q, D values for one region-age pair."""

    region_index = inputs.regions.index(region)
    age_index = inputs.age_groups.index(age_group)
    population = float(inputs.population[region_index, age_index])
    protected = float(inputs.protected_fraction[region_index, age_index])
    reported_sick = float(inputs.latest_weekly_cases[region_index, age_index])
    if current_cases is not None:
        reported_sick = max(float(current_cases), 0.0)

    protected_count = population * protected
    reported_sick = min(reported_sick, max(population - protected_count, 0.0))
    susceptible_count = max(population - protected_count - reported_sick, 0.0)
    return pd.DataFrame(
        [
            {
                "region": region,
                "age_group": age_group,
                "S_susceptible": susceptible_count,
                "H_protected": protected_count,
                "E_exposed_incubating": 0.0,
                "I_infectious": 0.0,
                "Q_sick": reported_sick,
                "D_deaths": 0.0,
                "note": (
                    "Reported cases initialise Q. E and I start at zero unless "
                    "you add a nowcast for hidden infections."
                ),
            }
        ]
    )


def parameters_table(parameters: ModelParameters) -> pd.DataFrame:
    """Return model parameters as a two-column table for notebook display."""

    return pd.DataFrame(
        [{"parameter": name, "value": value} for name, value in asdict(parameters).items()]
    )


def load_fitted_parameters(
    parameter_summary_path: Path | str = DEFAULT_PARAMETER_SUMMARY,
) -> ModelParameters:
    """Start with defaults, then load fitted medians where the CSV provides them."""

    params = ModelParameters()
    path = Path(parameter_summary_path)
    if not path.exists():
        return params

    summary = pd.read_csv(path).set_index("parameter")
    parameter_name_map = {
        "mu": "mu",
        "seed_infections_per_week": "seed_infections_per_week",
        "reporting_rate": "reporting_rate",
        "local_mixing": "local_mixing",
        "incubation_rate": "gamma",
        "sick_rate": "psi",
        "sick_contact_multiplier": "sick_contact_multiplier",
        "sick_mobility_multiplier": "sick_mobility_multiplier",
    }
    for csv_name, attribute_name in parameter_name_map.items():
        if csv_name in summary.index and "median" in summary.columns:
            setattr(params, attribute_name, float(summary.loc[csv_name, "median"]))
    return params


def forecast_outbreak_probability(
    region: str,
    age_group: str,
    outbreak_threshold: float = 10.0,
    horizon_weeks: int = 6,
    n_simulations: int = 200,
    random_seed: int | None = 42,
    inputs: RegionAgeInputs | None = None,
    parameters: ModelParameters | None = None,
    current_cases: float | None = None,
    sample_weekly_counts: bool = True,
) -> OutbreakResult:
    """Run many stochastic forecasts and estimate threshold-exceedance probability.

    ``outbreak_probability`` is:

    number of simulated trajectories where any weekly case count > threshold
    -----------------------------------------------------------------------
    total number of simulated trajectories

    Weekly integer surveillance counts are Poisson-sampled by default. Set
    ``sample_weekly_counts=False`` to retain the continuous expected number
    reported from each realised SDE path.
    """

    model_inputs = inputs or load_default_inputs()
    params = parameters or load_fitted_parameters()

    if region not in model_inputs.regions:
        raise ValueError(f"Unknown region {region!r}. Try one from model_inputs.regions.")
    if age_group not in model_inputs.age_groups:
        raise ValueError(f"Unknown age_group {age_group!r}. Try one from model_inputs.age_groups.")

    region_index = model_inputs.regions.index(region)
    age_index = model_inputs.age_groups.index(age_group)

    initial_reported_sick = model_inputs.latest_weekly_cases.copy()
    if current_cases is not None:
        initial_reported_sick[region_index, age_index] = max(float(current_cases), 0.0)
    available_unprotected = np.maximum(
        model_inputs.population
        - model_inputs.population * model_inputs.protected_fraction,
        0.0,
    )
    initial_reported_sick = np.clip(initial_reported_sick, 0.0, available_unprotected)

    all_weekly_cases = []
    rng = np.random.default_rng(random_seed)
    for _ in range(int(n_simulations)):
        simulation_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        weekly_cases = _run_one_simulation(
            inputs=model_inputs,
            params=params,
            initial_reported_sick=initial_reported_sick,
            horizon_weeks=horizon_weeks,
            random_seed=simulation_seed,
            sample_weekly_counts=sample_weekly_counts,
        )
        all_weekly_cases.append(weekly_cases)

    runs = np.asarray(all_weekly_cases, dtype=float)
    target_runs = runs[:, :, region_index, age_index]
    peak_weekly_cases = target_runs.max(axis=1)
    outbreak_flags = exceeds_outbreak_threshold(peak_weekly_cases, outbreak_threshold)

    trajectories = pd.DataFrame(
        {
            "simulation": np.repeat(np.arange(1, int(n_simulations) + 1), int(horizon_weeks)),
            "week": np.tile(np.arange(1, int(horizon_weeks) + 1), int(n_simulations)),
            "weekly_cases": target_runs.reshape(-1),
        }
    )
    trajectories["cumulative_cases"] = trajectories.groupby("simulation")[
        "weekly_cases"
    ].cumsum()
    trajectories["peak_weekly_cases"] = trajectories.groupby("simulation")[
        "weekly_cases"
    ].transform("max")
    trajectories["crosses_outbreak_threshold"] = np.repeat(outbreak_flags, int(horizon_weeks))

    weekly_summary = pd.DataFrame(
        {
            "week": np.arange(1, int(horizon_weeks) + 1),
            "mean_cases": target_runs.mean(axis=0),
            "median_cases": np.percentile(target_runs, 50, axis=0),
            "p10_cases": np.percentile(target_runs, 10, axis=0),
            "p90_cases": np.percentile(target_runs, 90, axis=0),
        }
    )

    return OutbreakResult(
        region=region,
        age_group=age_group,
        outbreak_threshold=float(outbreak_threshold),
        outbreak_metric="peak_weekly_cases_strictly_greater_than_threshold",
        horizon_weeks=int(horizon_weeks),
        n_simulations=int(n_simulations),
        outbreak_probability=float(outbreak_flags.mean()),
        current_cases=float(initial_reported_sick[region_index, age_index]),
        expected_future_cases=float(target_runs.mean(axis=0).sum()),
        expected_peak_weekly_cases=float(peak_weekly_cases.mean()),
        weekly_summary=weekly_summary,
        trajectories=trajectories,
        inputs=model_inputs,
        parameters=params,
    )


def _run_one_simulation(
    inputs: RegionAgeInputs,
    params: ModelParameters,
    initial_reported_sick: np.ndarray,
    horizon_weeks: int,
    random_seed: int,
    initial_exposed: np.ndarray | None = None,
    initial_infectious: np.ndarray | None = None,
    return_weekly_q: bool = False,
    sample_weekly_counts: bool = False,
    weekly_seed_infections: np.ndarray | None = None,
    seed_region: str | None = None,
    initial_state: CompartmentState | None = None,
    return_final_state: bool = False,
    return_weekly_latent: bool = False,
    simulation_start_date: str | pd.Timestamp | None = None,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, CompartmentState]
    | tuple[np.ndarray, np.ndarray, CompartmentState]
):
    """Run one stochastic trajectory and return weekly reported cases.

    When ``return_weekly_q`` is true, also return the active sick/reported
    stock at the end of each simulated week.  The default return value is
    unchanged so the outbreak-probability API continues to receive weekly
    incidence only.
    """

    if seed_region is not None and seed_region not in inputs.regions:
        raise ValueError(f"Unknown seed_region {seed_region!r}")
    seed_region_index = (
        None if seed_region is None else inputs.regions.index(seed_region)
    )

    # Calendar time is required only when seasonality is active.  Keeping the
    # default amplitude at zero makes all existing callers backward compatible.
    if params.seasonal_amplitude != 0.0 and simulation_start_date is None:
        raise ValueError(
            "simulation_start_date is required when seasonal_amplitude is non-zero"
        )
    start_date = (
        None if simulation_start_date is None else pd.Timestamp(simulation_start_date)
    )

    # Create the random-number generator for this trajectory.
    rng = np.random.default_rng(random_seed)

    if initial_state is None:
        # H starts with the protected population.
        H = inputs.population * inputs.protected_fraction
        # Put the supplied reported cases into Q, without exceeding capacity.
        Q = np.minimum(initial_reported_sick, np.maximum(inputs.population - H, 0.0))

        # Use supplied hidden E values, or start E at zero.
        requested_exposed = (
            np.zeros_like(Q)
            if initial_exposed is None
            else np.asarray(initial_exposed, dtype=float)
        )
        # Use supplied hidden I values, or start I at zero.
        requested_infectious = (
            np.zeros_like(Q)
            if initial_infectious is None
            else np.asarray(initial_infectious, dtype=float)
        )
        # Do not allow negative compartment sizes.
        E = np.maximum(requested_exposed, 0.0)
        I = np.maximum(requested_infectious, 0.0)

        # Calculate the number of unprotected people available for E, I and Q.
        living_capacity = np.maximum(inputs.population - H, 0.0)
        total_hidden = E + I
        # Find a factor that keeps E + I + Q within that available population.
        hidden_scale = np.minimum(
            np.divide(living_capacity - Q, np.maximum(total_hidden, 1e-12)),
            1.0,
        )
        # Never scale compartments by a negative factor.
        hidden_scale = np.maximum(hidden_scale, 0.0)
        # Apply the capacity correction to the hidden compartments.
        E *= hidden_scale
        I *= hidden_scale
        # Everyone not in H, E, I or Q starts as susceptible.
        S = np.maximum(inputs.population - H - E - I - Q, 0.0)
        # Disease deaths start at zero for this forecast.
        D = np.zeros_like(S)
    else:
        expected_shape = inputs.population.shape
        arrays = []
        for name in ("S", "H", "E", "I", "Q", "D"):
            value = np.asarray(getattr(initial_state, name), dtype=float)
            if value.shape != expected_shape:
                raise ValueError(
                    f"initial_state.{name} has shape {value.shape}; expected {expected_shape}"
                )
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"initial_state.{name} must be finite and non-negative")
            arrays.append(value.copy())
        S, H, E, I, Q, D = arrays

    # Convert the requested horizon from weeks to days.
    days = int(horizon_weeks) * 7
    # Convert the requested time step into an integer number of steps per day,
    # then use the exact reciprocal. This guarantees that a labelled day is
    # exactly one day even when a sensitivity scenario supplies a dt that does
    # not divide one perfectly.
    if not np.isfinite(params.dt) or params.dt <= 0:
        raise ValueError("dt must be a finite positive number of days")
    steps_per_day = max(1, int(round(1.0 / params.dt)))
    effective_params = ModelParameters(**asdict(params))
    effective_params.dt = 1.0 / steps_per_day
    # Store each day's newly reported cases.
    daily_reported_cases = []
    # Store latent/infectious stocks at the end of each week when requested.
    weekly_e = []
    weekly_i = []
    weekly_q = []

    seed_schedule = None
    if weekly_seed_infections is not None:
        seed_schedule = np.asarray(weekly_seed_infections, dtype=float)
        if seed_schedule.shape != (int(horizon_weeks),):
            raise ValueError("weekly_seed_infections must have one value per forecast week")
        if (seed_schedule < 0).any():
            raise ValueError("weekly_seed_infections cannot be negative")

    # Advance the model one day at a time.
    for _day in range(days):
        season = seasonal_multiplier(
            None if start_date is None else start_date + pd.Timedelta(days=_day),
            effective_params,
        )
        # Accumulate all within-day reported cases here.
        reported_today = np.zeros_like(S)
        # Use several small Euler-Maruyama steps to cover one day.
        for _step in range(steps_per_day):
            # Update every compartment by one internal time step.
            S, H, E, I, Q, D, reported = _step_compartments(
                S=S,
                H=H,
                E=E,
                I=I,
                Q=Q,
                D=D,
                inputs=inputs,
                params=effective_params,
                rng=rng,
                seed_infections_per_week_override=(
                    None if seed_schedule is None else seed_schedule[_day // 7]
                ),
                seed_region_index=seed_region_index,
                seasonal_scale=season,
            )
            # Add this small step's reported cases to today's total.
            reported_today += reported
        # Save the completed day's reported cases.
        daily_reported_cases.append(reported_today)
        # At the end of each seven-day period, save the active Q stock.
        if (_day + 1) % 7 == 0:
            if return_weekly_latent:
                weekly_e.append(E.copy())
                weekly_i.append(I.copy())
            weekly_q.append(Q.copy())

    # Convert the daily list into a NumPy array.
    daily = np.asarray(daily_reported_cases)
    # Group seven daily arrays into one weekly incidence array.
    weekly_reported = daily.reshape(int(horizon_weeks), 7, *daily.shape[1:]).sum(axis=1)
    # The SDE produces continuous expected reported incidence. When integer
    # surveillance counts are requested, sample once from each weekly total.
    if sample_weekly_counts:
        weekly_reported = rng.poisson(np.maximum(weekly_reported, 0.0)).astype(float)
    final_state = CompartmentState(S=S, H=H, E=E, I=I, Q=Q, D=D)
    if return_weekly_latent:
        latent = {
            "E": np.asarray(weekly_e),
            "I": np.asarray(weekly_i),
            "Q": np.asarray(weekly_q),
        }
        if return_final_state:
            return weekly_reported, latent, final_state
        return weekly_reported, latent
    # Return weekly incidence and optional diagnostic/state outputs.
    if return_weekly_q and return_final_state:
        return weekly_reported, np.asarray(weekly_q), final_state
    if return_weekly_q:
        return weekly_reported, np.asarray(weekly_q)
    if return_final_state:
        return weekly_reported, final_state
    # The normal output is weekly reported incidence only.
    return weekly_reported


def _step_compartments(
    S: np.ndarray,
    H: np.ndarray,
    E: np.ndarray,
    I: np.ndarray,
    Q: np.ndarray,
    D: np.ndarray,
    inputs: RegionAgeInputs,
    params: ModelParameters,
    rng: np.random.Generator,
    seed_infections_per_week_override: float | None = None,
    seed_region_index: int | None = None,
    seasonal_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance all compartments by one Euler--Maruyama time step.

    The drift contains the mean biological flows. Independent Wiener
    increments are then added to S, H, E, I, Q and D, following the stochastic
    structure of the baseline simulator. These noises are deliberately not
    paired across transitions, so their sum need not preserve total population
    exactly. Calibration obtains the expected path by setting ``noise_scale``
    to zero.
    """

    dt = params.dt
    if not np.isfinite(seasonal_scale) or seasonal_scale <= 0:
        raise ValueError("seasonal_scale must be finite and positive")
    force = float(seasonal_scale) * _force_of_infection(S, H, E, I, Q, inputs, params)

    weekly_seed = (
        params.seed_infections_per_week
        if seed_infections_per_week_override is None
        else float(seed_infections_per_week_override)
    )
    if not np.isfinite(weekly_seed) or weekly_seed < 0:
        raise ValueError("seed_infections_per_week must be finite and non-negative")
    seed_per_day = weekly_seed / 7.0
    seed_weights = inputs.population * inputs.region_risk_multiplier.reshape(-1, 1)
    if seed_region_index is not None:
        if not 0 <= int(seed_region_index) < len(inputs.regions):
            raise ValueError("seed_region_index is outside the modelled regions")
        target_weights = np.zeros_like(seed_weights)
        target_weights[int(seed_region_index)] = seed_weights[int(seed_region_index)]
        seed_weights = target_weights
    seed_weights = seed_weights / max(seed_weights.sum(), 1.0)
    seeded = seed_per_day * seed_weights

    infect_susceptible = params.sigma * force * S
    infect_protected = params.delta * force * H
    become_infectious = params.gamma * E
    become_sick = params.psi * I
    # Reporting is an observation process, not a biological transition. All
    # I -> Q onsets enter Q; only the selected fraction is observed as a case.
    reporting_fraction = np.clip(params.reporting_rate, 0.0, 1.0)
    recover = params.phi * Q
    disease_deaths = params.nu * Q

    living = S + H + E + I + Q
    drift_s = params.beta_0 * living - infect_susceptible - seeded - params.eta * S
    drift_h = recover - infect_protected - params.eta * H
    drift_e = (
        infect_susceptible + infect_protected + seeded
        - become_infectious - params.eta * E
    )
    drift_i = become_infectious - become_sick - params.eta * I
    drift_q = become_sick - recover - disease_deaths - params.eta * Q
    drift_d = disease_deaths

    noise_s = np.zeros_like(S)
    noise_h = np.zeros_like(H)
    noise_e = np.zeros_like(E)
    noise_i = np.zeros_like(I)
    noise_q = np.zeros_like(Q)
    noise_d = np.zeros_like(D)
    if params.noise_scale > 0:
        if not np.isfinite(params.noise_scale):
            raise ValueError("noise_scale must be finite and non-negative")
        sqrt_dt = np.sqrt(dt)
        omega = float(params.noise_scale)
        if (
            not np.isfinite(params.sh_noise_multiplier)
            or params.sh_noise_multiplier < 0
        ):
            raise ValueError("sh_noise_multiplier must be finite and non-negative")
        # Independent Wiener increments for every region-age compartment.
        # c_SH attenuates only the two much larger background stocks.
        c_sh = float(params.sh_noise_multiplier)
        noise_s = c_sh * omega * S * rng.normal(0.0, sqrt_dt, size=S.shape)
        noise_h = c_sh * omega * H * rng.normal(0.0, sqrt_dt, size=H.shape)
        noise_e = omega * E * rng.normal(0.0, sqrt_dt, size=E.shape)
        noise_i = omega * I * rng.normal(0.0, sqrt_dt, size=I.shape)
        noise_q = omega * Q * rng.normal(0.0, sqrt_dt, size=Q.shape)
        # The baseline simulator scaled death noise by the stock currently at
        # risk of disease death (Q), rather than by cumulative deaths D.
        noise_d = omega * Q * rng.normal(0.0, sqrt_dt, size=D.shape)
    elif params.noise_scale < 0 or not np.isfinite(params.noise_scale):
        raise ValueError("noise_scale must be finite and non-negative")

    S_next = S + drift_s * dt + noise_s
    H_next = H + drift_h * dt + noise_h
    E_next = E + drift_e * dt + noise_e
    I_next = I + drift_i * dt + noise_i
    Q_next = Q + drift_q * dt + noise_q
    D_next = D + drift_d * dt + noise_d
    reported_cases = reporting_fraction * become_sick * dt

    # Independent Gaussian increments can occasionally push a small stock
    # below zero, so each epidemiological state is truncated at zero.
    S_next, H_next, E_next, I_next, Q_next, D_next = (
        np.maximum(value, 0.0)
        for value in (S_next, H_next, E_next, I_next, Q_next, D_next)
    )
    return S_next, H_next, E_next, I_next, Q_next, D_next, reported_cases


def exceeds_outbreak_threshold(values: np.ndarray | float, threshold: float) -> np.ndarray:
    """Return whether weekly cases strictly exceed the operational threshold.

    A threshold of 10 means 11 or more integer reported cases in at least one
    forecast week. Continuous expected flows must likewise be greater than 10.
    """

    return np.asarray(values) > float(threshold)


def seasonal_multiplier(
    date: str | pd.Timestamp | None,
    params: ModelParameters,
    epoch: str | pd.Timestamp = "2024-01-01",
) -> float:
    """Return the annual transmission multiplier for a calendar date.

    The epoch only defines phase coordinates.  The cosine repeats every
    ``seasonal_period_weeks``, so it does not restrict the model to 2024.
    A zero amplitude returns one without requiring a date, preserving the
    pre-seasonality model and its existing call sites.
    """

    amplitude = float(params.seasonal_amplitude)
    period = float(params.seasonal_period_weeks)
    peak = float(params.seasonal_peak_week)
    if not np.isfinite(amplitude) or not 0.0 <= amplitude < 1.0:
        raise ValueError("seasonal_amplitude must be finite and in [0, 1)")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("seasonal_period_weeks must be finite and positive")
    if not np.isfinite(peak):
        raise ValueError("seasonal_peak_week must be finite")
    if amplitude == 0.0:
        return 1.0
    if date is None:
        raise ValueError("date is required when seasonal_amplitude is non-zero")
    elapsed_weeks = (pd.Timestamp(date) - pd.Timestamp(epoch)).total_seconds() / (
        7.0 * 24.0 * 60.0 * 60.0
    )
    return float(
        1.0
        + amplitude
        * np.cos(2.0 * np.pi * (elapsed_weeks - peak) / period)
    )


def _force_of_infection(
    S: np.ndarray,
    H: np.ndarray,
    E: np.ndarray,
    I: np.ndarray,
    Q: np.ndarray,
    inputs: RegionAgeInputs,
    params: ModelParameters,
) -> np.ndarray:
    """Compute lambda_r,a(t), the region-age force of infection."""

    living_population = np.maximum(S + H + E + I + Q, 1.0)
    infectious_prevalence = I / living_population
    sick_prevalence = Q / living_population
    normal_contact_matrix = inputs.contact_matrix
    sick_contact_matrix = _sick_contact_matrix(inputs.contact_matrix, params)

    local_pressure = (
        infectious_prevalence @ normal_contact_matrix.T
        + sick_prevalence @ sick_contact_matrix.T
    )
    sick_mobility_multiplier = float(np.clip(params.sick_mobility_multiplier, 0.0, 1.0))
    incoming_infectious_prevalence = inputs.mobility_matrix.T @ infectious_prevalence
    incoming_sick_prevalence = (
        sick_mobility_multiplier * (inputs.mobility_matrix.T @ sick_prevalence)
    )
    commuting_pressure = (
        incoming_infectious_prevalence @ normal_contact_matrix.T
        + incoming_sick_prevalence @ sick_contact_matrix.T
    )

    mixing = np.clip(params.local_mixing, 0.0, 1.0)
    # Density changes how strongly a given age-contact pattern is realised in
    # local space. London is the calibration reference (multiplier 1); less
    # urban areas retain non-zero transmission through the bounded multiplier
    # and through incoming mobility pressure.
    combined_pressure = (
        mixing
        * inputs.citiness_contact_multiplier.reshape(-1, 1)
        * local_pressure
        + (1.0 - mixing) * commuting_pressure
    )
    return params.mu * inputs.region_risk_multiplier.reshape(-1, 1) * combined_pressure


def _sick_contact_matrix(contact_matrix: np.ndarray, params: ModelParameters) -> np.ndarray:
    """Return the contact matrix used by sick people in Q.

    Sick people are treated as biologically infectious, but they have fewer
    contacts. A multiplier of 0.2 means Q uses 20% of the normal contacts.
    """

    multiplier = float(np.clip(params.sick_contact_multiplier, 0.0, 1.0))
    return multiplier * contact_matrix


def _pivot(
    data: pd.DataFrame,
    regions: list[str],
    age_groups: list[str],
    value: str,
    fill_value: float | None = None,
) -> np.ndarray:
    table = data.pivot(index="region", columns="age_group", values=value)
    table = table.reindex(index=regions, columns=age_groups)
    if fill_value is not None:
        table = table.fillna(fill_value)
    return table.to_numpy(dtype=float)


def _load_mobility_matrix(path: Path | str | None, regions: list[str]) -> np.ndarray:
    """Load receiving-region mobility weights W_q,r.

    Rows identify source/residence region q and columns identify receiving
    region r.  Columns, rather than rows, are normalised so ``W.T @ prevalence``
    is a convex average for every receiver.  This preserves the essential
    invariant that spatial mixing cannot change a prevalence field that is
    identical in every region.
    """

    if path is None or not Path(path).exists():
        return np.eye(len(regions), dtype=float)

    mobility = pd.read_csv(path, index_col=0)
    mobility = mobility.reindex(index=regions, columns=regions).fillna(0.0)

    matrix = mobility.to_numpy(dtype=float)
    # Local within-region pressure is already represented separately. Remove
    # diagonal residence-to-same-region flows so the commuting layer means
    # genuinely external pressure and local_mixing retains a clear meaning.
    np.fill_diagonal(matrix, 0.0)
    column_sums = matrix.sum(axis=0, keepdims=True)
    matrix = np.divide(
        matrix,
        column_sums,
        out=np.zeros_like(matrix),
        where=column_sums > 0,
    )

    # If a region has no incoming external mobility, use its own prevalence.
    empty_columns = np.where(column_sums.reshape(-1) <= 0)[0]
    matrix[empty_columns, empty_columns] = 1.0
    return matrix


def _load_combined_mobility_matrix(
    *,
    regions: list[str],
    work_mobility_matrix_path: Path | str | None,
    student_mobility_matrix_path: Path | str | None,
    work_mobility_weight: float,
    student_mobility_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Load and combine enabled mobility layers into receiving-region weights."""

    layers = []
    weights: dict[str, float] = {}

    if work_mobility_matrix_path is not None and work_mobility_weight > 0:
        layers.append(
            (
                float(work_mobility_weight),
                _load_mobility_matrix(work_mobility_matrix_path, regions),
            )
        )
        weights["work"] = float(work_mobility_weight)

    if student_mobility_matrix_path is not None and student_mobility_weight > 0:
        layers.append(
            (
                float(student_mobility_weight),
                _load_mobility_matrix(student_mobility_matrix_path, regions),
            )
        )
        weights["student"] = float(student_mobility_weight)

    if not layers:
        return np.eye(len(regions), dtype=float), {"identity": 1.0}

    combined = np.zeros((len(regions), len(regions)), dtype=float)
    for weight, matrix in layers:
        combined += weight * matrix
    column_sums = combined.sum(axis=0, keepdims=True)
    combined = np.divide(
        combined,
        column_sums,
        out=np.zeros_like(combined),
        where=column_sums > 0,
    )
    empty_columns = np.where(column_sums.reshape(-1) <= 0)[0]
    combined[empty_columns, empty_columns] = 1.0
    return combined, weights


def _align_contact_matrix(contact_matrix: np.ndarray, age_groups: list[str]) -> np.ndarray:
    """Subset/reorder the default contact matrix to match loaded age groups."""

    matrix = np.asarray(contact_matrix, dtype=float)
    expected_size = len(CONTACT_MATRIX_AGE_GROUPS)
    if matrix.shape != (expected_size, expected_size):
        if matrix.shape == (len(age_groups), len(age_groups)):
            return matrix
        raise ValueError(
            "contact_matrix must either match the loaded age groups or the "
            f"default {expected_size}x{expected_size} age-group layout."
        )

    indices = [CONTACT_MATRIX_AGE_GROUPS.index(age) for age in age_groups]
    return matrix[np.ix_(indices, indices)]
