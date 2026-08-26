"""Four-week-conditioned stochastic outbreak forecasts for every model region.

This module deliberately reuses the calibrated Mathsy S/H/E/I/Q/D simulator
used by the London outbreak-probability notebook.  The London-fitted biological,
contact and seasonal quantities are transferred to each target region.  The
fitted London seeding quantity is first scaled by relative population and then
introduced only into the target region.  Each region otherwise supplies its own
recent cases, population/protection, density, mobility context, age allocation,
and locally conditioned forecast-origin state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .london_calibration import (
    DEFAULT_LONDON_SEASONAL_POISSON_FITTED_PARAMETERS,
    HistoryConditioningConfig,
    condition_on_recent_history,
    load_london_fitted_parameters,
)
from .model import (
    RegionAgeInputs,
    _run_one_simulation,
    load_default_inputs,
)


@dataclass(frozen=True)
class RegionalForecastResult:
    """Tidy outputs from one independently conditioned regional forecast."""

    region: str
    origin_date: pd.Timestamp
    forecast_start: pd.Timestamp
    current_cases: float
    outbreak_threshold: float
    outbreak_probability: float
    trajectories: pd.DataFrame
    weekly_summary: pd.DataFrame
    conditioning_fit: pd.DataFrame
    conditioning_diagnostics: dict[str, object]


@dataclass(frozen=True)
class HistoricalRegionalAudit:
    """Leakage-safe forecasts and scores for historical regional origins."""

    forecasts: dict[tuple[str, pd.Timestamp], RegionalForecastResult]
    weekly_comparison: pd.DataFrame
    origin_metrics: pd.DataFrame
    regional_metrics: pd.DataFrame


def load_synthetic_regional_history(
    input_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-age and age-specific synthetic weekly regional targets."""

    frame = pd.read_csv(
        Path(input_dir) / "synthetic_region_age_weekly_cases.csv",
        parse_dates=["date"],
    )
    age = (
        frame.groupby(["date", "region", "age_group"], as_index=False)
        .synthetic_cases.sum()
        .sort_values(["region", "date", "age_group"])
    )
    all_age = (
        age.groupby(["date", "region"], as_index=False)
        .synthetic_cases.sum()
        .rename(columns={"synthetic_cases": "observed_cases"})
        .sort_values(["region", "date"])
    )
    return all_age, age


def population_scaled_thresholds(
    inputs: RegionAgeInputs,
    *,
    london_threshold: float = 15.0,
    minimum_threshold: float = 3.0,
) -> pd.Series:
    """Scale thresholds relative to London while keeping every other one lower.

    London is the explicit reference and always receives ``london_threshold``.
    Other regions are scaled by their population relative to London and capped
    at the London threshold.  The minimum prevents tiny thresholds dominated
    by single imported cases.
    """

    populations = pd.Series(inputs.population.sum(axis=1), index=inputs.regions)
    london_population = float(populations.loc["London"])
    raw = np.ceil(london_threshold * populations / london_population)
    non_london_cap = max(float(london_threshold), float(minimum_threshold))
    thresholds = raw.clip(lower=minimum_threshold, upper=non_london_cap)
    thresholds.loc["London"] = float(london_threshold)
    return thresholds.rename("outbreak_threshold")


def case_burden_scaled_thresholds(
    inputs: RegionAgeInputs,
    cases: pd.DataFrame,
    *,
    origin_date: pd.Timestamp | str | None = None,
    london_threshold: float = 15.0,
    minimum_threshold: float = 3.0,
    case_burden_weight: float = 0.0,
) -> pd.Series:
    """Blend population and past case burden relative to London.

    ``case_burden_weight=0`` is pure population scaling; ``1`` is pure
    historical-case scaling.  The primary model uses population-only scaling.
    A positive value is retained only for explicit sensitivity analysis.  When
    an origin is supplied, later observations are excluded to keep historical
    audits leakage-safe.
    """

    if not 0 <= case_burden_weight <= 1:
        raise ValueError("case_burden_weight must lie in [0, 1]")
    if float(case_burden_weight) == 0.0:
        return population_scaled_thresholds(
            inputs,
            london_threshold=london_threshold,
            minimum_threshold=minimum_threshold,
        )
    available = cases.copy()
    if origin_date is not None:
        available = available.loc[available.date.le(pd.Timestamp(origin_date))]
    means = available.groupby("region").observed_cases.mean().reindex(inputs.regions)
    means = means.fillna(0.0).clip(lower=1e-6)
    populations = pd.Series(inputs.population.sum(axis=1), index=inputs.regions)
    population_ratio = populations / float(populations.loc["London"])
    burden_ratio = means / float(means.loc["London"])
    weight = float(case_burden_weight)
    raw = london_threshold * population_ratio ** (1.0 - weight) * burden_ratio**weight
    non_london_cap = max(float(london_threshold), float(minimum_threshold))
    thresholds = np.ceil(raw).clip(lower=minimum_threshold, upper=non_london_cap)
    thresholds.loc["London"] = float(london_threshold)
    return thresholds.rename("outbreak_threshold")


def _regional_age_weights(
    age_history: pd.DataFrame,
    region: str,
    age_groups: list[str],
    origin_date: pd.Timestamp,
    history_weeks: int,
    inputs: RegionAgeInputs,
) -> np.ndarray:
    """Estimate the origin age mix from only the selected trailing history."""

    selected = age_history.loc[
        age_history.region.eq(region) & age_history.date.le(origin_date)
    ].copy()
    dates = selected.date.drop_duplicates().sort_values().tail(history_weeks)
    recent = (
        selected.loc[selected.date.isin(dates)]
        .groupby("age_group").synthetic_cases.sum()
        .reindex(age_groups, fill_value=0.0)
        .to_numpy(float)
    )
    if recent.sum() <= 0:
        region_index = inputs.regions.index(region)
        recent = inputs.population[region_index] * (
            1.0 - inputs.protected_fraction[region_index]
        )
    recent = recent + 1e-6
    return recent / recent.sum()


def population_scaled_seed_from_london(
    inputs: RegionAgeInputs,
    region: str,
    london_seed_infections_per_week: float,
) -> float:
    """Convert the London-fitted weekly seed count to a target-region count."""

    if region not in inputs.regions or "London" not in inputs.regions:
        raise ValueError("Both the target region and London must be modelled")
    london_seed = float(london_seed_infections_per_week)
    if not np.isfinite(london_seed) or london_seed < 0:
        raise ValueError("London weekly seeding must be finite and non-negative")
    populations = np.asarray(inputs.population, dtype=float).sum(axis=1)
    london_population = float(populations[inputs.regions.index("London")])
    if london_population <= 0:
        raise ValueError("London population must be positive")
    return london_seed * float(populations[inputs.regions.index(region)]) / london_population


def forecast_region(
    region: str,
    cases: pd.DataFrame,
    age_history: pd.DataFrame,
    *,
    inputs: RegionAgeInputs,
    outbreak_threshold: float,
    fitted_parameters_path: Path | str = DEFAULT_LONDON_SEASONAL_POISSON_FITTED_PARAMETERS,
    history_weeks: int = 4,
    horizon_weeks: int = 6,
    n_simulations: int = 500,
    random_seed: int = 42,
    sample_weekly_counts: bool = True,
    forecast_dt: float = 0.2,
    conditioning_maxiter: int = 30,
    origin_date: pd.Timestamp | str | None = None,
    national_trend_weight: float = 1.0,
    trend_damping: float = 0.8,
) -> RegionalForecastResult:
    """Condition on one region's last four weeks and simulate six-week paths."""

    if region not in inputs.regions:
        raise ValueError(f"Unknown region {region!r}")
    regional_cases = (
        cases.loc[cases.region.eq(region), ["date", "observed_cases"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if origin_date is not None:
        requested_origin = pd.Timestamp(origin_date)
        regional_cases = regional_cases.loc[
            regional_cases.date.le(requested_origin)
        ].reset_index(drop=True)
        if regional_cases.empty or pd.Timestamp(regional_cases.iloc[-1].date) != requested_origin:
            raise ValueError(f"{region} has no observation at origin {requested_origin.date()}")
    if len(regional_cases) < history_weeks:
        raise ValueError(f"{region} has fewer than {history_weeks} observations")

    params, _ = load_london_fitted_parameters(fitted_parameters_path)
    regional_seed = population_scaled_seed_from_london(
        inputs, region, params.seed_infections_per_week
    )
    params = replace(params, seed_infections_per_week=regional_seed)
    # The London calibration uses dt=0.02 days.  A nine-region interactive
    # forecast would otherwise require hours.  dt=0.2 retains 35 Euler steps
    # per week and the same equations; it is an explicit numerical
    # approximation that should be sensitivity-checked against 0.1/0.05.
    params.dt = float(forecast_dt)
    origin_date = pd.Timestamp(regional_cases.iloc[-1].date)
    age_weights = _regional_age_weights(
        age_history, region, inputs.age_groups, origin_date, history_weeks, inputs
    )
    config = HistoryConditioningConfig(
        history_weeks=history_weeks,
        maxiter=int(conditioning_maxiter),
    )
    conditioned = condition_on_recent_history(
        regional_cases,
        inputs,
        params,
        config,
        region=region,
        age_weights=age_weights,
        random_seed=random_seed,
    )
    forecast_inputs = replace(
        inputs,
        contact_matrix=(
            inputs.contact_matrix
            * float(params.contact_scale)
            * conditioned.transmission_multiplier
        ),
    )
    region_index = inputs.regions.index(region)
    forecast_start = origin_date + pd.Timedelta(days=7)
    forecast_dates = pd.date_range(forecast_start, periods=horizon_weeks, freq="7D")
    rng = np.random.default_rng(random_seed)
    paths = np.zeros((n_simulations, horizon_weeks), dtype=float)

    for simulation in range(n_simulations):
        weekly_imports = rng.poisson(
            max(float(params.seed_infections_per_week), 0.0),
            size=horizon_weeks,
        ).astype(float)
        simulated = _run_one_simulation(
            inputs=forecast_inputs,
            params=params,
            initial_reported_sick=conditioned.state.Q,
            initial_exposed=conditioned.state.E,
            initial_infectious=conditioned.state.I,
            horizon_weeks=horizon_weeks,
            random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            sample_weekly_counts=sample_weekly_counts,
            weekly_seed_infections=weekly_imports,
            seed_region=region,
            initial_state=conditioned.state,
            simulation_start_date=forecast_start,
        )
        paths[simulation] = simulated[:, region_index, :].sum(axis=1)

    # Synthetic regional timing was generated from the national weekly curve.
    # Assimilate its recent past-only momentum into the observation layer so a
    # London-calibrated latent-state decay cannot mechanically force every
    # region downward.  This changes neither biological rates nor compartments.
    national_history = (
        cases.loc[cases.date.le(origin_date)]
        .groupby("date", as_index=False).observed_cases.sum()
        .sort_values("date")
        .tail(history_weeks)
    )
    x_history = np.arange(len(national_history), dtype=float)
    weights = np.arange(1, len(national_history) + 1, dtype=float)
    national_log_slope = float(
        np.polyfit(
            x_history,
            np.log1p(national_history.observed_cases.to_numpy(float)),
            1,
            w=weights,
        )[0]
    )
    model_median = np.r_[float(regional_cases.iloc[-1].observed_cases), np.median(paths, axis=0)]
    model_log_slope = float(
        np.polyfit(
            np.arange(len(model_median), dtype=float),
            np.log1p(model_median),
            1,
        )[0]
    )
    damped_steps = np.cumsum(
        float(trend_damping) ** np.arange(1, horizon_weeks + 1, dtype=float)
    )
    correction = np.exp(
        float(national_trend_weight)
        * (national_log_slope - model_log_slope)
        * damped_steps
    )
    correction = np.clip(correction, 0.5, 2.0)
    scaled = np.maximum(paths * correction.reshape(1, -1), 0.0)
    # Stochastic rounding retains integer weekly counts without adding a
    # second Poisson variance layer.
    lower = np.floor(scaled)
    paths = lower + (
        rng.random(size=scaled.shape) < (scaled - lower)
    )

    peaks = paths.max(axis=1)
    event = peaks > float(outbreak_threshold)
    cumulative = np.cumsum(paths, axis=1)
    trajectory_rows = []
    for simulation in range(n_simulations):
        for week in range(horizon_weeks):
            trajectory_rows.append(
                {
                    "region": region,
                    "simulation": simulation + 1,
                    "week": week + 1,
                    "date": forecast_dates[week],
                    "weekly_cases": float(paths[simulation, week]),
                    "cumulative_cases": float(cumulative[simulation, week]),
                    "peak_weekly_cases": float(peaks[simulation]),
                    "crosses_outbreak_threshold": bool(event[simulation]),
                }
            )
    trajectories = pd.DataFrame(trajectory_rows)
    weekly_summary = pd.DataFrame(
        {
            "region": region,
            "week": np.arange(1, horizon_weeks + 1),
            "date": forecast_dates,
            "mean_cases": paths.mean(axis=0),
            "median_cases": np.median(paths, axis=0),
            "p10_cases": np.percentile(paths, 10, axis=0),
            "p90_cases": np.percentile(paths, 90, axis=0),
        }
    )
    return RegionalForecastResult(
        region=region,
        origin_date=origin_date,
        forecast_start=forecast_start,
        current_cases=float(regional_cases.iloc[-1].observed_cases),
        outbreak_threshold=float(outbreak_threshold),
        outbreak_probability=float(event.mean()),
        trajectories=trajectories,
        weekly_summary=weekly_summary,
        conditioning_fit=conditioned.history_fit.assign(region=region),
        conditioning_diagnostics={
            "region": region,
            "conditioning_start": conditioned.conditioning_start,
            "origin_date": conditioned.forecast_origin,
            "recent_transmission_multiplier": conditioned.transmission_multiplier,
            "initial_exposed_per_case": conditioned.initial_exposed_per_case,
            "initial_infectious_per_case": conditioned.initial_infectious_per_case,
            "origin_exposed_total": conditioned.origin_exposed_total,
            "origin_infectious_total": conditioned.origin_infectious_total,
            "origin_sick_total": conditioned.origin_sick_total,
            "conditioning_objective": conditioned.objective,
            "optimizer_success": conditioned.optimizer_success,
            "national_recent_log_slope": national_log_slope,
            "raw_model_log_slope": model_log_slope,
            "national_trend_weight": float(national_trend_weight),
            "trend_correction_week_1": float(correction[0]),
            "trend_correction_week_6": float(correction[-1]),
        },
    )


def forecast_all_regions(
    *,
    input_dir: Path | str,
    inputs: RegionAgeInputs | None = None,
    fitted_parameters_path: Path | str = DEFAULT_LONDON_SEASONAL_POISSON_FITTED_PARAMETERS,
    london_threshold: float = 15.0,
    minimum_threshold: float = 3.0,
    history_weeks: int = 4,
    horizon_weeks: int = 6,
    n_simulations: int = 500,
    random_seed: int = 42,
    forecast_dt: float = 0.2,
    conditioning_maxiter: int = 30,
    case_burden_weight: float = 0.0,
    exclude_regions: tuple[str, ...] = ("London",),
    national_trend_weight: float = 1.0,
) -> dict[str, RegionalForecastResult]:
    """Run the same four-week-conditioned forecast for every model region."""

    model_inputs = inputs or load_default_inputs(input_dir=input_dir)
    cases, age_history = load_synthetic_regional_history(input_dir)
    thresholds = case_burden_scaled_thresholds(
        model_inputs,
        cases,
        london_threshold=london_threshold,
        minimum_threshold=minimum_threshold,
        case_burden_weight=case_burden_weight,
    )
    forecast_regions = [
        region for region in model_inputs.regions if region not in set(exclude_regions)
    ]
    return {
        region: forecast_region(
            region,
            cases,
            age_history,
            inputs=model_inputs,
            outbreak_threshold=float(thresholds.loc[region]),
            fitted_parameters_path=fitted_parameters_path,
            history_weeks=history_weeks,
            horizon_weeks=horizon_weeks,
            n_simulations=n_simulations,
            random_seed=random_seed + index * 1009,
            forecast_dt=forecast_dt,
            conditioning_maxiter=conditioning_maxiter,
            national_trend_weight=national_trend_weight,
        )
        for index, region in enumerate(forecast_regions)
    }


def historical_regional_audit(
    *,
    input_dir: Path | str,
    inputs: RegionAgeInputs | None = None,
    fitted_parameters_path: Path | str = DEFAULT_LONDON_SEASONAL_POISSON_FITTED_PARAMETERS,
    n_origins: int = 4,
    origin_dates: list[pd.Timestamp | str] | None = None,
    london_threshold: float = 15.0,
    minimum_threshold: float = 3.0,
    history_weeks: int = 4,
    horizon_weeks: int = 6,
    n_simulations: int = 100,
    random_seed: int = 20260818,
    forecast_dt: float = 0.2,
    conditioning_maxiter: int = 20,
    case_burden_weight: float = 0.0,
    exclude_regions: tuple[str, ...] = ("London",),
    national_trend_weight: float = 1.0,
) -> HistoricalRegionalAudit:
    """Compare forecasts with six genuinely later synthetic observations.

    Origins are shared across regions.  Each fit is given only observations on
    or before its origin; weeks 1--6 are retrieved only after simulation for
    scoring and plotting.
    """

    model_inputs = inputs or load_default_inputs(input_dir=input_dir)
    cases, age_history = load_synthetic_regional_history(input_dir)
    all_dates = pd.DatetimeIndex(sorted(cases.date.unique()))
    eligible = all_dates[
        history_weeks - 1 : len(all_dates) - horizon_weeks
    ]
    if origin_dates is None:
        if n_origins < 1:
            raise ValueError("n_origins must be positive")
        positions = np.linspace(0, len(eligible) - 1, n_origins, dtype=int)
        selected_origins = list(eligible[np.unique(positions)])
    else:
        selected_origins = [pd.Timestamp(value) for value in origin_dates]
        invalid = [value for value in selected_origins if value not in eligible]
        if invalid:
            raise ValueError(f"Origins lack four prior or six future weeks: {invalid}")

    forecasts: dict[tuple[str, pd.Timestamp], RegionalForecastResult] = {}
    weekly_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    audit_regions = [
        region for region in model_inputs.regions if region not in set(exclude_regions)
    ]
    for region_index, region in enumerate(audit_regions):
        region_series = cases.loc[cases.region.eq(region)].set_index("date")["observed_cases"]
        for origin_index, origin in enumerate(selected_origins):
            # Recalculate from past-only burden at every historical origin.
            thresholds = case_burden_scaled_thresholds(
                model_inputs,
                cases,
                origin_date=origin,
                london_threshold=london_threshold,
                minimum_threshold=minimum_threshold,
                case_burden_weight=case_burden_weight,
            )
            result = forecast_region(
                region,
                cases,
                age_history,
                inputs=model_inputs,
                outbreak_threshold=float(thresholds.loc[region]),
                fitted_parameters_path=fitted_parameters_path,
                history_weeks=history_weeks,
                horizon_weeks=horizon_weeks,
                n_simulations=n_simulations,
                random_seed=random_seed + region_index * 1009 + origin_index * 7919,
                forecast_dt=forecast_dt,
                conditioning_maxiter=conditioning_maxiter,
                origin_date=origin,
                national_trend_weight=national_trend_weight,
            )
            forecasts[(region, pd.Timestamp(origin))] = result
            future_dates = pd.date_range(
                pd.Timestamp(origin) + pd.Timedelta(days=7),
                periods=horizon_weeks,
                freq="7D",
            )
            truth = region_series.reindex(future_dates).to_numpy(float)
            summary = result.weekly_summary.reset_index(drop=True)
            for week in range(horizon_weeks):
                weekly_rows.append(
                    {
                        "region": region,
                        "origin_date": pd.Timestamp(origin),
                        "week": week + 1,
                        "date": future_dates[week],
                        "observed_cases": truth[week],
                        "median_cases": float(summary.loc[week, "median_cases"]),
                        "p10_cases": float(summary.loc[week, "p10_cases"]),
                        "p90_cases": float(summary.loc[week, "p90_cases"]),
                        "covered_80": bool(
                            summary.loc[week, "p10_cases"]
                            <= truth[week]
                            <= summary.loc[week, "p90_cases"]
                        ),
                    }
                )
            flags = result.trajectories.groupby("simulation")[
                "crosses_outbreak_threshold"
            ].first()
            observed_event = bool(np.any(truth > float(thresholds.loc[region])))
            probability = float(flags.mean())
            metric_rows.append(
                {
                    "region": region,
                    "origin_date": pd.Timestamp(origin),
                    "outbreak_threshold": float(thresholds.loc[region]),
                    "observed_outbreak": observed_event,
                    "predicted_probability": probability,
                    "brier_score": (probability - float(observed_event)) ** 2,
                    "mae": float(np.mean(np.abs(summary.median_cases.to_numpy() - truth))),
                    "interval_80_coverage": float(
                        np.mean(
                            (summary.p10_cases.to_numpy() <= truth)
                            & (truth <= summary.p90_cases.to_numpy())
                        )
                    ),
                }
            )
    weekly = pd.DataFrame(weekly_rows)
    origin_metrics = pd.DataFrame(metric_rows)
    regional_metrics = (
        origin_metrics.groupby("region", as_index=False)
        .agg(
            origins=("origin_date", "size"),
            mean_brier_score=("brier_score", "mean"),
            mean_absolute_error=("mae", "mean"),
            interval_80_coverage=("interval_80_coverage", "mean"),
            observed_outbreak_rate=("observed_outbreak", "mean"),
            mean_predicted_probability=("predicted_probability", "mean"),
        )
    )
    return HistoricalRegionalAudit(
        forecasts=forecasts,
        weekly_comparison=weekly,
        origin_metrics=origin_metrics,
        regional_metrics=regional_metrics,
    )


def find_below_to_above_origins(
    *,
    input_dir: Path | str,
    inputs: RegionAgeInputs | None = None,
    london_threshold: float = 15.0,
    minimum_threshold: float = 3.0,
    case_burden_weight: float = 0.0,
    history_weeks: int = 4,
    horizon_weeks: int = 6,
    exclude_regions: tuple[str, ...] = ("London",),
) -> pd.DataFrame:
    """Find every past origin below threshold whose holdout later crosses it.

    This is a cheap descriptive scan, not a model fit. Future observations are
    used only to label interesting audit episodes; they are never supplied to
    ``forecast_region`` when those episodes are subsequently forecast.
    """

    model_inputs = inputs or load_default_inputs(input_dir=input_dir)
    cases, _ = load_synthetic_regional_history(input_dir)
    dates = pd.DatetimeIndex(sorted(cases.date.unique()))
    eligible = dates[history_weeks - 1 : len(dates) - horizon_weeks]
    regions = [r for r in model_inputs.regions if r not in set(exclude_regions)]
    result_columns = [
        "region",
        "origin_date",
        "origin_cases",
        "outbreak_threshold",
        "first_crossing_week",
        "maximum_next_6_weeks",
        "future_cases",
    ]
    rows: list[dict[str, object]] = []
    for origin in eligible:
        thresholds = case_burden_scaled_thresholds(
            model_inputs,
            cases,
            origin_date=origin,
            london_threshold=london_threshold,
            minimum_threshold=minimum_threshold,
            case_burden_weight=case_burden_weight,
        )
        future_dates = pd.date_range(
            origin + pd.Timedelta(days=7), periods=horizon_weeks, freq="7D"
        )
        for region in regions:
            series = cases.loc[cases.region.eq(region)].set_index("date").observed_cases
            current = float(series.loc[origin])
            future = series.reindex(future_dates).to_numpy(float)
            threshold = float(thresholds.loc[region])
            crossing = np.flatnonzero(future > threshold)
            if current <= threshold and crossing.size:
                rows.append(
                    {
                        "region": region,
                        "origin_date": pd.Timestamp(origin),
                        "origin_cases": current,
                        "outbreak_threshold": threshold,
                        "first_crossing_week": int(crossing[0] + 1),
                        "maximum_next_6_weeks": float(future.max()),
                        "future_cases": future.tolist(),
                    }
                )
    # No qualifying crossing is a valid audit result. Supplying the schema
    # keeps the empty table sortable and usable by downstream notebook cells.
    return pd.DataFrame.from_records(rows, columns=result_columns).sort_values(
        ["region", "first_crossing_week", "origin_date"]
    ).reset_index(drop=True)
