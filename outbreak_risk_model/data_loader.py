"""Load region-age inputs used by network preprocessing and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "measles_local_age"
    / "inputs"
)


@dataclass(frozen=True)
class RegionAgeSnapshot:
    """Current model inputs for one region-age group."""

    region: str
    age_group: str
    current_cases: float
    recent_weekly_cases: tuple[float, ...]
    population: float | None
    protected_fraction: float | None
    susceptible_fraction: float | None
    region_risk_multiplier: float


@dataclass(frozen=True)
class LocalAgeModelInputs:
    """Matrix inputs needed by ``LocalAgeMeaslesSim``."""

    regions: list[str]
    age_groups: list[str]
    population: np.ndarray
    protected_fraction: np.ndarray
    region_risk_multiplier: np.ndarray
    latest_cases: np.ndarray
    latest_case_date: pd.Timestamp


def load_region_age_snapshot(
    region: str,
    age_group: str,
    *,
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    history_weeks: int = 6,
) -> RegionAgeSnapshot:
    """Load recent synthetic cases and protection data for a region-age group."""

    input_path = Path(input_dir)
    cases = pd.read_csv(input_path / "synthetic_region_age_weekly_cases.csv")
    profile = pd.read_csv(input_path / "region_age_population_protection.csv")

    selected = cases[
        (cases["region"].astype(str) == region)
        & (cases["age_group"].astype(str) == age_group)
    ].copy()
    if selected.empty:
        raise ValueError(f"No weekly cases found for region={region!r}, age_group={age_group!r}")

    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected.sort_values("date").tail(history_weeks)
    recent_cases = tuple(float(x) for x in selected["synthetic_cases"].to_numpy())
    current_cases = recent_cases[-1]

    profile_row = profile[
        (profile["region"].astype(str) == region)
        & (profile["age_group"].astype(str) == age_group)
    ]
    if profile_row.empty:
        return RegionAgeSnapshot(
            region=region,
            age_group=age_group,
            current_cases=current_cases,
            recent_weekly_cases=recent_cases,
            population=None,
            protected_fraction=None,
            susceptible_fraction=None,
            region_risk_multiplier=1.0,
        )

    row = profile_row.iloc[0]
    protected = float(row["protected_fraction"])
    return RegionAgeSnapshot(
        region=region,
        age_group=age_group,
        current_cases=current_cases,
        recent_weekly_cases=recent_cases,
        population=float(row["population"]),
        protected_fraction=protected,
        susceptible_fraction=max(0.0, 1.0 - protected),
        region_risk_multiplier=float(row.get("region_risk_multiplier", 1.0)),
    )


def load_model_inputs(
    *,
    input_dir: Path | str = DEFAULT_INPUT_DIR,
) -> LocalAgeModelInputs:
    """Load matrix-shaped inputs for the existing local-age simulator."""

    input_path = Path(input_dir)
    profile = pd.read_csv(input_path / "region_age_population_protection.csv")
    cases = pd.read_csv(input_path / "synthetic_region_age_weekly_cases.csv")
    cases["date"] = pd.to_datetime(cases["date"])

    regions = list(profile["region"].drop_duplicates())
    age_groups = list(profile["age_group"].drop_duplicates())
    latest_date = cases["date"].max()
    latest_cases = cases[cases["date"] == latest_date]

    population = _pivot(profile, regions, age_groups, "population")
    protected_fraction = _pivot(profile, regions, age_groups, "protected_fraction")
    latest_case_matrix = _pivot(
        latest_cases,
        regions,
        age_groups,
        "synthetic_cases",
        fill_value=0.0,
    )
    region_risk = (
        profile.groupby("region")["region_risk_multiplier"]
        .first()
        .reindex(regions)
        .fillna(1.0)
        .to_numpy(dtype=float)
    )

    return LocalAgeModelInputs(
        regions=regions,
        age_groups=age_groups,
        population=population,
        protected_fraction=protected_fraction,
        region_risk_multiplier=region_risk,
        latest_cases=latest_case_matrix,
        latest_case_date=latest_date,
    )


def _pivot(
    data: pd.DataFrame,
    regions: list[str],
    age_groups: list[str],
    value: str,
    *,
    fill_value: float | None = None,
) -> np.ndarray:
    table = data.pivot(index="region", columns="age_group", values=value)
    table = table.reindex(index=regions, columns=age_groups)
    if fill_value is not None:
        table = table.fillna(fill_value)
    return table.to_numpy(dtype=float)
