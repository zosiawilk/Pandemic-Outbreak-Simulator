"""Load and validate surveillance data used by the final London forecast.

This module contains data-shaping code only.  Model calibration and simulation
live in :mod:`outbreak_probability_model.london_calibration` and
:mod:`outbreak_probability_model.model`, respectively.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UKHSA_CASES = REPO_ROOT / "data" / "ukhsa-measles_weekly_cases.csv"


def load_london_observed(
    cases_path: Path | str = DEFAULT_UKHSA_CASES,
    *,
    exclude_reporting_delay: bool = True,
) -> pd.DataFrame:
    """Return consecutive London onset-week counts from a UKHSA export.

    The raw export contains several geographies and metrics.  This function
    selects the London ``casesByOnsetWeek`` series and, by default, removes
    observations that UKHSA marks as incomplete because of reporting delay.
    """

    path = Path(cases_path)
    if not path.exists():
        raise FileNotFoundError(
            f"UKHSA case export not found: {path}. See data/README.md for "
            "the expected file and update procedure."
        )

    data = pd.read_csv(path, parse_dates=["date"])
    required = {
        "date",
        "geography_type",
        "geography",
        "metric",
        "metric_value",
        "in_reporting_delay_period",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"UKHSA case export is missing columns: {sorted(missing)}")

    london = data.loc[
        data["geography_type"].eq("UKHSA Region")
        & data["geography"].eq("London")
        & data["metric"].eq("measles_cases_casesByOnsetWeek")
    ].copy()
    if exclude_reporting_delay:
        london = london.loc[~london["in_reporting_delay_period"].astype(bool)]

    london = (
        london.sort_values("date")
        .rename(columns={"metric_value": "observed_cases"})
        [["date", "observed_cases", "in_reporting_delay_period"]]
        .reset_index(drop=True)
    )
    if london.empty:
        raise ValueError("No London weekly measles observations were found.")
    if london["date"].duplicated().any():
        raise ValueError("London observations contain duplicate weekly dates.")
    if (london["observed_cases"] < 0).any():
        raise ValueError("London observations cannot contain negative cases.")

    gaps = london["date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=7)).all():
        raise ValueError("London observations are not a consecutive weekly series.")
    return london
