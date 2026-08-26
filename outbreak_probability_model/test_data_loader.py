from pathlib import Path

import pandas as pd
import pytest

from outbreak_probability_model.data_loader import load_london_observed


def _write_export(path: Path, dates: list[str], delay: list[bool]) -> Path:
    pd.DataFrame(
        {
            "date": dates,
            "geography_type": "UKHSA Region",
            "geography": "London",
            "metric": "measles_cases_casesByOnsetWeek",
            "metric_value": range(1, len(dates) + 1),
            "in_reporting_delay_period": delay,
        }
    ).to_csv(path, index=False)
    return path


def test_reporting_delay_rows_are_excluded_by_default(tmp_path):
    path = _write_export(
        tmp_path / "cases.csv",
        ["2026-01-05", "2026-01-12", "2026-01-19"],
        [False, False, True],
    )

    result = load_london_observed(path)

    assert result["observed_cases"].tolist() == [1, 2]
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-05",
        "2026-01-12",
    ]


def test_reporting_delay_rows_can_be_retained(tmp_path):
    path = _write_export(
        tmp_path / "cases.csv",
        ["2026-01-05", "2026-01-12"],
        [False, True],
    )

    result = load_london_observed(path, exclude_reporting_delay=False)

    assert len(result) == 2


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "cases.csv"
    pd.DataFrame({"date": ["2026-01-05"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_london_observed(path)


def test_nonconsecutive_weeks_are_rejected(tmp_path):
    path = _write_export(
        tmp_path / "cases.csv",
        ["2026-01-05", "2026-01-19"],
        [False, False],
    )

    with pytest.raises(ValueError, match="consecutive weekly series"):
        load_london_observed(path)
