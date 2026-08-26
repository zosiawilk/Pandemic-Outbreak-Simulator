import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from outbreak_probability_model.london_calibration import (
    FIT_PARAMETER_BOUNDS,
    HistoryConditioningConfig,
    PIPELINE_VERSION,
    SEASONAL_FIT_PARAMETER_BOUNDS,
    _simulate_block,
    binary_classification_metrics_across_cutoffs,
    calibration_starting_vector,
    condition_on_recent_history,
    evaluate_vector,
    fit_simple_grid_parameters,
    load_london_fitted_parameters,
    make_balanced_blocked_validation_design,
    make_six_week_blocks,
)
from outbreak_probability_model.regional_outbreak_forecast import (
    find_below_to_above_origins,
    population_scaled_seed_from_london,
    population_scaled_thresholds,
)
import outbreak_probability_model.regional_outbreak_forecast as regional_forecast
from outbreak_probability_model.model import (
    ModelParameters,
    RegionAgeInputs,
    _run_one_simulation,
    _force_of_infection,
    _step_compartments,
    exceeds_outbreak_threshold,
    load_default_inputs,
)


def _notebook_source(filename: str) -> str:
    notebook_path = Path(__file__).with_name(filename)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_final_fit_notebooks_keep_gradient_optional_and_search_canonical():
    nonseasonal = _notebook_source(
        "London_Calibration_6Week_Rolling_Poisson_Fit.ipynb"
    )
    seasonal = _notebook_source("London_Seasonal_Poisson_Fit.ipynb")

    for source in (nonseasonal, seasonal):
        assert "RUN_OPTIONAL_GRADIENT = False" in source
        assert "if RUN_OPTIONAL_GRADIENT:" in source

    assert "selected_vector = search_vector" in nonseasonal
    assert "seasonal_vector = search_vector" in seasonal
    assert "gradient_vector if" not in nonseasonal
    assert "gradient_vector if" not in seasonal


def test_default_inputs_disable_outcome_derived_regional_risk():
    """Historical outcome totals must not silently enter forecast inputs."""

    inputs = load_default_inputs()
    assert np.allclose(inputs.region_risk_multiplier, 1.0)
    assert "neutral" in inputs.data_sources["region_risk_multiplier"]


def test_mobility_preserves_uniform_prevalence():
    """Spatial averaging must leave equal regional prevalence unchanged."""

    inputs = load_default_inputs(citiness_exponent=0.0)
    assert np.allclose(inputs.mobility_matrix.sum(axis=0), 1.0)

    shape = inputs.population.shape
    living = np.full(shape, 1000.0)
    infectious = np.full(shape, 10.0)
    susceptible = living - infectious
    zero = np.zeros(shape)
    local_params = _closed_parameters(
        mu=1.0,
        local_mixing=1.0,
        sick_contact_multiplier=0.0,
    )
    commuting_params = _closed_parameters(
        mu=1.0,
        local_mixing=0.0,
        sick_contact_multiplier=0.0,
    )
    local_pressure = _force_of_infection(
        susceptible,
        zero,
        zero,
        infectious,
        zero,
        inputs,
        local_params,
    )
    commuting_pressure = _force_of_infection(
        susceptible,
        zero,
        zero,
        infectious,
        zero,
        inputs,
        commuting_params,
    )
    assert np.allclose(commuting_pressure, local_pressure)


def _one_cell_inputs() -> RegionAgeInputs:
    return RegionAgeInputs(
        regions=["London"],
        age_groups=["all"],
        population=np.asarray([[100.0]]),
        protected_fraction=np.asarray([[0.0]]),
        region_risk_multiplier=np.asarray([1.0]),
        population_density_per_km2=np.asarray([1000.0]),
        citiness_contact_multiplier=np.asarray([1.0]),
        latest_weekly_cases=np.asarray([[0.0]]),
        latest_case_date=pd.Timestamp("2026-01-05"),
        mobility_matrix=np.asarray([[1.0]]),
        mobility_layer_weights={"test": 1.0},
        contact_matrix=np.asarray([[1.0]]),
        data_sources={"test": "constructed unit-test input"},
    )


def _two_region_inputs() -> RegionAgeInputs:
    return RegionAgeInputs(
        regions=["London", "North East"],
        age_groups=["all"],
        population=np.asarray([[100.0], [50.0]]),
        protected_fraction=np.zeros((2, 1)),
        region_risk_multiplier=np.ones(2),
        population_density_per_km2=np.asarray([1000.0, 300.0]),
        citiness_contact_multiplier=np.ones(2),
        latest_weekly_cases=np.zeros((2, 1)),
        latest_case_date=pd.Timestamp("2026-01-05"),
        mobility_matrix=np.eye(2),
        mobility_layer_weights={"test": 1.0},
        contact_matrix=np.asarray([[1.0]]),
        data_sources={"test": "constructed two-region input"},
    )


def _closed_parameters(**overrides: float) -> ModelParameters:
    values = {
        "mu": 0.0,
        "sigma": 0.0,
        "delta": 0.0,
        "gamma": 0.0,
        "psi": 0.0,
        "phi": 0.0,
        "nu": 0.0,
        "eta": 0.0,
        "beta_0": 0.0,
        "seed_infections_per_week": 0.0,
        "reporting_rate": 1.0,
        "noise_scale": 0.0,
        "dt": 0.1,
    }
    values.update(overrides)
    return ModelParameters(**values)


def test_operational_event_is_strictly_more_than_ten():
    flags = exceeds_outbreak_threshold(np.asarray([9.0, 10.0, 10.0001, 11.0]), 10)
    assert flags.tolist() == [False, False, True, True]


def test_weekly_latent_diagnostics_are_returned_for_extinction_checks():
    inputs = _one_cell_inputs()
    weekly, latent = _run_one_simulation(
        inputs=inputs,
        params=_closed_parameters(),
        initial_reported_sick=np.asarray([[1.0]]),
        horizon_weeks=2,
        random_seed=1,
        weekly_seed_infections=np.zeros(2),
        return_weekly_latent=True,
    )
    assert weekly.shape == (2, 1, 1)
    assert set(latent) == {"E", "I", "Q"}
    assert all(values.shape == (2, 1, 1) for values in latent.values())


def test_target_region_seeding_does_not_seed_unobserved_regions():
    inputs = _two_region_inputs()
    weekly, final_state = _run_one_simulation(
        inputs=inputs,
        params=_closed_parameters(seed_infections_per_week=7.0),
        initial_reported_sick=np.zeros((2, 1)),
        horizon_weeks=1,
        random_seed=1,
        weekly_seed_infections=np.asarray([7.0]),
        seed_region="London",
        return_final_state=True,
    )
    assert weekly.shape == (1, 2, 1)
    assert final_state.E[:, 0].tolist() == pytest.approx([7.0, 0.0])
    assert final_state.S[:, 0].tolist() == pytest.approx([93.0, 50.0])


def test_regional_seed_is_population_scaled_from_london_fit():
    inputs = _two_region_inputs()
    assert population_scaled_seed_from_london(inputs, "London", 6.0) == pytest.approx(6.0)
    assert population_scaled_seed_from_london(inputs, "North East", 6.0) == pytest.approx(3.0)
    assert FIT_PARAMETER_BOUNDS["seed_infections_per_week"] == (0.0, 20.0)


def test_population_scaled_threshold_uses_london_floor_and_cap():
    inputs = _two_region_inputs()
    thresholds = population_scaled_thresholds(inputs, london_threshold=15, minimum_threshold=3)
    assert thresholds.loc["London"] == 15
    assert thresholds.loc["North East"] == 8

    inputs.population[1, 0] = 200.0
    capped = population_scaled_thresholds(inputs, london_threshold=15, minimum_threshold=3)
    assert capped.loc["North East"] == 15


def test_below_to_above_scan_returns_structured_empty_table(monkeypatch, tmp_path):
    dates = pd.date_range("2026-01-05", periods=12, freq="7D")
    cases = pd.DataFrame(
        [
            {"date": date, "region": region, "observed_cases": 1.0}
            for date in dates
            for region in ("London", "North East")
        ]
    )
    monkeypatch.setattr(
        regional_forecast,
        "load_synthetic_regional_history",
        lambda _input_dir: (cases, pd.DataFrame()),
    )
    monkeypatch.setattr(
        regional_forecast,
        "case_burden_scaled_thresholds",
        lambda *_args, **_kwargs: pd.Series(
            {"London": 15.0, "North East": 3.0}
        ),
    )

    result = find_below_to_above_origins(
        input_dir=tmp_path,
        inputs=_two_region_inputs(),
        history_weeks=4,
        horizon_weeks=6,
    )

    assert result.empty
    assert result.columns.tolist() == [
        "region",
        "origin_date",
        "origin_cases",
        "outbreak_threshold",
        "first_crossing_week",
        "maximum_next_6_weeks",
        "future_cases",
    ]


def test_stale_fitted_parameters_are_rejected(tmp_path):
    path = tmp_path / "old_fit.csv"
    pd.DataFrame(
        [{
            "gamma": 1 / 7,
            "local_mixing": 0.8,
            "contact_scale": 0.05,
            "sick_contact_multiplier": 0.2,
            "sick_mobility_multiplier": 0.0,
            "seed_infections_per_week": 1.0,
            "initial_exposed_per_case": 1.0,
            "initial_infectious_per_case": 1.0,
        }]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match=PIPELINE_VERSION):
        load_london_fitted_parameters(path)


def test_seasonal_fit_file_loads_optional_amplitude_and_phase(tmp_path):
    path = tmp_path / "seasonal_fit.csv"
    vector = calibration_starting_vector(
        parameter_bounds=SEASONAL_FIT_PARAMETER_BOUNDS
    )
    vector["seasonal_amplitude"] = 0.31
    vector["seasonal_peak_week"] = 12.5
    pd.DataFrame([{"pipeline_version": PIPELINE_VERSION, **vector}]).to_csv(
        path, index=False
    )

    params, loaded = load_london_fitted_parameters(path)

    assert loaded["seasonal_amplitude"] == pytest.approx(0.31)
    assert loaded["seasonal_peak_week"] == pytest.approx(12.5)
    assert params.seasonal_amplitude == pytest.approx(0.31)
    assert params.seasonal_peak_week == pytest.approx(12.5)


def test_rolling_block_uses_preceding_observation_as_week_zero():
    cases = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05", periods=5, freq="7D"),
            "observed_cases": [3, 7, 11, 5, 2],
        }
    )
    blocks = make_six_week_blocks(cases, block_weeks=2, step_weeks=1)
    first = blocks.query("block_id == 0")
    assert first["observed_cases"].tolist() == [7, 11]
    assert first["initial_reference_cases"].unique().tolist() == [3.0]
    assert first["conditioning_date"].unique().tolist() == [pd.Timestamp("2026-01-05")]
    assert first["week_in_block"].tolist() == [1, 2]


def test_hidden_warmup_is_rejected_to_protect_date_alignment():
    block = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-12", periods=2, freq="7D"),
            "observed_cases": [2.0, 3.0],
            "initial_reference_cases": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="warmup_weeks must be 0"):
        _simulate_block(
            _one_cell_inputs(),
            _closed_parameters(),
            block,
            random_seed=1,
            warmup_weeks=2,
        )


def test_terminal_state_can_be_continued_without_resetting_history():
    inputs = _one_cell_inputs()
    params = _closed_parameters(psi=0.2, reporting_rate=1.0)
    initial_i = np.asarray([[10.0]])
    initial_q = np.asarray([[0.0]])
    full = _run_one_simulation(
        inputs, params, initial_q, 4, 9, initial_infectious=initial_i
    )
    first, state = _run_one_simulation(
        inputs,
        params,
        initial_q,
        2,
        9,
        initial_infectious=initial_i,
        return_final_state=True,
    )
    second = _run_one_simulation(
        inputs, params, initial_q, 2, 10, initial_state=state
    )
    assert np.concatenate([first, second], axis=0) == pytest.approx(full)


def test_history_conditioning_uses_only_trailing_observations_and_returns_origin_state():
    inputs = _one_cell_inputs()
    params = _closed_parameters(gamma=1 / 7, psi=1 / 3)
    params.contact_scale = 1.0
    params.initial_exposed_per_case = 1.0
    params.initial_infectious_per_case = 1.0
    cases = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05", periods=7, freq="7D"),
            "observed_cases": [99, 2, 3, 4, 6, 8, 11],
        }
    )
    result = condition_on_recent_history(
        cases,
        inputs,
        params,
        HistoryConditioningConfig(history_weeks=4, maxiter=5),
    )
    assert result.conditioning_start == pd.Timestamp("2026-01-26")
    assert result.forecast_origin == pd.Timestamp("2026-02-16")
    assert result.history_fit["observed_cases"].tolist() == [6.0, 8.0, 11.0]
    assert len(result.history_fit) + 1 == 4
    assert np.isfinite(result.history_fit["conditioned_expected_cases"]).all()
    assert isinstance(result.optimizer_success, bool)
    assert result.optimizer_message
    assert result.origin_exposed_total >= 0
    assert result.origin_infectious_total >= 0
    assert result.origin_sick_total >= 0


def test_reporting_rate_changes_observation_but_not_i_to_q_flow():
    state = _step_compartments(
        S=np.asarray([[90.0]]),
        H=np.asarray([[0.0]]),
        E=np.asarray([[0.0]]),
        I=np.asarray([[10.0]]),
        Q=np.asarray([[0.0]]),
        D=np.asarray([[0.0]]),
        inputs=_one_cell_inputs(),
        params=_closed_parameters(psi=1.0, reporting_rate=0.2),
        rng=np.random.default_rng(3),
    )
    S, H, E, I, Q, D, reported = state
    assert Q.item() == pytest.approx(1.0)
    assert I.item() == pytest.approx(9.0)
    assert reported.item() == pytest.approx(0.2)
    assert (S + H + E + I + Q + D).item() == pytest.approx(100.0)


def test_simple_grid_fit_is_deterministic_and_reports_plain_mse():
    cases = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05", periods=4, freq="7D"),
            "observed_cases": [2.0, 3.0, 4.0, 3.0],
        }
    )
    blocks = make_six_week_blocks(cases, block_weeks=2)
    kwargs = dict(
        blocks=blocks,
        inputs=_one_cell_inputs(),
        base_parameters=_closed_parameters(),
        parameter_grids={"contact_scale": (0.01, 0.02)},
        passes=1,
        progress=False,
    )
    selected_a, trials_a, diagnostics_a = fit_simple_grid_parameters(**kwargs)
    selected_b, trials_b, diagnostics_b = fit_simple_grid_parameters(**kwargs)
    assert selected_a == selected_b
    assert trials_a["objective_mse"].tolist() == pytest.approx(
        trials_b["objective_mse"].tolist()
    )
    assert diagnostics_a["objective_metric"].eq("mean_squared_error").all()
    assert diagnostics_a["mean_squared_error"].tolist() == pytest.approx(
        diagnostics_b["mean_squared_error"].tolist()
    )


def test_poisson_objective_is_available_and_reported_in_diagnostics():
    cases = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05", periods=4, freq="7D"),
            "observed_cases": [2.0, 3.0, 4.0, 3.0],
        }
    )
    blocks = make_six_week_blocks(cases, block_weeks=2)
    score, diagnostics = evaluate_vector(
        calibration_starting_vector(_closed_parameters()),
        blocks,
        _one_cell_inputs(),
        base_parameters=_closed_parameters(),
        force_deterministic=True,
        overlap_weighted=True,
        objective_metric="poisson_nll",
    )
    expected = np.average(diagnostics["poisson_nll"], weights=diagnostics["window_weight"])
    assert np.isfinite(score)
    assert score == pytest.approx(expected)
    assert diagnostics["objective_metric"].eq("poisson_nll").all()


def test_independent_compartment_noise_is_reproducible_but_not_population_preserving():
    initial = {
        "S": np.asarray([[100.0]]),
        "H": np.asarray([[80.0]]),
        "E": np.asarray([[20.0]]),
        "I": np.asarray([[15.0]]),
        "Q": np.asarray([[10.0]]),
        "D": np.asarray([[5.0]]),
    }
    kwargs = dict(
        **initial,
        inputs=_one_cell_inputs(),
        params=_closed_parameters(noise_scale=0.2),
    )
    first = _step_compartments(**kwargs, rng=np.random.default_rng(17))
    repeated = _step_compartments(**kwargs, rng=np.random.default_rng(17))
    different = _step_compartments(**kwargs, rng=np.random.default_rng(18))

    assert all(a == pytest.approx(b) for a, b in zip(first, repeated))
    assert any(not np.allclose(a, b) for a, b in zip(first[:6], different[:6]))

    initial_total = sum(initial.values()).item()
    realised_total = sum(first[:6]).item()
    assert realised_total != pytest.approx(initial_total)
    assert min(value.item() for value in first[:6]) >= 0.0


def test_sh_noise_multiplier_controls_only_s_and_h_noise():
    initial = {
        "S": np.asarray([[100.0]]),
        "H": np.asarray([[80.0]]),
        "E": np.asarray([[20.0]]),
        "I": np.asarray([[15.0]]),
        "Q": np.asarray([[10.0]]),
        "D": np.asarray([[5.0]]),
    }
    state = _step_compartments(
        **initial,
        inputs=_one_cell_inputs(),
        params=_closed_parameters(noise_scale=0.2, sh_noise_multiplier=0.0),
        rng=np.random.default_rng(17),
    )
    assert state[0].item() == pytest.approx(initial["S"].item())
    assert state[1].item() == pytest.approx(initial["H"].item())
    assert any(
        not np.isclose(state[index].item(), initial[name].item())
        for index, name in enumerate(("E", "I", "Q", "D"), start=2)
    )


def test_zero_noise_retains_the_deterministic_mean_path():
    state = _step_compartments(
        S=np.asarray([[90.0]]),
        H=np.asarray([[0.0]]),
        E=np.asarray([[0.0]]),
        I=np.asarray([[10.0]]),
        Q=np.asarray([[0.0]]),
        D=np.asarray([[0.0]]),
        inputs=_one_cell_inputs(),
        params=_closed_parameters(psi=0.5, noise_scale=0.0),
        rng=np.random.default_rng(3),
    )
    S, H, E, I, Q, D, reported = state
    assert (S + H + E + I + Q + D).item() == pytest.approx(100.0)
    assert I.item() == pytest.approx(9.5)
    assert Q.item() == pytest.approx(0.5)
    assert reported.item() == pytest.approx(0.5)


def test_balanced_blocked_design_has_disjoint_equal_classes():
    values = [1.0] * 12 + [11.0] * 6 + [1.0] * 6 + [12.0] * 6
    cases = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05", periods=len(values), freq="7D"),
            "observed_cases": values,
        }
    )
    design = make_balanced_blocked_validation_design(
        cases, training_weeks=6, horizon_weeks=6, random_seed=7
    )
    selected = design.query("selected_for_balanced_test")
    assert selected["observed_event"].value_counts().to_dict() == {False: 2, True: 2}
    ordered = selected.sort_values("forecast_start")
    assert (ordered["forecast_start"].iloc[1:].to_numpy() >
            ordered["forecast_end"].iloc[:-1].to_numpy()).all()
    assert selected["training_end_date"].nunique() == 1


def test_binary_metrics_use_probability_cutoff_and_strict_labels():
    forecasts = pd.DataFrame(
        {
            "predicted_probability": [0.1, 0.3, 0.2, 0.8],
            "observed_event": [False, False, True, True],
        }
    )
    metrics = binary_classification_metrics_across_cutoffs(
        forecasts, probability_cutoffs=[0.2]
    ).iloc[0]
    assert metrics.true_positive == 2
    assert metrics.false_negative == 0
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.sensitivity == pytest.approx(1.0)
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
