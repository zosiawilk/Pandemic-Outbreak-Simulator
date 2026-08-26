"""Fit local-age measles parameters to synthetic region-age weekly targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from local_age_model import LocalAgeMeaslesSim, LocalAgeParams, daily_to_weekly


PARAMETER_BOUNDS = {
    # The structured model has large susceptible pools in several age groups.
    # National-model bounds such as mu=8..13.5 make this prototype explode, so
    # these are intentionally wider/lower.
    "mu": (0.05, 5.0),
    "seed_infections_per_week": (0.0, 5.0),
    "reporting_rate": (1e-6, 0.05),
    "local_mixing": (0.65, 0.98),
    "incubation_rate": (1.0 / 14.0, 1.0 / 2.0),
    "sick_rate": (1.0 / 7.0, 1.0),
    "quarantine_adherence": (0.5, 1.0),
    "protection_multiplier": (0.95, 1.05),
}


def anchor_first_week_to_observed(ensemble, observed_block):
    """Use the first observed week as the visible starting point.

    The simulator's hidden initial state starts from the first observed week,
    but its plotted output is weekly reported *new* cases. Without anchoring,
    the first plotted simulated point can be near zero even when the internal
    infected compartment is nonzero. For block visualisation, we want the
    curves to start at the same historical point and then compare the simulated
    evolution after that.
    """
    anchored = ensemble.copy()
    anchored[:, 0, :, :] = observed_block[0]
    return anchored


def sample_parameters(rng):
    params = {}
    for key, (low, high) in PARAMETER_BOUNDS.items():
        if key == "reporting_rate":
            params[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            params[key] = float(rng.uniform(low, high))
    return params


def load_inputs(input_dir: Path, target_year: int):
    meta = pd.read_csv(input_dir / "region_age_population_protection.csv")
    targets = pd.read_csv(input_dir / "synthetic_region_age_weekly_cases.csv", parse_dates=["date"])
    targets = targets[targets["year"].eq(target_year)].copy()
    if targets.empty:
        raise ValueError(f"No synthetic weekly targets for {target_year}")

    regions = sorted(meta["region"].unique())
    ages = ["under_1", "1_to_4", "5_to_10", "11_to_14", "15_to_24", "25_to_34", "35_and_over"]
    region_index = {r: i for i, r in enumerate(regions)}
    age_index = {a: i for i, a in enumerate(ages)}

    population = np.zeros((len(regions), len(ages)))
    protection = np.zeros_like(population)
    region_risk = np.ones(len(regions))
    for _, row in meta.iterrows():
        if row["region"] in region_index and row["age_group"] in age_index:
            idx = (region_index[row["region"]], age_index[row["age_group"]])
            population[idx] = row["population"]
            protection[idx] = row["protected_fraction"]
    if "region_risk_multiplier" in meta.columns:
        risk_table = meta.groupby("region", as_index=False)["region_risk_multiplier"].mean()
        for _, row in risk_table.iterrows():
            if row["region"] in region_index:
                region_risk[region_index[row["region"]]] = float(row["region_risk_multiplier"])

    dates = sorted(targets["date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}
    observed = np.zeros((len(dates), len(regions), len(ages)))
    for _, row in targets.iterrows():
        observed[
            date_index[row["date"]],
            region_index[row["region"]],
            age_index[row["age_group"]],
        ] = row["synthetic_cases"]

    return {
        "regions": regions,
        "ages": ages,
        "dates": pd.to_datetime(dates),
        "population": population,
        "protection": protection,
        "region_risk": region_risk,
        "observed": observed,
    }


def block_slices(n_weeks, block_weeks):
    return [slice(i, i + block_weeks) for i in range(0, n_weeks - block_weeks + 1, block_weeks)]


def initial_from_observed(block_observed, population):
    total = block_observed[0].sum()
    if total <= 0:
        weights = population / np.maximum(population.sum(), 1.0)
        return weights
    return np.maximum(block_observed[0], 0.0)


def simulate(params, data, weeks, n_sims, seed, initial_infected):
    p = LocalAgeParams(
        mu=params["mu"],
        seed_infections_per_week=params["seed_infections_per_week"],
        reporting_rate=params["reporting_rate"],
        local_mixing=params["local_mixing"],
        incubation_rate=params.get("incubation_rate", 1.0 / 7.0),
        sick_rate=params.get("sick_rate", 1.0 / 3.0),
        quarantine_adherence=params.get("quarantine_adherence", 1.0),
        quarantined_sick_infectiousness=params.get("quarantined_sick_infectiousness", 0.2),
        unquarantined_sick_infectiousness=params.get("unquarantined_sick_infectiousness", 1.0),
        sick_infectiousness=params.get("sick_infectiousness"),
        noise_scale=params.get("noise_scale", 0.02),
        dt=params.get("dt", 0.02),
    )
    protection = np.clip(data["protection"] * params["protection_multiplier"], 0, 0.99)
    trajectories = []
    for sim in range(n_sims):
        np.random.seed(seed + sim)
        model = LocalAgeMeaslesSim(
            population=data["population"],
            protected_fraction=protection,
            initial_infected=initial_infected,
            params=p,
            region_risk=data.get("region_risk"),
        )
        daily_reported, _ = model.run_days(weeks * 7)
        trajectories.append(daily_to_weekly(daily_reported))
    return np.asarray(trajectories)


def score_curve(sim, obs):
    residual = sim - obs
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    obs_total = float(obs.sum())
    sim_total = float(sim.sum())
    rel = abs(sim_total - obs_total) / max(obs_total, 1.0)
    return {
        "rmse": rmse,
        "mae": mae,
        "observed_total_cases": obs_total,
        "simulated_total_cases": sim_total,
        "relative_total_case_error_percent": 100 * rel,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit local-age measles blocks.")
    parser.add_argument("--input-dir", default="experiments/measles_local_age/inputs")
    parser.add_argument("--outdir", default="experiments/measles_local_age/output")
    parser.add_argument("--target-year", type=int, default=2024)
    parser.add_argument("--block-weeks", type=int, default=6)
    parser.add_argument("--number-of-blocks", default="4")
    parser.add_argument("--trials-per-block", type=int, default=10)
    parser.add_argument("--optimizer-sims", type=int, default=2)
    parser.add_argument("--final-sims", type=int, default=10)
    parser.add_argument("--selection-metric", choices=["rmse", "mae"], default="rmse")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(Path(args.input_dir), args.target_year)
    slices = block_slices(data["observed"].shape[0], args.block_weeks)
    if str(args.number_of_blocks).lower() != "all":
        slices = slices[-int(args.number_of_blocks) :]

    trial_rows = []
    best_rows = []
    curve_rows = []
    trajectory_rows = []
    rng = np.random.default_rng(args.seed)
    for block_number, block_slice in enumerate(slices, start=1):
        observed_block = data["observed"][block_slice]
        initial = initial_from_observed(observed_block, data["population"])
        block_trials = []
        for trial in range(args.trials_per_block):
            params = sample_parameters(rng)
            ens = simulate(
                params,
                data,
                args.block_weeks,
                args.optimizer_sims,
                args.seed + 10_000 * block_number,
                initial,
            )
            median = np.median(ens, axis=0)
            metrics = score_curve(median[1:], observed_block[1:])
            row = {
                "block": block_number,
                "trial": trial,
                **params,
                **metrics,
                "selection_metric": args.selection_metric,
                "selection_score": metrics[args.selection_metric],
            }
            block_trials.append(row)
            trial_rows.append(row)
        best = min(block_trials, key=lambda r: r["selection_score"])
        best_rows.append(best)

        ens = simulate(
            best,
            data,
            args.block_weeks,
            args.final_sims,
            args.seed + 50_000 * block_number,
            initial,
        )
        plot_ens = anchor_first_week_to_observed(ens, observed_block)
        median = np.median(plot_ens, axis=0)
        lower = np.percentile(plot_ens, 2.5, axis=0)
        upper = np.percentile(plot_ens, 97.5, axis=0)
        block_dates = data["dates"][block_slice]
        for sim_i in range(plot_ens.shape[0]):
            for week_i, date in enumerate(block_dates):
                for r_i, region in enumerate(data["regions"]):
                    for a_i, age in enumerate(data["ages"]):
                        trajectory_rows.append(
                            {
                                "block": block_number,
                                "simulation": sim_i + 1,
                                "date": date,
                                "region": region,
                                "age_group": age,
                                "sim_cases": plot_ens[sim_i, week_i, r_i, a_i],
                            }
                        )
        for week_i, date in enumerate(block_dates):
            for r_i, region in enumerate(data["regions"]):
                for a_i, age in enumerate(data["ages"]):
                    curve_rows.append(
                        {
                            "block": block_number,
                            "date": date,
                            "region": region,
                            "age_group": age,
                            "observed_cases": observed_block[week_i, r_i, a_i],
                            "sim_median_cases": median[week_i, r_i, a_i],
                            "sim_lower_cases": lower[week_i, r_i, a_i],
                            "sim_upper_cases": upper[week_i, r_i, a_i],
                        }
                    )
        print(f"Finished local-age block {block_number}/{len(slices)}")

    trials = pd.DataFrame(trial_rows)
    best = pd.DataFrame(best_rows)
    curves = pd.DataFrame(curve_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    trials.to_csv(outdir / "local_age_block_trials.csv", index=False)
    best.to_csv(outdir / "local_age_best_parameters_by_block.csv", index=False)
    curves.to_csv(outdir / "local_age_block_curves.csv", index=False)
    trajectories.to_csv(outdir / "local_age_block_stochastic_trajectories.csv", index=False)

    parameter_summary = pd.DataFrame(
        [
            {
                "parameter": name,
                "mean": float(best[name].mean()),
                "median": float(best[name].median()),
                "std": float(best[name].std(ddof=0)),
                "minimum": float(best[name].min()),
                "maximum": float(best[name].max()),
            }
            for name in PARAMETER_BOUNDS
        ]
    )
    parameter_summary.to_csv(outdir / "local_age_parameter_summary.csv", index=False)
    print(f"Saved local-age fitting outputs in {outdir}")
    print(parameter_summary.to_string(index=False))


if __name__ == "__main__":
    main()
