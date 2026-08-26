"""Paired seasonal/non-seasonal complete rolling-origin visual audit."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .london_calibration import (
    DEFAULT_LONDON_POISSON_FITTED_PARAMETERS,
    DEFAULT_LONDON_SEASONAL_POISSON_FITTED_PARAMETERS,
    HistoryConditioningConfig,
    forecast_london_age_groups,
    load_london_confirmed_cases,
)
from .model import RegionAgeInputs, load_default_inputs


def run_paired_seasonal_audit(
    *,
    output_dir: Path | str,
    seasonal_amplitude: float = 0.15,
    seasonal_peak_week: float = 20.0,
    n_simulations: int = 20,
    horizon_weeks: int = 6,
    outbreak_threshold: float = 15.0,
    alarm_probability_cutoff: float = 0.40,
    stride: int = 1,
    random_seed: int = 20260880,
    cases: pd.DataFrame | None = None,
    inputs: RegionAgeInputs | None = None,
    history_conditioning: HistoryConditioningConfig | None = None,
    display_pages: bool = True,
    nonseasonal_fitted_parameters_path: Path | str = DEFAULT_LONDON_POISSON_FITTED_PARAMETERS,
    seasonal_fitted_parameters_path: Path | str = DEFAULT_LONDON_SEASONAL_POISSON_FITTED_PARAMETERS,
) -> tuple[pd.DataFrame, Path]:
    """Audit both models at every eligible origin and save paired pages.

    Each seasonal/non-seasonal pair receives the same truncated observations,
    origin, six held-out weeks, simulation count, and random seed. Each panel
    loads its own independently fitted Poisson parameter vector, so this is a
    comparison of the complete fitted models rather than only a switch of the
    seasonal multiplier.
    """

    if not 0 <= seasonal_amplitude < 1:
        raise ValueError("seasonal_amplitude must be in [0, 1)")
    if not 0 <= seasonal_peak_week < 52.18:
        raise ValueError("seasonal_peak_week must be in [0, 52.18)")
    if n_simulations < 1 or horizon_weeks < 1 or stride < 1:
        raise ValueError("n_simulations, horizon_weeks, and stride must be positive")

    observed = load_london_confirmed_cases() if cases is None else cases.copy()
    observed["date"] = pd.to_datetime(observed["date"])
    model_inputs = inputs or load_default_inputs()
    conditioning = history_conditioning or HistoryConditioningConfig(
        history_weeks=4,
        transmission_multiplier_bounds=(0.8, 1.25),
        regularization_strength=1.0,
        origin_observation_weight=4.0,
        maxiter=80,
    )
    minimum_position = conditioning.history_weeks - 1
    maximum_position = len(observed) - horizon_weeks - 1
    positions = list(range(minimum_position, maximum_position + 1, stride))
    if not positions:
        raise ValueError("No complete rolling origins are available")

    destination = Path(output_dir)
    page_dir = destination / "pages"
    destination.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "seasonal_amplitude": seasonal_amplitude,
        "seasonal_peak_week": seasonal_peak_week,
        "seasonal_period_weeks": 52.18,
        "n_simulations": n_simulations,
        "horizon_weeks": horizon_weeks,
        "outbreak_threshold": outbreak_threshold,
        "alarm_probability_cutoff": alarm_probability_cutoff,
        "origins": len(positions),
        "nonseasonal_fitted_parameters_path": str(Path(nonseasonal_fitted_parameters_path)),
        "seasonal_fitted_parameters_path": str(Path(seasonal_fitted_parameters_path)),
        **{f"history_{k}": v for k, v in asdict(conditioning).items()},
    }]).to_csv(destination / "configuration.csv", index=False)

    panels: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for audit_number, position in enumerate(positions):
        origin_date = pd.Timestamp(observed.iloc[position]["date"])
        origin_count = float(observed.iloc[position]["observed_cases"])
        training = observed.iloc[: position + 1].copy()
        truth = observed.iloc[
            position + 1 : position + 1 + horizon_weeks
        ]["observed_cases"].to_numpy(float)
        target_dates = observed.iloc[
            position + 1 : position + 1 + horizon_weeks
        ]["date"].to_numpy()
        history = training.iloc[-conditioning.history_weeks :]["observed_cases"].to_numpy(float)
        for model_name, fitted_path, override in (
            (
                "non-seasonal",
                nonseasonal_fitted_parameters_path,
                {"seasonal_amplitude": 0.0},
            ),
            (
                "seasonal",
                seasonal_fitted_parameters_path,
                {
                    "seasonal_amplitude": float(seasonal_amplitude),
                    "seasonal_peak_week": float(seasonal_peak_week),
                    "seasonal_period_weeks": 52.18,
                },
            ),
        ):
            result = forecast_london_age_groups(
                cases=training,
                inputs=model_inputs,
                fitted_parameters_path=fitted_path,
                fitted_vector_override=override,
                outbreak_threshold=outbreak_threshold,
                horizon_weeks=horizon_weeks,
                n_simulations=n_simulations,
                random_seed=random_seed + audit_number * 100003,
                warmup_weeks=0,
                sample_weekly_counts=True,
                history_conditioning=conditioning,
            )
            wide = result.all_age_trajectories.pivot(
                index="week", columns="simulation", values="weekly_cases"
            ).sort_index()
            q10 = wide.quantile(.10, axis=1).to_numpy(float)
            q50 = wide.quantile(.50, axis=1).to_numpy(float)
            q90 = wide.quantile(.90, axis=1).to_numpy(float)
            probability = float((wide.max(axis=0) > outbreak_threshold).mean())
            event = bool((truth > outbreak_threshold).any())
            alarm = probability >= alarm_probability_cutoff
            classification = (
                "DETECTED OUTBREAK" if event and alarm else
                "MISSED OUTBREAK" if event else
                "FALSE ALARM" if alarm else
                "CORRECT BELOW THRESHOLD"
            )
            metric = {
                "model": model_name,
                "origin_date": origin_date,
                "origin_cases": origin_count,
                "forecast_probability": probability,
                "observed_event": event,
                "classification": classification,
                "median_mae": float(np.mean(np.abs(q50 - truth))),
                "median_rmse": float(np.sqrt(np.mean((q50 - truth) ** 2))),
                "p10_p90_coverage": float(np.mean((truth >= q10) & (truth <= q90))),
                "brier_score": float((probability - float(event)) ** 2),
            }
            metric_rows.append(metric)
            panels.append({
                "model": model_name, "origin_date": origin_date,
                "origin_count": origin_count, "history": history, "truth": truth,
                "target_dates": target_dates,
                "wide": wide, "q10": q10, "q50": q50, "q90": q90,
                "metrics": metric,
            })
        if (audit_number + 1) % 10 == 0 or audit_number + 1 == len(positions):
            print(f"Prepared {audit_number + 1}/{len(positions)} origins")

    metrics = pd.DataFrame(metric_rows).sort_values(["origin_date", "model"])
    metrics.to_csv(destination / "paired_rolling_origin_metrics.csv", index=False)
    summary = metrics.groupby("model", as_index=False).agg(
        origins=("origin_date", "nunique"),
        mean_MAE=("median_mae", "mean"),
        mean_RMSE=("median_rmse", "mean"),
        mean_Brier=("brier_score", "mean"),
        p10_p90_coverage=("p10_p90_coverage", "mean"),
    )
    summary.to_csv(destination / "paired_model_summary.csv", index=False)

    # Retain every stochastic forecast value so pooled calendar-week bands can
    # be calculated from the simulated paths rather than from origin summaries.
    stochastic_path_tables: list[pd.DataFrame] = []
    for panel in panels:
        wide = panel["wide"]
        long = (
            wide.rename_axis(index="forecast_week", columns="simulation")
            .stack()
            .rename("weekly_cases")
            .reset_index()
        )
        week_to_date = dict(zip(wide.index, panel["target_dates"]))
        week_to_truth = dict(zip(wide.index, panel["truth"]))
        long.insert(0, "model", panel["model"])
        long.insert(1, "origin_date", panel["origin_date"])
        long.insert(2, "origin_cases", panel["origin_count"])
        long["target_date"] = long["forecast_week"].map(week_to_date)
        long["observed_cases"] = long["forecast_week"].map(week_to_truth)
        stochastic_path_tables.append(long)

    stochastic_paths = pd.concat(stochastic_path_tables, ignore_index=True)
    stochastic_paths = stochastic_paths[
        [
            "model", "origin_date", "origin_cases", "target_date",
            "forecast_week", "simulation", "weekly_cases", "observed_cases",
        ]
    ].sort_values(["model", "origin_date", "forecast_week", "simulation"])
    stochastic_paths.to_csv(
        destination / "paired_rolling_stochastic_paths.csv", index=False
    )

    calendar_summary = (
        stochastic_paths.groupby(["model", "target_date"], as_index=False)
        .agg(
            observed_cases=("observed_cases", "first"),
            p10_cases=("weekly_cases", lambda values: values.quantile(.10)),
            median_cases=("weekly_cases", "median"),
            p90_cases=("weekly_cases", lambda values: values.quantile(.90)),
            stochastic_predictions=("weekly_cases", "size"),
            contributing_origins=("origin_date", "nunique"),
        )
        .sort_values(["model", "target_date"])
    )
    calendar_summary.to_csv(
        destination / "paired_rolling_calendar_summary.csv", index=False
    )

    pdf_path = destination / "paired_complete_rolling_origin_audit.pdf"
    x = np.arange(horizon_weeks + 1)
    future_x = np.arange(1, horizon_weeks + 1)
    with PdfPages(pdf_path) as pdf:
        classification_styles = {
            "DETECTED OUTBREAK": {"face": "#dff3e4", "edge": "#238b45"},
            "MISSED OUTBREAK": {"face": "#fde0dd", "edge": "#c51b1d"},
            "FALSE ALARM": {"face": "#fff0c7", "edge": "#d97706"},
            "CORRECT BELOW THRESHOLD": {"face": "#e8eef7", "edge": "#4c78a8"},
        }
        for origin_number in range(len(positions)):
            pair = panels[2 * origin_number : 2 * origin_number + 2]
            page_max = max(
                outbreak_threshold,
                *(float(p["truth"].max()) for p in pair),
                *(float(p["q90"].max()) for p in pair),
            )
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharex=True, sharey=True)
            for ax, panel in zip(axes, pair):
                colour = "tab:orange" if panel["model"] == "seasonal" else "tab:blue"
                for simulation_id in list(panel["wide"].columns[:20]):
                    ax.plot(x, np.r_[panel["origin_count"], panel["wide"][simulation_id]],
                            color=colour, alpha=.09, lw=.7)
                ax.fill_between(future_x, panel["q10"], panel["q90"], color=colour, alpha=.20)
                ax.plot(x, np.r_[panel["origin_count"], panel["q50"]], color=colour, lw=2.2,
                        label="forecast median")
                ax.plot(x, np.r_[panel["origin_count"], panel["truth"]], color="black",
                        marker="o", lw=2, label="held-out truth")
                ax.axhline(outbreak_threshold, color="crimson", ls="--", lw=1.2)
                m = panel["metrics"]
                style = classification_styles[m["classification"]]
                observed_label = "observed outbreak" if m["observed_event"] else "no observed outbreak"
                decision_label = (
                    "alarm issued"
                    if m["forecast_probability"] >= alarm_probability_cutoff
                    else "no alarm"
                )
                ax.set_title(
                    f"{panel['model'].capitalize()}\n{m['classification']}",
                    color=style["edge"], fontweight="bold",
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor(style["edge"])
                    spine.set_linewidth(2.0)
                ax.text(.02, .96,
                        f"{observed_label}; {decision_label}\n"
                        f"P(outbreak): {m['forecast_probability']:.1%}\n"
                        f"MAE: {m['median_mae']:.2f} | RMSE: {m['median_rmse']:.2f}\n"
                        f"p10–p90 coverage: {m['p10_p90_coverage']:.0%}",
                        transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(
                            facecolor=style["face"], edgecolor=style["edge"],
                            alpha=.92, boxstyle="round,pad=.3",
                        ))
                ax.set_xticks(x); ax.set_ylim(0, page_max * 1.16); ax.grid(alpha=.2)
                ax.set_xlabel("Forecast week (0 = observed origin)")
            axes[0].set_ylabel("Reported cases per week")
            axes[1].legend(loc="lower right")
            history_text = " → ".join(f"{v:g}" for v in pair[0]["history"])
            fig.suptitle(
                f"Origin {pair[0]['origin_date'].date()} — history {history_text}\n"
                f"Seasonality: a={seasonal_amplitude:.2f}, peak week={seasonal_peak_week:.1f}",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, .90))
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(page_dir / f"origin_{pair[0]['origin_date']:%Y%m%d}.png",
                        dpi=160, bbox_inches="tight")
            if display_pages:
                plt.show()
            plt.close(fig)
    return summary, pdf_path
