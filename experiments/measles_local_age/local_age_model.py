"""Vectorized region-age measles simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


DEFAULT_CONTACT_MATRIX = np.asarray(
    [
        # u1  1-4  5-10 11-14 15-24 25-34 35+
        [2.0, 3.0, 1.5, 0.8, 1.2, 2.5, 1.8],
        [1.5, 6.0, 3.0, 1.0, 1.5, 2.5, 1.8],
        [0.8, 2.5, 8.0, 4.0, 1.5, 1.5, 1.0],
        [0.5, 1.0, 4.0, 7.0, 2.5, 1.0, 0.8],
        [0.8, 1.2, 1.5, 2.5, 5.0, 2.5, 1.5],
        [1.2, 2.0, 1.2, 1.0, 2.5, 4.0, 2.5],
        [1.0, 1.5, 1.0, 0.8, 1.5, 2.5, 3.5],
    ],
    dtype=float,
)


@dataclass
class LocalAgeParams:
    mu: float = 10.0
    sigma: float = 0.090
    delta: float = 0.0009
    incubation_rate: float = 1.0 / 7.0
    sick_rate: float = 1.0 / 3.0
    quarantine_adherence: float = 1.0
    quarantined_sick_infectiousness: float = 0.2
    unquarantined_sick_infectiousness: float = 1.0
    sick_infectiousness: Optional[float] = None
    phi: float = 0.12
    nu: float = 0.003
    natural_death: float = 0.000027
    birth_rate: float = 0.0000466
    local_mixing: float = 0.85
    seed_infections_per_week: float = 0.0
    reporting_rate: float = 1.0
    noise_scale: float = 0.02
    dt: float = 0.02


class LocalAgeMeaslesSim:
    """S/H/E/I/Q/D simulator with state arrays shaped (region, age).

    E is infected but incubating, I is infectious before sickness, and Q is sick.
    A fraction of Q self-quarantines, so Q has an averaged relative
    infectiousness.
    """

    def __init__(
        self,
        population,
        protected_fraction,
        initial_infected,
        params: LocalAgeParams,
        contact_matrix=None,
        region_risk=None,
    ):
        self.population = np.asarray(population, dtype=float)
        self.protected_fraction = np.asarray(protected_fraction, dtype=float)
        self.initial_infected = np.asarray(initial_infected, dtype=float)
        self.params = params
        self.contact_matrix = np.asarray(
            DEFAULT_CONTACT_MATRIX if contact_matrix is None else contact_matrix,
            dtype=float,
        )
        self.contact_matrix = self.contact_matrix / self.contact_matrix.mean()
        if region_risk is None:
            self.region_risk = np.ones((self.population.shape[0], 1), dtype=float)
        else:
            self.region_risk = np.asarray(region_risk, dtype=float).reshape(-1, 1)
        self.reset()

    def _effective_sick_infectiousness(self):
        p = self.params
        if p.sick_infectiousness is not None:
            return float(np.clip(p.sick_infectiousness, 0.0, 1.0))
        theta = float(np.clip(p.quarantine_adherence, 0.0, 1.0))
        chi_q = float(np.clip(p.quarantined_sick_infectiousness, 0.0, 1.0))
        chi_u = float(np.clip(p.unquarantined_sick_infectiousness, 0.0, 1.0))
        return theta * chi_q + (1.0 - theta) * chi_u

    def reset(self):
        H0 = self.population * self.protected_fraction
        E0 = np.minimum(self.initial_infected, np.maximum(self.population - H0, 0))
        S0 = np.maximum(self.population - H0 - E0, 0)
        self.S = S0.copy()
        self.H = H0.copy()
        self.E = E0.copy()
        self.I = np.zeros_like(S0)
        self.Q = np.zeros_like(S0)
        self.D = np.zeros_like(S0)

    def _force_of_infection(self):
        p = self.params
        living = np.maximum(self.S + self.H + self.E + self.I + self.Q, 1.0)
        infectious_equivalent = self.I + self._effective_sick_infectiousness() * self.Q
        prevalence = infectious_equivalent / living
        age_pressure = prevalence @ self.contact_matrix.T
        global_age_prevalence = infectious_equivalent.sum(axis=0) / np.maximum(living.sum(axis=0), 1.0)
        global_pressure = np.tile(global_age_prevalence @ self.contact_matrix.T, (self.I.shape[0], 1))
        pressure = p.local_mixing * age_pressure + (1.0 - p.local_mixing) * global_pressure
        return p.mu * pressure * self.region_risk

    def step(self):
        p = self.params
        dW = np.random.randn(*self.E.shape) * np.sqrt(p.dt)
        force = self._force_of_infection()
        seed_total_day = p.seed_infections_per_week / 7.0
        seed_weight_source = self.population * self.region_risk
        seed_weights = seed_weight_source / np.maximum(seed_weight_source.sum(), 1.0)
        seeded = seed_total_day * seed_weights

        inf_s = p.sigma * force * self.S
        inf_h = p.delta * force * self.H
        become_infectious = p.incubation_rate * self.E
        become_sick = p.sick_rate * self.I
        recover_q = p.phi * self.Q
        disease_death_q = p.nu * self.Q

        infectious_equivalent = self.I + self._effective_sick_infectiousness() * self.Q
        noise_e = p.noise_scale * infectious_equivalent * dW
        new_infections = np.maximum((inf_s + inf_h + seeded) * p.dt + np.maximum(noise_e, 0), 0)
        e_losses = (become_infectious + p.natural_death * self.E) * p.dt + np.maximum(-noise_e, 0)
        i_losses = (become_sick + p.natural_death * self.I) * p.dt
        q_losses = (recover_q + disease_death_q + p.natural_death * self.Q) * p.dt
        new_sick = become_sick * p.dt

        births = p.birth_rate * (self.S + self.H + self.E + self.I + self.Q) * p.dt
        natural_s = p.natural_death * self.S * p.dt
        natural_h = p.natural_death * self.H * p.dt

        self.S = np.maximum(self.S + births - inf_s * p.dt - seeded * p.dt - natural_s, 0)
        self.H = np.maximum(self.H + recover_q * p.dt - inf_h * p.dt - natural_h, 0)
        self.E = np.maximum(self.E + new_infections - e_losses, 0)
        self.I = np.maximum(self.I + become_infectious * p.dt - i_losses, 0)
        self.Q = np.maximum(self.Q + new_sick - q_losses, 0)
        self.D = np.maximum(self.D + disease_death_q * p.dt, 0)

        return {
            "new_infections": new_infections,
            "new_infectious": become_infectious * p.dt,
            "new_sick": new_sick,
            "reported_cases": p.reporting_rate * new_sick,
            "deaths": disease_death_q * p.dt,
            "active_infections": (self.E + self.I + self.Q).copy(),
            "incubating": self.E.copy(),
            "infectious": self.I.copy(),
            "sick": self.Q.copy(),
        }

    def run_days(self, days):
        """Run the model for whole days and aggregate sub-step incidence.

        This mirrors ``SDEEnvExt.step`` in the original England simulator: the
        numerical SDE step uses ``dt`` as a fraction of one day, so one calendar
        day requires roughly ``1 / dt`` internal Euler-Maruyama updates.
        """
        steps_per_day = max(1, int(round(1.0 / self.params.dt)))
        reported = []
        deaths = []
        for _ in range(days):
            daily_reported = np.zeros_like(self.E)
            daily_deaths = np.zeros_like(self.E)
            for _ in range(steps_per_day):
                info = self.step()
                daily_reported += info["reported_cases"]
                daily_deaths += info["deaths"]
            reported.append(daily_reported)
            deaths.append(daily_deaths)
        return np.asarray(reported), np.asarray(deaths)


def daily_to_weekly(daily_values):
    days, n_regions, n_ages = daily_values.shape
    weeks = days // 7
    trimmed = daily_values[: weeks * 7]
    return trimmed.reshape(weeks, 7, n_regions, n_ages).sum(axis=1)
