from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbsog_md.forces.base import ForceSolver
from rbsog_md.system import ParticleSystem
from rbsog_md.utils import kinetic_energy, pressure_from_virial, summarize_stats


def _linear_drift_per_time(time_values: np.ndarray, data_values: np.ndarray) -> float:
    mask = np.isfinite(time_values) & np.isfinite(data_values)
    if int(np.sum(mask)) < 2:
        return float("nan")
    x = time_values[mask] - time_values[mask][0]
    y = data_values[mask]
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


@dataclass
class SimulationConfig:
    steps: int = 500
    dt: float = 0.002
    sample_interval: int = 10
    target_temperature: float = 1.0
    thermostat_tau: float = 0.2
    target_pressure: float = 1.0
    barostat_tau: float = 1.0
    compressibility: float = 1e-3


def apply_berendsen_thermostat(
    system: ParticleSystem,
    target_temperature: float,
    dt: float,
    tau_t: float,
) -> None:
    if tau_t <= 0.0:
        return
    current_temperature = system.temperature
    if current_temperature <= 1e-12:
        return
    scale = np.sqrt(1.0 + (dt / tau_t) * (target_temperature / current_temperature - 1.0))
    system.velocities *= scale


def apply_isotropic_berendsen_barostat(
    system: ParticleSystem,
    target_pressure: float,
    current_pressure: float,
    dt: float,
    tau_p: float,
    compressibility: float,
) -> float:
    if tau_p <= 0.0:
        return 1.0
    scale = 1.0 - compressibility * (target_pressure - current_pressure) * (dt / tau_p)
    scale = float(np.clip(scale, 0.98, 1.02))
    system.positions *= scale
    system.box *= scale
    system.velocities *= scale
    system.wrap()
    return scale


def run_simulation(
    system: ParticleSystem,
    solver: ForceSolver,
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()

    records: list[dict[str, float]] = []

    force_result = solver.compute(system=system, rng=rng)
    forces = force_result.forces

    for step in range(1, config.steps + 1):
        t0 = time.perf_counter()

        system.velocities += 0.5 * config.dt * (forces / system.masses[:, None])
        system.positions += config.dt * system.velocities
        system.wrap()

        force_result = solver.compute(system=system, rng=rng)
        forces = force_result.forces

        system.velocities += 0.5 * config.dt * (forces / system.masses[:, None])

        apply_berendsen_thermostat(
            system=system,
            target_temperature=config.target_temperature,
            dt=config.dt,
            tau_t=config.thermostat_tau,
        )

        temperature = system.temperature
        pressure = pressure_from_virial(
            n_particles=system.n_particles,
            volume=system.volume,
            temperature=temperature,
            virial=force_result.virial,
        )

        barostat_scale = apply_isotropic_berendsen_barostat(
            system=system,
            target_pressure=config.target_pressure,
            current_pressure=pressure,
            dt=config.dt,
            tau_p=config.barostat_tau,
            compressibility=config.compressibility,
        )

        kinetic = kinetic_energy(system.velocities, system.masses)
        total_energy = kinetic + force_result.potential
        elapsed = time.perf_counter() - t0

        if step == 1 or step == config.steps or step % config.sample_interval == 0:
            records.append(
                {
                    "step": float(step),
                    "time": float(step * config.dt),
                    "step_time": float(elapsed),
                    "temperature": float(temperature),
                    "pressure": float(pressure),
                    "kinetic": float(kinetic),
                    "potential": float(force_result.potential),
                    "total_energy": float(total_energy),
                    "box_x": float(system.box[0]),
                    "box_y": float(system.box[1]),
                    "box_z": float(system.box[2]),
                    "area_per_lipid": float(system.area_per_lipid()),
                    "thickness_proxy": float(system.membrane_thickness_proxy()),
                    "barostat_scale": float(barostat_scale),
                }
            )

    summary = summarize_records(
        records=records,
        dt=config.dt,
        sample_interval=config.sample_interval,
    )
    return {
        "records": records,
        "summary": summary,
    }


def summarize_records(
    records: list[dict[str, float]],
    dt: float,
    sample_interval: int,
) -> dict[str, float]:
    if not records:
        return {
            "samples": 0,
            "mean_step_time": float("nan"),
            "pressure_variance": float("nan"),
            "temperature_mean": float("nan"),
            "energy_drift_per_time": float("nan"),
        }

    step_times = np.array([r["step_time"] for r in records], dtype=float)
    pressure = np.array([r["pressure"] for r in records], dtype=float)
    temperature = np.array([r["temperature"] for r in records], dtype=float)
    total_energy = np.array([r["total_energy"] for r in records], dtype=float)
    time_values = np.array([r["time"] for r in records], dtype=float)
    area_values = np.array([r["area_per_lipid"] for r in records], dtype=float)
    thickness_values = np.array([r["thickness_proxy"] for r in records], dtype=float)

    total_time = max((records[-1]["step"] - records[0]["step"]) * dt, dt)
    energy_drift = float((total_energy[-1] - total_energy[0]) / total_time)

    return {
        "samples": float(len(records)),
        "mean_step_time": float(np.mean(step_times)),
        "std_step_time": float(np.std(step_times)),
        "pressure_variance": float(np.var(pressure)),
        "temperature_mean": float(np.mean(temperature)),
        "temperature_std": float(np.std(temperature)),
        "energy_drift_per_time": energy_drift,
        "area_per_lipid_mean": float(np.nanmean(area_values)),
        "area_per_lipid_std": float(np.nanstd(area_values)),
        "area_per_lipid_drift_per_time": _linear_drift_per_time(time_values, area_values),
        "thickness_proxy_mean": float(np.nanmean(thickness_values)),
        "thickness_proxy_std": float(np.nanstd(thickness_values)),
        "thickness_proxy_drift_per_time": _linear_drift_per_time(time_values, thickness_values),
        "step_time_stats_mean": summarize_stats(step_times)["mean"],
    }


def write_records_csv(path: Path, records: list[dict[str, float]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
