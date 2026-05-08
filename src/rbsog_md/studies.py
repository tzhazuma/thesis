from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rbsog_md.benchmark import create_solver, run_batch_size_sweep, run_benchmark
from rbsog_md.simulation import SimulationConfig, run_simulation, write_records_csv
from rbsog_md.stats import (
    bootstrap_mean_ratio_ci,
    ci95_half_width,
    loglog_fit,
    mean_and_std,
    permutation_test_dispersion,
    permutation_test_mean_difference,
    rolling_mean,
    rolling_std,
)
from rbsog_md.system import ParticleSystem, build_membrane_proxy_system
from rbsog_md.utils import pressure_from_virial, save_json


BASE_BOX = (16.0, 16.0, 20.0)
BASE_LIPIDS = 128
BASE_SOLVENT = 512
BASE_PARTICLES = BASE_LIPIDS + BASE_SOLVENT


@dataclass
class SystemSpec:
    label: str
    n_lipids: int
    n_solvent: int
    box: tuple[float, float, float]

    @property
    def n_particles(self) -> int:
        return int(self.n_lipids + self.n_solvent)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scaled_box(n_particles: int) -> tuple[float, float, float]:
    factor = (float(n_particles) / float(BASE_PARTICLES)) ** (1.0 / 3.0)
    return tuple(float(v * factor) for v in BASE_BOX)


def _relative_force_rmse(reference: np.ndarray, approx: np.ndarray) -> float:
    denom = float(np.sum(reference * reference))
    if denom <= 1e-16:
        return float("nan")
    diff = approx - reference
    return float(np.sqrt(np.sum(diff * diff) / denom))


def _relative_scalar_error(reference: float, approx: float) -> float:
    return float(abs(approx - reference) / max(abs(reference), 1e-12))


def _equilibrated_snapshot(
    *,
    spec: SystemSpec,
    seed: int,
    warmup_steps: int,
    grid_shape: tuple[int, int, int],
) -> ParticleSystem:
    system = build_membrane_proxy_system(
        n_lipids=spec.n_lipids,
        n_solvent=spec.n_solvent,
        box=spec.box,
        temperature=1.0,
        seed=seed,
    )
    solver = create_solver("pppm", grid_shape=grid_shape)
    warmup = SimulationConfig(
        steps=warmup_steps,
        dt=0.002,
        sample_interval=max(warmup_steps, 1),
        target_temperature=1.0,
        thermostat_tau=0.2,
        target_pressure=1.0,
        barostat_tau=1.0,
        compressibility=1e-3,
    )
    result = run_simulation(system=system, solver=solver, config=warmup, rng=np.random.default_rng(seed + 99_000))
    return result["final_system"]


def run_pppm_grid_fairness_study(
    out_dir: Path,
    *,
    grid_sizes: list[int],
    runtime_seeds: list[int],
    runtime_steps: int,
    accuracy_seeds: list[int],
    accuracy_spec: SystemSpec,
    warmup_steps: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_config = SimulationConfig(steps=runtime_steps, dt=0.002, sample_interval=20)
    runtime_rows: list[dict[str, Any]] = []

    for grid in grid_sizes:
        grid_shape = (grid, grid, grid)
        result = run_benchmark(
            out_dir=out_dir / f"runtime_grid_{grid}",
            solver_names=["pppm"],
            seeds=runtime_seeds,
            sim_config=sim_config,
            n_lipids=BASE_LIPIDS,
            n_solvent=BASE_SOLVENT,
            box=BASE_BOX,
            init_temperature=1.0,
            grid_shape=grid_shape,
            batch_size=100,
            cutoff_short=1.2,
            sog_terms=12,
            neighbor_skin=0.3,
            neighbor_rebuild_interval=10,
            use_numba=True,
        )
        row = result["summary"]["by_solver"][0]
        runtime_rows.append(
            {
                "grid_size": float(grid),
                "mean_step_time": float(row["mean_step_time"]),
                "pressure_variance": float(row["pressure_variance"]),
            }
        )

    accuracy_rows: list[dict[str, Any]] = []
    for seed in accuracy_seeds:
        snapshot = _equilibrated_snapshot(
            spec=accuracy_spec,
            seed=seed,
            warmup_steps=warmup_steps,
            grid_shape=(32, 32, 32),
        )
        reference_solver = create_solver("direct")
        ref_result = reference_solver.compute(snapshot.copy())
        ref_pressure = pressure_from_virial(
            n_particles=snapshot.n_particles,
            volume=snapshot.volume,
            temperature=snapshot.temperature,
            virial=ref_result.virial,
        )
        for grid in grid_sizes:
            solver = create_solver("pppm", grid_shape=(grid, grid, grid))
            approx = solver.compute(snapshot.copy())
            approx_pressure = pressure_from_virial(
                n_particles=snapshot.n_particles,
                volume=snapshot.volume,
                temperature=snapshot.temperature,
                virial=approx.virial,
            )
            accuracy_rows.append(
                {
                    "seed": float(seed),
                    "grid_size": float(grid),
                    "force_rmse_rel": _relative_force_rmse(ref_result.forces, approx.forces),
                    "potential_error_rel": _relative_scalar_error(ref_result.potential, approx.potential),
                    "virial_error_rel": _relative_scalar_error(ref_result.virial, approx.virial),
                    "pressure_error_rel": _relative_scalar_error(ref_pressure, approx_pressure),
                }
            )

    accuracy_summary: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    force_errors = []
    time_values = []
    for grid in grid_sizes:
        subset = [r for r in accuracy_rows if int(r["grid_size"]) == int(grid)]
        force_mean = float(np.mean([float(r["force_rmse_rel"]) for r in subset]))
        pot_mean = float(np.mean([float(r["potential_error_rel"]) for r in subset]))
        vir_mean = float(np.mean([float(r["virial_error_rel"]) for r in subset]))
        pres_mean = float(np.mean([float(r["pressure_error_rel"]) for r in subset]))
        runtime_row = next(r for r in runtime_rows if int(r["grid_size"]) == int(grid))
        accuracy_summary.append(
            {
                "grid_size": float(grid),
                "force_rmse_rel": force_mean,
                "potential_error_rel": pot_mean,
                "virial_error_rel": vir_mean,
                "pressure_error_rel": pres_mean,
            }
        )
        force_errors.append(force_mean)
        time_values.append(float(runtime_row["mean_step_time"]))

    time_arr = np.asarray(time_values, dtype=float)
    error_arr = np.asarray(force_errors, dtype=float)
    time_norm = (time_arr - np.min(time_arr)) / (np.ptp(time_arr) + 1e-12)
    error_norm = (error_arr - np.min(error_arr)) / (np.ptp(error_arr) + 1e-12)
    scores = np.sqrt(0.5 * time_norm * time_norm + 0.5 * error_norm * error_norm)
    best_idx = int(np.argmin(scores))
    recommended_grid = int(grid_sizes[best_idx])

    for idx, grid in enumerate(grid_sizes):
        runtime_row = next(r for r in runtime_rows if int(r["grid_size"]) == int(grid))
        accuracy_row = next(r for r in accuracy_summary if int(r["grid_size"]) == int(grid))
        combined_rows.append(
            {
                "grid_size": float(grid),
                "mean_step_time": float(runtime_row["mean_step_time"]),
                "pressure_variance": float(runtime_row["pressure_variance"]),
                "force_rmse_rel": float(accuracy_row["force_rmse_rel"]),
                "potential_error_rel": float(accuracy_row["potential_error_rel"]),
                "virial_error_rel": float(accuracy_row["virial_error_rel"]),
                "pressure_error_rel": float(accuracy_row["pressure_error_rel"]),
                "fairness_score": float(scores[idx]),
                "recommended": float(1.0 if idx == best_idx else 0.0),
            }
        )

    _write_csv(out_dir / "pppm_grid_runtime.csv", runtime_rows)
    _write_csv(out_dir / "pppm_grid_accuracy_runs.csv", accuracy_rows)
    _write_csv(out_dir / "pppm_grid_summary.csv", combined_rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(time_arr, error_arr, color="#0f766e", s=60)
    for idx, grid in enumerate(grid_sizes):
        ax.annotate(f"{grid}$^3$", (time_arr[idx], error_arr[idx]), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.scatter([time_arr[best_idx]], [error_arr[best_idx]], color="#be123c", marker="D", s=90)
    ax.set_xlabel("Mean step time (s)")
    ax.set_ylabel("Relative force RMSE vs direct")
    ax.set_title("PPPM Grid Fairness Trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pppm_grid_tradeoff.png", dpi=220)
    plt.close(fig)

    payload = {
        "recommended_grid": recommended_grid,
        "summary_csv": "pppm_grid_summary.csv",
        "tradeoff_plot": "pppm_grid_tradeoff.png",
    }
    save_json(out_dir / "pppm_grid_summary.json", payload)
    return payload


def run_accuracy_reference_study(
    out_dir: Path,
    *,
    specs: list[SystemSpec],
    snapshot_seeds: list[int],
    warmup_steps: int,
    pppm_grid: int,
    batch_sizes: list[int],
    sog_terms_list: list[int],
    cutoff_short: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        for seed in snapshot_seeds:
            snapshot = _equilibrated_snapshot(
                spec=spec,
                seed=seed,
                warmup_steps=warmup_steps,
                grid_shape=(pppm_grid, pppm_grid, pppm_grid),
            )
            direct = create_solver("direct").compute(snapshot.copy())
            direct_pressure = pressure_from_virial(
                n_particles=snapshot.n_particles,
                volume=snapshot.volume,
                temperature=snapshot.temperature,
                virial=direct.virial,
            )

            pppm = create_solver("pppm", grid_shape=(pppm_grid, pppm_grid, pppm_grid)).compute(snapshot.copy())
            pppm_pressure = pressure_from_virial(
                n_particles=snapshot.n_particles,
                volume=snapshot.volume,
                temperature=snapshot.temperature,
                virial=pppm.virial,
            )
            rows.append(
                {
                    "system": spec.label,
                    "seed": float(seed),
                    "family": "pppm",
                    "config": f"grid={pppm_grid}^3",
                    "force_rmse_rel": _relative_force_rmse(direct.forces, pppm.forces),
                    "potential_error_rel": _relative_scalar_error(direct.potential, pppm.potential),
                    "virial_error_rel": _relative_scalar_error(direct.virial, pppm.virial),
                    "pressure_error_rel": _relative_scalar_error(direct_pressure, pppm_pressure),
                    "batch_size": float("nan"),
                    "sog_terms": float("nan"),
                }
            )

            for batch_size in batch_sizes:
                solver = create_solver(
                    "rbsog",
                    grid_shape=(pppm_grid, pppm_grid, pppm_grid),
                    batch_size=batch_size,
                    cutoff_short=cutoff_short,
                    sog_terms=12,
                )
                approx = solver.compute(snapshot.copy(), rng=np.random.default_rng(seed + batch_size + 1234))
                approx_pressure = pressure_from_virial(
                    n_particles=snapshot.n_particles,
                    volume=snapshot.volume,
                    temperature=snapshot.temperature,
                    virial=approx.virial,
                )
                rows.append(
                    {
                        "system": spec.label,
                        "seed": float(seed),
                        "family": "rbsog_batch",
                        "config": f"P={batch_size},M=12",
                        "force_rmse_rel": _relative_force_rmse(direct.forces, approx.forces),
                        "potential_error_rel": _relative_scalar_error(direct.potential, approx.potential),
                        "virial_error_rel": _relative_scalar_error(direct.virial, approx.virial),
                        "pressure_error_rel": _relative_scalar_error(direct_pressure, approx_pressure),
                        "batch_size": float(batch_size),
                        "sog_terms": 12.0,
                    }
                )

            for sog_terms in sog_terms_list:
                solver = create_solver(
                    "rbsog",
                    grid_shape=(pppm_grid, pppm_grid, pppm_grid),
                    batch_size=100,
                    cutoff_short=cutoff_short,
                    sog_terms=sog_terms,
                )
                approx = solver.compute(snapshot.copy(), rng=np.random.default_rng(seed + sog_terms + 4321))
                approx_pressure = pressure_from_virial(
                    n_particles=snapshot.n_particles,
                    volume=snapshot.volume,
                    temperature=snapshot.temperature,
                    virial=approx.virial,
                )
                rows.append(
                    {
                        "system": spec.label,
                        "seed": float(seed),
                        "family": "rbsog_terms",
                        "config": f"P=100,M={sog_terms}",
                        "force_rmse_rel": _relative_force_rmse(direct.forces, approx.forces),
                        "potential_error_rel": _relative_scalar_error(direct.potential, approx.potential),
                        "virial_error_rel": _relative_scalar_error(direct.virial, approx.virial),
                        "pressure_error_rel": _relative_scalar_error(direct_pressure, approx_pressure),
                        "batch_size": 100.0,
                        "sog_terms": float(sog_terms),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    keys = sorted({(str(r["system"]), str(r["family"]), str(r["config"])) for r in rows})
    for system_label, family, config in keys:
        subset = [r for r in rows if str(r["system"]) == system_label and str(r["family"]) == family and str(r["config"]) == config]
        summary_rows.append(
            {
                "system": system_label,
                "family": family,
                "config": config,
                "batch_size": float(subset[0]["batch_size"]),
                "sog_terms": float(subset[0]["sog_terms"]),
                "force_rmse_rel": float(np.mean([float(r["force_rmse_rel"]) for r in subset])),
                "potential_error_rel": float(np.mean([float(r["potential_error_rel"]) for r in subset])),
                "virial_error_rel": float(np.mean([float(r["virial_error_rel"]) for r in subset])),
                "pressure_error_rel": float(np.mean([float(r["pressure_error_rel"]) for r in subset])),
            }
        )

    _write_csv(out_dir / "accuracy_runs.csv", rows)
    _write_csv(out_dir / "accuracy_summary.csv", summary_rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for spec in specs:
        subset = sorted(
            [r for r in summary_rows if r["system"] == spec.label and r["family"] == "rbsog_batch"],
            key=lambda r: float(r["batch_size"]),
        )
        ax.plot(
            [float(r["batch_size"]) for r in subset],
            [float(r["force_rmse_rel"]) for r in subset],
            marker="o",
            label=spec.label,
        )
    ax.set_xlabel("Batch size P")
    ax.set_ylabel("Relative force RMSE vs direct")
    ax.set_title("RBSOG Force Error vs Batch Size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_force_vs_batch.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for spec in specs:
        subset = sorted(
            [r for r in summary_rows if r["system"] == spec.label and r["family"] == "rbsog_terms"],
            key=lambda r: float(r["sog_terms"]),
        )
        ax.plot(
            [float(r["sog_terms"]) for r in subset],
            [float(r["force_rmse_rel"]) for r in subset],
            marker="s",
            label=spec.label,
        )
    ax.set_xlabel("Number of SOG terms M")
    ax.set_ylabel("Relative force RMSE vs direct")
    ax.set_title("RBSOG Force Error vs SOG Terms")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_force_vs_sog_terms.png", dpi=220)
    plt.close(fig)

    payload = {
        "summary_csv": "accuracy_summary.csv",
        "force_vs_batch_plot": "accuracy_force_vs_batch.png",
        "force_vs_sog_terms_plot": "accuracy_force_vs_sog_terms.png",
    }
    save_json(out_dir / "accuracy_summary.json", payload)
    return payload


def run_scaling_study(
    out_dir: Path,
    *,
    specs: list[SystemSpec],
    seeds: list[int],
    steps: int,
    sample_interval: int,
    grid_size: int,
    batch_size: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_config = SimulationConfig(steps=steps, dt=0.002, sample_interval=sample_interval)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        result = run_benchmark(
            out_dir=out_dir / spec.label,
            solver_names=["pppm", "rbsog"],
            seeds=seeds,
            sim_config=sim_config,
            n_lipids=spec.n_lipids,
            n_solvent=spec.n_solvent,
            box=spec.box,
            init_temperature=1.0,
            grid_shape=(grid_size, grid_size, grid_size),
            batch_size=batch_size,
            cutoff_short=1.2,
            sog_terms=12,
            neighbor_skin=0.3,
            neighbor_rebuild_interval=10,
            use_numba=True,
        )
        for row in result["summary"]["by_solver"]:
            rows.append(
                {
                    "system": spec.label,
                    "n_particles": float(spec.n_particles),
                    "solver": str(row["solver"]),
                    "mean_step_time": float(row["mean_step_time"]),
                    "pressure_variance": float(row["pressure_variance"]),
                }
            )

    _write_csv(out_dir / "scaling_summary.csv", rows)
    solvers = sorted({str(r["solver"]) for r in rows})
    fit_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for solver in solvers:
        subset = sorted([r for r in rows if str(r["solver"]) == solver], key=lambda r: float(r["n_particles"]))
        x = [float(r["n_particles"]) for r in subset]
        y = [float(r["mean_step_time"]) for r in subset]
        ax.plot(x, y, marker="o", label=solver.upper())
        fit = loglog_fit(x, y)
        fit_rows.append(
            {
                "solver": solver,
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r_squared": fit.r_squared,
            }
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of particles N")
    ax.set_ylabel("Mean step time (s)")
    ax.set_title("Scaling of Mean Step Time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_step_time.png", dpi=220)
    plt.close(fig)

    pppm = {int(r["n_particles"]): float(r["mean_step_time"]) for r in rows if str(r["solver"]) == "pppm"}
    rbsog = {int(r["n_particles"]): float(r["mean_step_time"]) for r in rows if str(r["solver"]) == "rbsog"}
    ratio_rows = []
    for n_particles in sorted(pppm.keys()):
        ratio_rows.append(
            {
                "n_particles": float(n_particles),
                "slowdown_ratio": float(rbsog[n_particles] / max(pppm[n_particles], 1e-12)),
            }
        )
    _write_csv(out_dir / "scaling_exponent_table.csv", fit_rows)
    _write_csv(out_dir / "scaling_slowdown_ratio.csv", ratio_rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(
        [float(r["n_particles"]) for r in ratio_rows],
        [float(r["slowdown_ratio"]) for r in ratio_rows],
        marker="D",
        color="#b45309",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of particles N")
    ax.set_ylabel("RBSOG / PPPM slowdown ratio")
    ax.set_title("Scaling of Relative Slowdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_slowdown_ratio.png", dpi=220)
    plt.close(fig)

    payload = {
        "summary_csv": "scaling_summary.csv",
        "fit_table_csv": "scaling_exponent_table.csv",
        "step_time_plot": "scaling_step_time.png",
        "slowdown_plot": "scaling_slowdown_ratio.png",
    }
    save_json(out_dir / "scaling_summary.json", payload)
    return payload


def run_long_stability_study(
    out_dir: Path,
    *,
    seeds: list[int],
    steps: int,
    sample_interval: int,
    grid_size: int,
    batch_size: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_config = SimulationConfig(steps=steps, dt=0.002, sample_interval=sample_interval)
    result = run_benchmark(
        out_dir=out_dir / "benchmark",
        solver_names=["pppm", "rbsog"],
        seeds=seeds,
        sim_config=sim_config,
        n_lipids=BASE_LIPIDS,
        n_solvent=BASE_SOLVENT,
        box=BASE_BOX,
        init_temperature=1.0,
        grid_shape=(grid_size, grid_size, grid_size),
        batch_size=batch_size,
        cutoff_short=1.2,
        sog_terms=12,
        neighbor_skin=0.3,
        neighbor_rebuild_interval=10,
        use_numba=True,
    )

    summary_rows: list[dict[str, Any]] = []
    rolling_payload: dict[str, dict[str, list[np.ndarray]]] = {
        "pppm": {"time": [], "area": [], "thickness": [], "pressure": []},
        "rbsog": {"time": [], "area": [], "thickness": [], "pressure": []},
    }
    benchmark_root = out_dir / "benchmark"
    for solver in ["pppm", "rbsog"]:
        for seed in seeds:
            run_dir = benchmark_root / solver / f"seed_{seed}"
            rows = _read_csv(run_dir / "stability_timeseries.csv")
            time = np.array([float(r["time"]) for r in rows], dtype=float)
            area = np.array([float(r["area_per_lipid"]) for r in rows], dtype=float)
            thickness = np.array([float(r["thickness_proxy"]) for r in rows], dtype=float)
            pressure = np.array([float(r["pressure"]) for r in rows], dtype=float)
            rolling_payload[solver]["time"].append(time)
            rolling_payload[solver]["area"].append(rolling_mean(area, window=10))
            rolling_payload[solver]["thickness"].append(rolling_mean(thickness, window=10))
            rolling_payload[solver]["pressure"].append(rolling_mean(pressure, window=10))
        row = next(r for r in result["summary"]["by_solver"] if str(r["solver"]) == solver)
        summary_rows.append(
            {
                "solver": solver,
                "mean_step_time": float(row["mean_step_time"]),
                "pressure_variance": float(row["pressure_variance"]),
                "area_per_lipid_std": float(row["area_per_lipid_std"]),
                "thickness_proxy_std": float(row["thickness_proxy_std"]),
                "area_per_lipid_drift_per_time": float(row["area_per_lipid_drift_per_time"]),
                "thickness_proxy_drift_per_time": float(row["thickness_proxy_drift_per_time"]),
            }
        )

    _write_csv(out_dir / "long_stability_summary.csv", summary_rows)

    for metric, filename, ylabel, title in [
        ("area", "rolling_area.png", "Area per lipid (rolling mean)", "Long-horizon Area Stability"),
        ("thickness", "rolling_thickness.png", "Thickness proxy (rolling mean)", "Long-horizon Thickness Stability"),
        ("pressure", "rolling_pressure.png", "Pressure (rolling mean)", "Long-horizon Pressure Stability"),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for solver, color in [("pppm", "#0f766e"), ("rbsog", "#b45309")]:
            curves = rolling_payload[solver][metric]
            times = rolling_payload[solver]["time"]
            common = np.mean(np.stack(curves, axis=0), axis=0)
            spread = np.std(np.stack(curves, axis=0), axis=0)
            ax.plot(times[0], common, color=color, label=solver.upper())
            ax.fill_between(times[0], common - spread, common + spread, color=color, alpha=0.18)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    payload = {
        "summary_csv": "long_stability_summary.csv",
        "rolling_area_plot": "rolling_area.png",
        "rolling_thickness_plot": "rolling_thickness.png",
        "rolling_pressure_plot": "rolling_pressure.png",
    }
    save_json(out_dir / "long_stability_summary.json", payload)
    return payload


def run_rbsog_ablation_study(
    out_dir: Path,
    *,
    accuracy_spec: SystemSpec,
    snapshot_seeds: list[int],
    warmup_steps: int,
    pppm_grid: int,
    batch_sizes: list[int],
    sog_terms_list: list[int],
    cutoff_values: list[float],
    performance_seeds: list[int],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    error_rows: list[dict[str, Any]] = []
    for seed in snapshot_seeds:
        snapshot = _equilibrated_snapshot(
            spec=accuracy_spec,
            seed=seed,
            warmup_steps=warmup_steps,
            grid_shape=(pppm_grid, pppm_grid, pppm_grid),
        )
        direct = create_solver("direct").compute(snapshot.copy())
        for batch_size in batch_sizes:
            for sog_terms in sog_terms_list:
                solver = create_solver(
                    "rbsog",
                    grid_shape=(pppm_grid, pppm_grid, pppm_grid),
                    batch_size=batch_size,
                    cutoff_short=1.2,
                    sog_terms=sog_terms,
                )
                approx = solver.compute(snapshot.copy(), rng=np.random.default_rng(seed + batch_size + 10 * sog_terms))
                error_rows.append(
                    {
                        "seed": float(seed),
                        "batch_size": float(batch_size),
                        "sog_terms": float(sog_terms),
                        "force_rmse_rel": _relative_force_rmse(direct.forces, approx.forces),
                    }
                )
    _write_csv(out_dir / "ablation_error_runs.csv", error_rows)

    perf_rows: list[dict[str, Any]] = []
    sim_config = SimulationConfig(steps=200, dt=0.002, sample_interval=20)
    for batch_size in batch_sizes:
        for cutoff in cutoff_values:
            result = run_benchmark(
                out_dir=out_dir / f"perf_P{batch_size}_C{str(cutoff).replace('.', 'p')}",
                solver_names=["rbsog"],
                seeds=performance_seeds,
                sim_config=sim_config,
                n_lipids=BASE_LIPIDS,
                n_solvent=BASE_SOLVENT,
                box=BASE_BOX,
                init_temperature=1.0,
                grid_shape=(pppm_grid, pppm_grid, pppm_grid),
                batch_size=batch_size,
                cutoff_short=cutoff,
                sog_terms=12,
                neighbor_skin=0.3,
                neighbor_rebuild_interval=10,
                use_numba=True,
            )
            row = result["summary"]["by_solver"][0]
            perf_rows.append(
                {
                    "batch_size": float(batch_size),
                    "cutoff_short": float(cutoff),
                    "mean_step_time": float(row["mean_step_time"]),
                    "pressure_variance": float(row["pressure_variance"]),
                }
            )
    _write_csv(out_dir / "ablation_performance_summary.csv", perf_rows)

    def _heatmap(rows: list[dict[str, Any]], x_key: str, y_key: str, value_key: str, path: Path, title: str, cmap: str) -> None:
        x_values = sorted({float(r[x_key]) for r in rows})
        y_values = sorted({float(r[y_key]) for r in rows})
        grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
        for yi, y_val in enumerate(y_values):
            for xi, x_val in enumerate(x_values):
                subset = [r for r in rows if float(r[x_key]) == x_val and float(r[y_key]) == y_val]
                if subset:
                    grid[yi, xi] = float(np.mean([float(r[value_key]) for r in subset]))
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(x_values)), [str(int(v)) if float(v).is_integer() else f"{v:.1f}" for v in x_values])
        ax.set_yticks(range(len(y_values)), [str(int(v)) if float(v).is_integer() else f"{v:.1f}" for v in y_values])
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(y_key.replace("_", " ").title())
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(value_key.replace("_", " "))
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    _heatmap(
        rows=error_rows,
        x_key="batch_size",
        y_key="sog_terms",
        value_key="force_rmse_rel",
        path=out_dir / "ablation_error_heatmap.png",
        title="RBSOG Accuracy Ablation (vs direct)",
        cmap="viridis",
    )
    _heatmap(
        rows=perf_rows,
        x_key="batch_size",
        y_key="cutoff_short",
        value_key="mean_step_time",
        path=out_dir / "ablation_speed_heatmap.png",
        title="RBSOG Speed Ablation",
        cmap="magma",
    )

    payload = {
        "error_runs_csv": "ablation_error_runs.csv",
        "performance_csv": "ablation_performance_summary.csv",
        "error_heatmap": "ablation_error_heatmap.png",
        "speed_heatmap": "ablation_speed_heatmap.png",
    }
    save_json(out_dir / "ablation_summary.json", payload)
    return payload


def run_profile_study(
    out_dir: Path,
    *,
    seeds: list[int],
    steps: int,
    sample_interval: int,
    grid_size: int,
    batch_size: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_config = SimulationConfig(steps=steps, dt=0.002, sample_interval=sample_interval)
    component_rows: list[dict[str, Any]] = []

    for solver_name in ["pppm", "rbsog"]:
        aggregated: dict[str, list[float]] = {}
        for seed in seeds:
            system = build_membrane_proxy_system(
                n_lipids=BASE_LIPIDS,
                n_solvent=BASE_SOLVENT,
                box=BASE_BOX,
                temperature=1.0,
                seed=seed,
            )
            solver = create_solver(
                solver_name,
                grid_shape=(grid_size, grid_size, grid_size),
                batch_size=batch_size,
                cutoff_short=1.2,
                sog_terms=12,
                neighbor_skin=0.3,
                neighbor_rebuild_interval=10,
                use_numba=True,
                profile=True,
            )
            result = run_simulation(system=system, solver=solver, config=sim_config, rng=np.random.default_rng(seed + 55_000))
            run_dir = out_dir / solver_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_records_csv(run_dir / "metrics.csv", result["records"])
            for row in result["records"]:
                for key, value in row.items():
                    if not str(key).startswith("diag_"):
                        continue
                    metric = str(key)[5:]
                    if not metric.endswith("_time"):
                        continue
                    aggregated.setdefault(metric, []).append(float(value))

        total_key = "pppm_total_time" if solver_name == "pppm" else "rbsog_total_time"
        total_mean = float(np.mean(aggregated.get(total_key, [float("nan")])) )
        for key, values in sorted(aggregated.items()):
            mean_value = float(np.mean(values))
            component_rows.append(
                {
                    "solver": solver_name,
                    "component": key,
                    "mean_time": mean_value,
                    "share_of_solver_total": mean_value / max(total_mean, 1e-12),
                }
            )

    _write_csv(out_dir / "profile_breakdown.csv", component_rows)

    solver_components: dict[str, list[dict[str, Any]]] = {}
    for row in component_rows:
        solver_components.setdefault(str(row["solver"]), []).append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x_positions = np.arange(len(solver_components), dtype=float)
    bottoms = np.zeros_like(x_positions)
    component_names = sorted({str(r["component"]) for r in component_rows})
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(component_names)))
    for color, component in zip(colors, component_names):
        values = []
        for solver in sorted(solver_components.keys()):
            subset = [r for r in solver_components[solver] if str(r["component"]) == component]
            values.append(float(subset[0]["mean_time"]) if subset else 0.0)
        ax.bar(x_positions, values, bottom=bottoms, label=component, color=color)
        bottoms += np.asarray(values, dtype=float)
    ax.set_xticks(x_positions, [solver.upper() for solver in sorted(solver_components.keys())])
    ax.set_ylabel("Mean component time per sampled step (s)")
    ax.set_title("Solver Profiling Breakdown")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "profile_stacked_bar.png", dpi=220)
    plt.close(fig)

    payload = {
        "breakdown_csv": "profile_breakdown.csv",
        "stacked_bar_plot": "profile_stacked_bar.png",
    }
    save_json(out_dir / "profile_summary.json", payload)
    return payload


def run_robust_stats_study(out_dir: Path, *, benchmark_runs_csv: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(benchmark_runs_csv)
    pppm_step = [float(r["mean_step_time"]) for r in rows if str(r["solver"]) == "pppm"]
    rbsog_step = [float(r["mean_step_time"]) for r in rows if str(r["solver"]) == "rbsog"]
    pppm_var = [float(r["pressure_variance"]) for r in rows if str(r["solver"]) == "pppm"]
    rbsog_var = [float(r["pressure_variance"]) for r in rows if str(r["solver"]) == "rbsog"]

    step_ratio = bootstrap_mean_ratio_ci(rbsog_step, pppm_step)
    var_ratio = bootstrap_mean_ratio_ci(rbsog_var, pppm_var)
    step_perm = permutation_test_mean_difference(pppm_step, rbsog_step)
    var_perm = permutation_test_mean_difference(pppm_var, rbsog_var)
    step_disp = permutation_test_dispersion(pppm_step, rbsog_step)
    var_disp = permutation_test_dispersion(pppm_var, rbsog_var)

    table_rows = [
        {
            "metric": "step_time",
            "pppm_mean": mean_and_std(pppm_step)[0],
            "rbsog_mean": mean_and_std(rbsog_step)[0],
            "ratio_point": step_ratio.point_estimate,
            "ratio_ci_lower": step_ratio.lower,
            "ratio_ci_upper": step_ratio.upper,
            "perm_p_value": step_perm.p_value,
            "dispersion_p_value": step_disp.p_value,
        },
        {
            "metric": "pressure_variance",
            "pppm_mean": mean_and_std(pppm_var)[0],
            "rbsog_mean": mean_and_std(rbsog_var)[0],
            "ratio_point": var_ratio.point_estimate,
            "ratio_ci_lower": var_ratio.lower,
            "ratio_ci_upper": var_ratio.upper,
            "perm_p_value": var_perm.p_value,
            "dispersion_p_value": var_disp.p_value,
        },
    ]
    _write_csv(out_dir / "robust_stats_summary.csv", table_rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = ["Step time ratio", "Variance ratio"]
    points = [step_ratio.point_estimate, var_ratio.point_estimate]
    lowers = [step_ratio.point_estimate - step_ratio.lower, var_ratio.point_estimate - var_ratio.lower]
    uppers = [step_ratio.upper - step_ratio.point_estimate, var_ratio.upper - var_ratio.point_estimate]
    ax.errorbar(labels, points, yerr=np.vstack([lowers, uppers]), fmt="o", capsize=6, color="#0f766e")
    ax.axhline(1.0, color="#7c2d12", linestyle="--", linewidth=1.2)
    ax.set_ylabel("RBSOG / PPPM ratio")
    ax.set_title("Bootstrap 95% CI for Main Benchmark Ratios")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_ratio_plot.png", dpi=220)
    plt.close(fig)

    payload = {
        "summary_csv": "robust_stats_summary.csv",
        "bootstrap_plot": "bootstrap_ratio_plot.png",
    }
    save_json(out_dir / "robust_stats_summary.json", payload)
    return payload


def run_thesis_suite(out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)

    fairness = run_pppm_grid_fairness_study(
        out_dir=out_root / "pppm_grid_fairness",
        grid_sizes=[16, 24, 32, 48, 64],
        runtime_seeds=[42, 43, 44],
        runtime_steps=600,
        accuracy_seeds=[42, 43, 44],
        accuracy_spec=SystemSpec("small", 32, 128, _scaled_box(160)),
        warmup_steps=120,
    )
    chosen_grid = int(fairness["recommended_grid"])

    main_benchmark_dir = out_root / "main_benchmark_10seeds"
    main_benchmark = run_benchmark(
        out_dir=main_benchmark_dir,
        solver_names=["pppm", "rbsog"],
        seeds=list(range(42, 52)),
        sim_config=SimulationConfig(steps=1000, dt=0.002, sample_interval=20),
        n_lipids=BASE_LIPIDS,
        n_solvent=BASE_SOLVENT,
        box=BASE_BOX,
        init_temperature=1.0,
        grid_shape=(chosen_grid, chosen_grid, chosen_grid),
        batch_size=100,
        cutoff_short=1.2,
        sog_terms=12,
        neighbor_skin=0.3,
        neighbor_rebuild_interval=10,
        use_numba=True,
    )

    batch_sweep = run_batch_size_sweep(
        out_dir=out_root / "batch_sweep",
        batch_sizes=[50, 100, 200, 400],
        seeds=[42, 43, 44],
        sim_config=SimulationConfig(steps=1000, dt=0.002, sample_interval=20),
        n_lipids=BASE_LIPIDS,
        n_solvent=BASE_SOLVENT,
        box=BASE_BOX,
        init_temperature=1.0,
        grid_shape=(chosen_grid, chosen_grid, chosen_grid),
        cutoff_short=1.2,
        sog_terms=12,
        neighbor_skin=0.3,
        neighbor_rebuild_interval=10,
        use_numba=True,
        objective_time_weight=2.0,
        objective_variance_weight=1.0,
    )
    recommended_batch = int((batch_sweep.get("recommended_batch") or {}).get("batch_size", 100))

    jit_on_dir = out_root / "jit_ablation_on"
    jit_off_dir = out_root / "jit_ablation_off"
    run_benchmark(
        out_dir=jit_on_dir,
        solver_names=["rbsog"],
        seeds=[42, 43],
        sim_config=SimulationConfig(steps=400, dt=0.002, sample_interval=10),
        n_lipids=BASE_LIPIDS,
        n_solvent=BASE_SOLVENT,
        box=BASE_BOX,
        init_temperature=1.0,
        grid_shape=(chosen_grid, chosen_grid, chosen_grid),
        batch_size=recommended_batch,
        cutoff_short=1.2,
        sog_terms=12,
        neighbor_skin=0.3,
        neighbor_rebuild_interval=10,
        use_numba=True,
    )
    run_benchmark(
        out_dir=jit_off_dir,
        solver_names=["rbsog"],
        seeds=[42, 43],
        sim_config=SimulationConfig(steps=400, dt=0.002, sample_interval=10),
        n_lipids=BASE_LIPIDS,
        n_solvent=BASE_SOLVENT,
        box=BASE_BOX,
        init_temperature=1.0,
        grid_shape=(chosen_grid, chosen_grid, chosen_grid),
        batch_size=recommended_batch,
        cutoff_short=1.2,
        sog_terms=12,
        neighbor_skin=0.3,
        neighbor_rebuild_interval=10,
        use_numba=False,
    )

    accuracy = run_accuracy_reference_study(
        out_dir=out_root / "accuracy_reference",
        specs=[
            SystemSpec("N160", 32, 128, _scaled_box(160)),
            SystemSpec("N320", 64, 256, _scaled_box(320)),
            SystemSpec("N640", 128, 512, _scaled_box(640)),
        ],
        snapshot_seeds=[42, 43, 44],
        warmup_steps=120,
        pppm_grid=chosen_grid,
        batch_sizes=[25, 50, 100, 200, 400],
        sog_terms_list=[6, 12, 18, 24],
        cutoff_short=1.2,
    )

    scaling = run_scaling_study(
        out_dir=out_root / "scaling",
        specs=[
            SystemSpec("N320", 64, 256, _scaled_box(320)),
            SystemSpec("N640", 128, 512, _scaled_box(640)),
            SystemSpec("N960", 192, 768, _scaled_box(960)),
            SystemSpec("N1280", 256, 1024, _scaled_box(1280)),
        ],
        seeds=[42, 43, 44],
        steps=400,
        sample_interval=20,
        grid_size=chosen_grid,
        batch_size=recommended_batch,
    )

    long_stability = run_long_stability_study(
        out_dir=out_root / "long_stability",
        seeds=[42, 43, 44, 45, 46],
        steps=5000,
        sample_interval=50,
        grid_size=chosen_grid,
        batch_size=recommended_batch,
    )

    ablation = run_rbsog_ablation_study(
        out_dir=out_root / "rbsog_ablation",
        accuracy_spec=SystemSpec("N320", 64, 256, _scaled_box(320)),
        snapshot_seeds=[42, 43, 44],
        warmup_steps=120,
        pppm_grid=chosen_grid,
        batch_sizes=[25, 50, 100, 200, 400],
        sog_terms_list=[6, 12, 18, 24],
        cutoff_values=[0.8, 1.2, 1.6],
        performance_seeds=[42, 43],
    )

    profile = run_profile_study(
        out_dir=out_root / "profiling",
        seeds=[42, 43, 44],
        steps=300,
        sample_interval=10,
        grid_size=chosen_grid,
        batch_size=recommended_batch,
    )

    robust_stats = run_robust_stats_study(
        out_dir=out_root / "robust_stats",
        benchmark_runs_csv=main_benchmark_dir / "benchmark_runs.csv",
    )

    manifest = {
        "pppm_grid_fairness": fairness,
        "chosen_pppm_grid": chosen_grid,
        "main_benchmark_dir": str(main_benchmark_dir.relative_to(out_root.parent)),
        "batch_sweep_dir": str((out_root / "batch_sweep").relative_to(out_root.parent)),
        "recommended_batch": recommended_batch,
        "jit_ablation_on_dir": str(jit_on_dir.relative_to(out_root.parent)),
        "jit_ablation_off_dir": str(jit_off_dir.relative_to(out_root.parent)),
        "accuracy": accuracy,
        "scaling": scaling,
        "long_stability": long_stability,
        "ablation": ablation,
        "profile": profile,
        "robust_stats": robust_stats,
    }
    save_json(out_root / "suite_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run thesis-oriented experiment suite")
    parser.add_argument("--out-root", type=Path, default=Path("results/thesis_suite"))
    args = parser.parse_args(argv)
    manifest = run_thesis_suite(args.out_root)
    print(json.dumps(manifest, indent=2))
