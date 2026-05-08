from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rbsog_md.analysis import export_stability_artifacts
from rbsog_md.forces import DirectCoulombSolver, PPPMSolver, RBSOGConfig, RBSOGSolver
from rbsog_md.simulation import SimulationConfig, run_simulation, write_records_csv
from rbsog_md.system import build_membrane_proxy_system
from rbsog_md.utils import save_json


def create_solver(
    solver_name: str,
    grid_shape: tuple[int, int, int] = (32, 32, 32),
    batch_size: int = 100,
    cutoff_short: float = 1.2,
    sog_terms: int = 12,
    neighbor_skin: float = 0.3,
    neighbor_rebuild_interval: int = 10,
    use_numba: bool = True,
    profile: bool = False,
) -> Any:
    name = solver_name.lower().strip()
    if name == "direct":
        return DirectCoulombSolver(cutoff=None, profile=profile)
    if name == "pppm":
        return PPPMSolver(grid_shape=grid_shape, profile=profile)
    if name == "rbsog":
        config = RBSOGConfig(
            batch_size=batch_size,
            cutoff_short=cutoff_short,
            sog_terms=sog_terms,
            neighbor_skin=neighbor_skin,
            neighbor_rebuild_interval=neighbor_rebuild_interval,
            use_numba=use_numba,
            profile=profile,
        )
        return RBSOGSolver(config=config)
    raise ValueError(f"Unsupported solver: {solver_name}")


def run_benchmark(
    out_dir: Path,
    solver_names: list[str],
    seeds: list[int],
    sim_config: SimulationConfig,
    n_lipids: int,
    n_solvent: int,
    box: tuple[float, float, float],
    init_temperature: float,
    grid_shape: tuple[int, int, int],
    batch_size: int,
    cutoff_short: float,
    sog_terms: int,
    neighbor_skin: float,
    neighbor_rebuild_interval: int,
    use_numba: bool,
    profile: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs: list[dict[str, float | str]] = []

    for solver_name in solver_names:
        solver_dir = out_dir / solver_name
        solver_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            system = build_membrane_proxy_system(
                n_lipids=n_lipids,
                n_solvent=n_solvent,
                box=box,
                temperature=init_temperature,
                seed=seed,
            )
            solver = create_solver(
                solver_name=solver_name,
                grid_shape=grid_shape,
                batch_size=batch_size,
                cutoff_short=cutoff_short,
                sog_terms=sog_terms,
                neighbor_skin=neighbor_skin,
                neighbor_rebuild_interval=neighbor_rebuild_interval,
                use_numba=use_numba,
                profile=profile,
            )

            result = run_simulation(
                system=system,
                solver=solver,
                config=sim_config,
                rng=np.random.default_rng(seed + 10_000),
            )

            run_dir = solver_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_records_csv(run_dir / "metrics.csv", result["records"])
            save_json(run_dir / "summary.json", result["summary"])

            stability_summary = export_stability_artifacts(
                records=result["records"],
                out_dir=run_dir,
                title=f"{solver_name.upper()} seed={seed} stability",
            )

            if hasattr(solver, "kernel_report"):
                save_json(run_dir / "kernel_report.json", solver.kernel_report())

            row = {
                "solver": solver_name,
                "seed": float(seed),
                **result["summary"],
                **stability_summary,
            }
            all_runs.append(row)

    write_table_csv(out_dir / "benchmark_runs.csv", all_runs)

    aggregated = aggregate_benchmark_rows(all_runs)
    save_json(out_dir / "benchmark_summary.json", aggregated)
    write_table_csv(out_dir / "benchmark_summary.csv", aggregated["by_solver"])

    return {
        "runs": all_runs,
        "summary": aggregated,
    }


def run_batch_size_sweep(
    out_dir: Path,
    batch_sizes: list[int],
    seeds: list[int],
    sim_config: SimulationConfig,
    n_lipids: int,
    n_solvent: int,
    box: tuple[float, float, float],
    init_temperature: float,
    grid_shape: tuple[int, int, int],
    cutoff_short: float,
    sog_terms: int,
    neighbor_skin: float,
    neighbor_rebuild_interval: int,
        use_numba: bool,
        objective_time_weight: float,
        objective_variance_weight: float,
        profile: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_batch_sizes = sorted({int(v) for v in batch_sizes if int(v) > 0})
    if not cleaned_batch_sizes:
        raise ValueError("batch_sizes must contain at least one positive integer")

    if objective_time_weight < 0.0 or objective_variance_weight < 0.0:
        raise ValueError("objective weights must be non-negative")

    weight_sum = objective_time_weight + objective_variance_weight
    if weight_sum <= 0.0:
        raise ValueError("at least one objective weight must be positive")

    time_weight = float(objective_time_weight / weight_sum)
    variance_weight = float(objective_variance_weight / weight_sum)

    baseline_result = run_benchmark(
        out_dir=out_dir / "baseline_pppm",
        solver_names=["pppm"],
        seeds=seeds,
        sim_config=sim_config,
        n_lipids=n_lipids,
        n_solvent=n_solvent,
        box=box,
        init_temperature=init_temperature,
        grid_shape=grid_shape,
        batch_size=cleaned_batch_sizes[0],
        cutoff_short=cutoff_short,
        sog_terms=sog_terms,
        neighbor_skin=neighbor_skin,
        neighbor_rebuild_interval=neighbor_rebuild_interval,
        use_numba=use_numba,
        profile=profile,
    )
    baseline_row = _find_solver_row(baseline_result["summary"]["by_solver"], "pppm")
    baseline_step_time = float(baseline_row["mean_step_time"])
    baseline_variance = float(baseline_row["pressure_variance"])

    sweep_rows: list[dict[str, float | str]] = []

    for batch_size in cleaned_batch_sizes:
        single_result = run_benchmark(
            out_dir=out_dir / f"batch_{batch_size}",
            solver_names=["rbsog"],
            seeds=seeds,
            sim_config=sim_config,
            n_lipids=n_lipids,
            n_solvent=n_solvent,
            box=box,
            init_temperature=init_temperature,
            grid_shape=grid_shape,
            batch_size=batch_size,
            cutoff_short=cutoff_short,
            sog_terms=sog_terms,
            neighbor_skin=neighbor_skin,
            neighbor_rebuild_interval=neighbor_rebuild_interval,
            use_numba=use_numba,
            profile=profile,
        )
        rbsog_row = _find_solver_row(single_result["summary"]["by_solver"], "rbsog")

        rbsog_step_time = float(rbsog_row["mean_step_time"])
        rbsog_variance = float(rbsog_row["pressure_variance"])

        sweep_rows.append(
            {
                "batch_size": float(batch_size),
                "mean_step_time": rbsog_step_time,
                "pressure_variance": rbsog_variance,
                "speedup_vs_pppm": float(baseline_step_time / max(rbsog_step_time, 1e-12)),
                "variance_ratio_vs_pppm": float(rbsog_variance / max(baseline_variance, 1e-12)),
            }
        )

    pareto_mask = _pareto_mask(sweep_rows)
    utopia_scores = _utopia_scores(
        sweep_rows,
        time_weight=time_weight,
        variance_weight=variance_weight,
    )
    for idx, row in enumerate(sweep_rows):
        row["is_pareto"] = float(1.0 if pareto_mask[idx] else 0.0)
        row["utopia_score"] = float(utopia_scores[idx])
        row["recommended"] = float(0.0)

    recommended_idx = _recommend_batch_index(sweep_rows, pareto_mask, utopia_scores)
    if recommended_idx is not None:
        sweep_rows[recommended_idx]["recommended"] = float(1.0)

    write_table_csv(out_dir / "batch_sweep_summary.csv", sweep_rows)
    write_paper_table_csv(
        path=out_dir / "batch_sweep_paper_table.csv",
        rows=sweep_rows,
    )
    plot_batch_pareto(
        rows=sweep_rows,
        baseline_step_time=baseline_step_time,
        baseline_variance=baseline_variance,
        recommended_batch_size=(
            int(sweep_rows[recommended_idx]["batch_size"]) if recommended_idx is not None else None
        ),
        path=out_dir / "batch_sweep_pareto.png",
    )

    recommended_payload: dict[str, float] | None = None
    if recommended_idx is not None:
        row = sweep_rows[recommended_idx]
        recommended_payload = {
            "batch_size": float(row["batch_size"]),
            "mean_step_time": float(row["mean_step_time"]),
            "pressure_variance": float(row["pressure_variance"]),
            "speedup_vs_pppm": float(row["speedup_vs_pppm"]),
            "variance_ratio_vs_pppm": float(row["variance_ratio_vs_pppm"]),
            "utopia_score": float(row["utopia_score"]),
        }

    payload = {
        "baseline_pppm": {
            "mean_step_time": baseline_step_time,
            "pressure_variance": baseline_variance,
        },
        "objective_weights": {
            "time": time_weight,
            "variance": variance_weight,
        },
        "rows": sweep_rows,
        "recommended_batch": recommended_payload,
        "pareto_plot": "batch_sweep_pareto.png",
        "paper_table_csv": "batch_sweep_paper_table.csv",
    }
    save_json(out_dir / "batch_sweep_summary.json", payload)
    return payload


def _find_solver_row(rows: list[dict[str, float | str]], solver_name: str) -> dict[str, float | str]:
    for row in rows:
        if str(row["solver"]) == solver_name:
            return row
    raise ValueError(f"Solver '{solver_name}' not found in summary rows")


def plot_batch_pareto(
    rows: list[dict[str, float | str]],
    baseline_step_time: float,
    baseline_variance: float,
    recommended_batch_size: int | None,
    path: Path,
) -> None:
    if not rows:
        return

    x = [float(r["mean_step_time"]) for r in rows]
    y = [float(r["pressure_variance"]) for r in rows]
    labels = [int(r["batch_size"]) for r in rows]

    pareto_points = sorted(
        [
            (float(r["mean_step_time"]), float(r["pressure_variance"]))
            for r in rows
            if float(r.get("is_pareto", 0.0)) > 0.5
        ],
        key=lambda t: t[0],
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(x, y, s=48, color="#0f766e", alpha=0.85, label="RBSOG")
    if pareto_points:
        px = [p[0] for p in pareto_points]
        py = [p[1] for p in pareto_points]
        ax.plot(px, py, color="#155e75", linewidth=1.5, alpha=0.9, label="Pareto front")

    for xv, yv, batch in zip(x, y, labels):
        ax.annotate(f"P={batch}", (xv, yv), textcoords="offset points", xytext=(6, 5), fontsize=9)

    ax.scatter(
        [baseline_step_time],
        [baseline_variance],
        s=130,
        marker="*",
        color="#b45309",
        label="PPPM baseline",
    )

    if recommended_batch_size is not None:
        for row in rows:
            if int(float(row["batch_size"])) != int(recommended_batch_size):
                continue
            rx = float(row["mean_step_time"])
            ry = float(row["pressure_variance"])
            ax.scatter([rx], [ry], s=90, marker="D", color="#be123c", label="Recommended P")
            ax.annotate(
                f"Recommended P={recommended_batch_size}",
                (rx, ry),
                textcoords="offset points",
                xytext=(8, -14),
                fontsize=9,
                color="#be123c",
            )
            break

    ax.set_xlabel("Mean Step Time (s)")
    ax.set_ylabel("Pressure Variance")
    ax.set_title("RBSOG Batch Sweep Pareto Front")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _pareto_mask(rows: list[dict[str, float | str]]) -> np.ndarray:
    if not rows:
        return np.zeros((0,), dtype=bool)

    time_values = np.array([float(r["mean_step_time"]) for r in rows], dtype=float)
    variance_values = np.array([float(r["pressure_variance"]) for r in rows], dtype=float)
    n = len(rows)
    mask = np.ones((n,), dtype=bool)

    for i in range(n):
        dominates_i = (
            (time_values <= time_values[i])
            & (variance_values <= variance_values[i])
            & ((time_values < time_values[i]) | (variance_values < variance_values[i]))
        )
        dominates_i[i] = False
        if np.any(dominates_i):
            mask[i] = False

    return mask


def _utopia_scores(
    rows: list[dict[str, float | str]],
    time_weight: float,
    variance_weight: float,
) -> np.ndarray:
    if not rows:
        return np.zeros((0,), dtype=float)

    time_values = np.array([float(r["mean_step_time"]) for r in rows], dtype=float)
    variance_values = np.array([float(r["pressure_variance"]) for r in rows], dtype=float)

    time_norm = (time_values - np.min(time_values)) / (np.ptp(time_values) + 1e-12)
    var_norm = (variance_values - np.min(variance_values)) / (np.ptp(variance_values) + 1e-12)
    return np.sqrt(
        time_weight * time_norm * time_norm
        + variance_weight * var_norm * var_norm
    )


def _recommend_batch_index(
    rows: list[dict[str, float | str]],
    pareto_mask: np.ndarray,
    utopia_scores: np.ndarray,
) -> int | None:
    if not rows or pareto_mask.size == 0:
        return None

    candidate_indices = np.where(pareto_mask)[0]
    if candidate_indices.size == 0:
        return None

    best_idx = int(candidate_indices[0])
    best_score = float(utopia_scores[best_idx])
    best_batch = int(float(rows[best_idx]["batch_size"]))

    for idx in candidate_indices[1:]:
        idx_int = int(idx)
        score = float(utopia_scores[idx_int])
        batch = int(float(rows[idx_int]["batch_size"]))
        if score < best_score - 1e-12:
            best_idx = idx_int
            best_score = score
            best_batch = batch
            continue
        if abs(score - best_score) <= 1e-12 and batch < best_batch:
            best_idx = idx_int
            best_score = score
            best_batch = batch

    return best_idx


def aggregate_benchmark_rows(rows: list[dict[str, float | str]]) -> dict[str, Any]:
    by_solver: list[dict[str, float | str]] = []
    solver_names = sorted({str(row["solver"]) for row in rows})

    for solver in solver_names:
        subset = [row for row in rows if row["solver"] == solver]
        mean_step_time = float(np.mean([float(r["mean_step_time"]) for r in subset]))
        pressure_variance = float(np.mean([float(r["pressure_variance"]) for r in subset]))
        temperature_mean = float(np.mean([float(r["temperature_mean"]) for r in subset]))
        energy_drift = float(np.mean([float(r["energy_drift_per_time"]) for r in subset]))
        area_per_lipid_std = float(np.mean([float(r["area_per_lipid_std"]) for r in subset]))
        thickness_proxy_std = float(np.mean([float(r["thickness_proxy_std"]) for r in subset]))
        area_per_lipid_drift = float(
            np.mean([float(r["area_per_lipid_drift_per_time"]) for r in subset])
        )
        thickness_proxy_drift = float(
            np.mean([float(r["thickness_proxy_drift_per_time"]) for r in subset])
        )
        by_solver.append(
            {
                "solver": solver,
                "n_runs": float(len(subset)),
                "mean_step_time": mean_step_time,
                "pressure_variance": pressure_variance,
                "temperature_mean": temperature_mean,
                "energy_drift_per_time": energy_drift,
                "area_per_lipid_std": area_per_lipid_std,
                "thickness_proxy_std": thickness_proxy_std,
                "area_per_lipid_drift_per_time": area_per_lipid_drift,
                "thickness_proxy_drift_per_time": thickness_proxy_drift,
            }
        )

    return {
        "n_total_runs": float(len(rows)),
        "by_solver": by_solver,
    }


def write_table_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_paper_table_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "mean_step_time_s",
        "speedup_vs_pppm",
        "pressure_variance",
        "variance_ratio_vs_pppm",
        "is_pareto",
        "is_recommended",
        "utopia_score",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "batch_size": int(float(row["batch_size"])),
                    "mean_step_time_s": f"{float(row['mean_step_time']):.8f}",
                    "speedup_vs_pppm": f"{float(row['speedup_vs_pppm']):.6f}",
                    "pressure_variance": f"{float(row['pressure_variance']):.8e}",
                    "variance_ratio_vs_pppm": f"{float(row['variance_ratio_vs_pppm']):.8f}",
                    "is_pareto": int(float(row.get("is_pareto", 0.0)) > 0.5),
                    "is_recommended": int(float(row.get("recommended", 0.0)) > 0.5),
                    "utopia_score": f"{float(row.get('utopia_score', float('nan'))):.8f}",
                }
            )
