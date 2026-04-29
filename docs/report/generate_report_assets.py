from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: float, digits: int = 6) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def _fmt_sci(value: float, digits: int = 3) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}e}"


def _fmt_pm(mean: float, std: float, digits: int = 6) -> str:
    if mean != mean or std != std:
        return "nan"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _escape_tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("&", "\\&")
    )


def _solver_order_key(name: str) -> tuple[int, str]:
    low = name.lower().strip()
    if low == "pppm":
        return (0, low)
    if low == "rbsog":
        return (1, low)
    return (2, low)


def _t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    if n > 30:
        return 1.96

    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    return table.get(n, 1.96)


def _mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return (mean, 0.0, float("nan"))

    std = float(np.std(arr, ddof=1))
    tcrit = _t_critical_95(int(arr.size))
    ci95 = float("nan") if tcrit != tcrit else float(tcrit * std / math.sqrt(float(arr.size)))
    return (mean, std, ci95)


def _cohen_d(sample_a: list[float], sample_b: list[float]) -> float:
    if len(sample_a) <= 1 or len(sample_b) <= 1:
        return float("nan")

    arr_a = np.array(sample_a, dtype=float)
    arr_b = np.array(sample_b, dtype=float)
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))
    n_a = float(arr_a.size)
    n_b = float(arr_b.size)
    pooled = math.sqrt(max(((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0), 1e-16))
    return (mean_b - mean_a) / pooled


def _solver_stats(runs_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in runs_rows:
        solver = str(row.get("solver", "")).lower().strip()
        grouped.setdefault(solver, []).append(row)

    stats: dict[str, dict[str, float]] = {}
    for solver, rows in grouped.items():
        step_values = [_to_float(r.get("mean_step_time")) for r in rows]
        var_values = [_to_float(r.get("pressure_variance")) for r in rows]
        temp_values = [_to_float(r.get("temperature_mean")) for r in rows]

        step_mean, step_std, step_ci = _mean_std_ci(step_values)
        var_mean, var_std, var_ci = _mean_std_ci(var_values)
        temp_mean, temp_std, temp_ci = _mean_std_ci(temp_values)

        stats[solver] = {
            "n": float(len(rows)),
            "step_mean": step_mean,
            "step_std": step_std,
            "step_ci95": step_ci,
            "var_mean": var_mean,
            "var_std": var_std,
            "var_ci95": var_ci,
            "temp_mean": temp_mean,
            "temp_std": temp_std,
            "temp_ci95": temp_ci,
        }

    return stats


def _write_solver_table(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("No benchmark rows available.\n", encoding="utf-8")
        return

    sorted_rows = sorted(rows, key=lambda r: _solver_order_key(str(r.get("solver", ""))))

    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Solver & Mean Step Time (s) & Pressure Variance & Temp Mean & Area Drift & Thickness Drift \\\\"  # noqa: E501
    )
    lines.append("\\midrule")

    for row in sorted_rows:
        solver = _escape_tex(str(row.get("solver", "")))
        step_time = _fmt(_to_float(row.get("mean_step_time")), 6)
        variance = _fmt_sci(_to_float(row.get("pressure_variance")), 3)
        temperature = _fmt(_to_float(row.get("temperature_mean")), 4)
        area_drift = _fmt(_to_float(row.get("area_per_lipid_drift_per_time")), 5)
        thickness_drift = _fmt(_to_float(row.get("thickness_proxy_drift_per_time")), 5)
        lines.append(
            f"{solver} & {step_time} & {variance} & {temperature} & {area_drift} & {thickness_drift} \\\\"  # noqa: E501
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_solver_stats_table(path: Path, stats: dict[str, dict[str, float]]) -> None:
    if not stats:
        path.write_text("No run-level statistics available.\n", encoding="utf-8")
        return

    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Solver & n & Step Time Mean$\\pm$Std & CI95 (Step) & Pressure Var Mean$\\pm$Std \\\\")
    lines.append("\\midrule")

    for solver in sorted(stats.keys(), key=_solver_order_key):
        st = stats[solver]
        n = int(st["n"])
        step_pm = _fmt_pm(st["step_mean"], st["step_std"], digits=6)
        step_ci = _fmt(st["step_ci95"], digits=6)
        var_pm = f"{_fmt_sci(st['var_mean'], 3)} $\\pm$ {_fmt_sci(st['var_std'], 3)}"
        lines.append(
            f"{_escape_tex(solver)} & {n} & {step_pm} & {step_ci} & {var_pm} \\\\"  # noqa: E501
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_effect_size_table(path: Path, runs_rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in runs_rows:
        solver = str(row.get("solver", "")).lower().strip()
        grouped.setdefault(solver, []).append(row)

    if "pppm" not in grouped or "rbsog" not in grouped:
        path.write_text("Insufficient solver rows for effect-size analysis.\n", encoding="utf-8")
        return

    pppm_step = [_to_float(r.get("mean_step_time")) for r in grouped["pppm"]]
    rbsog_step = [_to_float(r.get("mean_step_time")) for r in grouped["rbsog"]]
    pppm_var = [_to_float(r.get("pressure_variance")) for r in grouped["pppm"]]
    rbsog_var = [_to_float(r.get("pressure_variance")) for r in grouped["rbsog"]]

    step_mean_a, step_std_a, _ = _mean_std_ci(pppm_step)
    step_mean_b, step_std_b, _ = _mean_std_ci(rbsog_step)
    var_mean_a, var_std_a, _ = _mean_std_ci(pppm_var)
    var_mean_b, var_std_b, _ = _mean_std_ci(rbsog_var)

    d_step = _cohen_d(pppm_step, rbsog_step)
    d_var = _cohen_d(pppm_var, rbsog_var)
    rel_step = (step_mean_b - step_mean_a) / max(step_mean_a, 1e-12)
    rel_var = (var_mean_b - var_mean_a) / max(var_mean_a, 1e-12)

    lines: list[str] = []
    row_break = r"\\"
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Metric & PPPM Mean$\\pm$Std & RBSOG Mean$\\pm$Std & Cohen's $d$ & Relative Change "
        + row_break
    )
    lines.append("\\midrule")
    step_line = (
        "Step time "
        f"& {_fmt_pm(step_mean_a, step_std_a, 6)} "
        f"& {_fmt_pm(step_mean_b, step_std_b, 6)} "
        f"& {_fmt(d_step, 3)} "
        f"& {_fmt(100.0 * rel_step, 2)}\\% "
        + row_break
    )
    var_line = (
        "Pressure variance "
        f"& {_fmt_sci(var_mean_a, 3)} $\\pm$ {_fmt_sci(var_std_a, 3)} "
        f"& {_fmt_sci(var_mean_b, 3)} $\\pm$ {_fmt_sci(var_std_b, 3)} "
        f"& {_fmt(d_var, 3)} "
        f"& {_fmt(100.0 * rel_var, 2)}\\% "
        + row_break
    )
    lines.append(step_line)
    lines.append(var_line)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sweep_table(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("No sweep rows available.\n", encoding="utf-8")
        return

    def as_flag(value: str) -> bool:
        return str(value).strip() in {"1", "1.0", "true", "True"}

    lines: list[str] = []
    lines.append("\\begin{tabular}{rrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "P & Step Time (s) & Speedup vs PPPM & Variance Ratio & Utopia Score & Pareto & Recommended \\\\"  # noqa: E501
    )
    lines.append("\\midrule")

    for row in rows:
        p = int(_to_float(row.get("batch_size"), 0.0))
        step_time = _fmt(_to_float(row.get("mean_step_time_s")), 6)
        speedup = _fmt(_to_float(row.get("speedup_vs_pppm")), 4)
        variance_ratio = _fmt(_to_float(row.get("variance_ratio_vs_pppm")), 5)
        utopia = _fmt(_to_float(row.get("utopia_score")), 5)
        pareto = "yes" if as_flag(str(row.get("is_pareto", "0"))) else "no"
        recommended = "yes" if as_flag(str(row.get("is_recommended", "0"))) else "no"

        if recommended == "yes":
            lines.append(
                f"\\textbf{{{p}}} & \\textbf{{{step_time}}} & \\textbf{{{speedup}}} & \\textbf{{{variance_ratio}}} & \\textbf{{{utopia}}} & \\textbf{{{pareto}}} & \\textbf{{{recommended}}} \\\\"  # noqa: E501
            )
        else:
            lines.append(
                f"{p} & {step_time} & {speedup} & {variance_ratio} & {utopia} & {pareto} & {recommended} \\\\"  # noqa: E501
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_config_table(path: Path, run_config: dict[str, Any]) -> None:
    sim = run_config.get("sim_config", {}) if isinstance(run_config, dict) else {}
    box = run_config.get("box", [float("nan"), float("nan"), float("nan")])
    grid = run_config.get("grid", [float("nan"), float("nan"), float("nan")])

    lines = [
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Parameter & Value \\\\",
        "\\midrule",
        (
            "System size & "
            f"lipids={int(_to_float(run_config.get('n_lipids'), 0.0))}, "
            f"solvent={int(_to_float(run_config.get('n_solvent'), 0.0))} \\\\"  # noqa: E501
        ),
        f"Box (reduced) & ({_fmt(_to_float(box[0]), 2)}, {_fmt(_to_float(box[1]), 2)}, {_fmt(_to_float(box[2]), 2)}) \\\\",
        (
            "Integrator & "
            f"steps={int(_to_float(sim.get('steps'), 0.0))}, "
            f"dt={_fmt(_to_float(sim.get('dt')), 4)}, "
            f"sample={int(_to_float(sim.get('sample_interval'), 0.0))} \\\\"  # noqa: E501
        ),
        (
            "Target ensemble & "
            f"T={_fmt(_to_float(sim.get('target_temperature')), 3)}, "
            f"P={_fmt(_to_float(sim.get('target_pressure')), 3)} \\\\"  # noqa: E501
        ),
        (
            "RBSOG kernel & "
            f"P={int(_to_float(run_config.get('batch_size'), 0.0))}, "
            f"cutoff={_fmt(_to_float(run_config.get('cutoff_short')), 3)}, "
            f"terms={int(_to_float(run_config.get('sog_terms'), 0.0))} \\\\"  # noqa: E501
        ),
        (
            "Neighbor list & "
            f"skin={_fmt(_to_float(run_config.get('neighbor_skin')), 3)}, "
            f"rebuild={int(_to_float(run_config.get('neighbor_rebuild_interval'), 0.0))} \\\\"  # noqa: E501
        ),
        (
            "Mesh baseline & "
            f"grid=({int(_to_float(grid[0], 0.0))}, {int(_to_float(grid[1], 0.0))}, {int(_to_float(grid[2], 0.0))}) \\\\"  # noqa: E501
        ),
        f"JIT status & {'enabled' if bool(run_config.get('use_numba', False)) else 'disabled'} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_jit_row(path: Path | None) -> dict[str, float] | None:
    if path is None or (not path.exists()):
        return None

    rows = _read_csv_rows(path)
    for row in rows:
        if str(row.get("solver", "")).lower().strip() == "rbsog":
            return {
                "mean_step_time": _to_float(row.get("mean_step_time")),
                "pressure_variance": _to_float(row.get("pressure_variance")),
            }
    return None


def _write_jit_ablation_table(
    path: Path,
    jit_on: dict[str, float] | None,
    jit_off: dict[str, float] | None,
) -> tuple[float, float]:
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Mode & Step Time (s) & Pressure Variance & Relative Speed & Variance Ratio \\\\")
    lines.append("\\midrule")

    speedup = float("nan")
    variance_ratio = float("nan")

    if jit_on is None or jit_off is None:
        lines.append("JIT on & N/A & N/A & N/A & N/A \\\\")
        lines.append("JIT off & N/A & N/A & N/A & N/A \\\\")
    else:
        off_time = max(jit_off["mean_step_time"], 1e-12)
        on_time = max(jit_on["mean_step_time"], 1e-12)
        off_var = max(jit_off["pressure_variance"], 1e-12)
        on_var = max(jit_on["pressure_variance"], 1e-12)

        speedup = off_time / on_time
        variance_ratio = on_var / off_var

        lines.append(
            f"JIT on & {_fmt(on_time, 6)} & {_fmt_sci(on_var, 3)} & {_fmt(speedup, 3)}x & {_fmt(variance_ratio, 4)} \\\\"  # noqa: E501
        )
        lines.append(
            f"JIT off & {_fmt(off_time, 6)} & {_fmt_sci(off_var, 3)} & 1.000x & 1.0000 \\\\"  # noqa: E501
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (speedup, variance_ratio)


def _write_macros(
    path: Path,
    sweep_summary: dict[str, Any],
    stability_summary: dict[str, Any],
    benchmark_summary_rows: list[dict[str, str]],
    stats: dict[str, dict[str, float]],
    jit_speedup: float,
    jit_variance_ratio: float,
) -> None:
    recommended = sweep_summary.get("recommended_batch", {}) or {}
    objective = sweep_summary.get("objective_weights", {}) or {}
    baseline = sweep_summary.get("baseline_pppm", {}) or {}

    summary_by_solver = {
        str(row.get("solver", "")).lower().strip(): row for row in benchmark_summary_rows
    }
    pppm_row = summary_by_solver.get("pppm", {})
    rbsog_row = summary_by_solver.get("rbsog", {})

    pppm_step = _to_float(pppm_row.get("mean_step_time"))
    rbsog_step = _to_float(rbsog_row.get("mean_step_time"))
    pppm_var = _to_float(pppm_row.get("pressure_variance"))
    rbsog_var = _to_float(rbsog_row.get("pressure_variance"))

    slowdown = rbsog_step / max(pppm_step, 1e-12)
    variance_reduction = pppm_var / max(rbsog_var, 1e-12)

    pppm_stats = stats.get("pppm", {})
    rbsog_stats = stats.get("rbsog", {})

    lines = [
        "% Auto-generated file. Do not edit manually.",
        f"\\newcommand{{\\RecommendedBatch}}{{{int(_to_float(recommended.get('batch_size', 0.0), 0.0))}}}",
        f"\\newcommand{{\\RecommendedSpeedup}}{{{_fmt(_to_float(recommended.get('speedup_vs_pppm')), 3)}}}",
        f"\\newcommand{{\\RecommendedVarianceRatio}}{{{_fmt(_to_float(recommended.get('variance_ratio_vs_pppm')), 4)}}}",
        f"\\newcommand{{\\ObjectiveTimeWeight}}{{{_fmt(_to_float(objective.get('time')), 3)}}}",
        f"\\newcommand{{\\ObjectiveVarianceWeight}}{{{_fmt(_to_float(objective.get('variance')), 3)}}}",
        f"\\newcommand{{\\BaselinePPPMStepTime}}{{{_fmt(_to_float(baseline.get('mean_step_time')), 6)}}}",
        f"\\newcommand{{\\BaselinePPPMVariance}}{{{_fmt_sci(_to_float(baseline.get('pressure_variance')), 3)}}}",
        f"\\newcommand{{\\RBSOGSlowdownVsPPPM}}{{{_fmt(slowdown, 3)}}}",
        f"\\newcommand{{\\RBSOGVarianceReduction}}{{{_fmt(variance_reduction, 3)}}}",
        f"\\newcommand{{\\PPPMStepCI}}{{{_fmt(_to_float(pppm_stats.get('step_ci95')), 6)}}}",
        f"\\newcommand{{\\RBSOGStepCI}}{{{_fmt(_to_float(rbsog_stats.get('step_ci95')), 6)}}}",
        f"\\newcommand{{\\AreaDrift}}{{{_fmt(_to_float(stability_summary.get('area_per_lipid_drift_per_time')), 5)}}}",
        f"\\newcommand{{\\ThicknessDrift}}{{{_fmt(_to_float(stability_summary.get('thickness_proxy_drift_per_time')), 5)}}}",
        f"\\newcommand{{\\JITSpeedup}}{{{_fmt(jit_speedup, 3)}}}",
        f"\\newcommand{{\\JITVarianceRatio}}{{{_fmt(jit_variance_ratio, 4)}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_solver_comparison(path: Path, stats: dict[str, dict[str, float]]) -> None:
    if not stats:
        return

    solvers = sorted(stats.keys(), key=_solver_order_key)
    labels = [s.upper() for s in solvers]
    step_means = [stats[s]["step_mean"] for s in solvers]
    step_stds = [stats[s]["step_std"] for s in solvers]
    var_means = [stats[s]["var_mean"] for s in solvers]
    var_stds = [stats[s]["var_std"] for s in solvers]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    colors = ["#b45309", "#0f766e", "#334155"][: len(labels)]

    axes[0].bar(labels, step_means, yerr=step_stds, capsize=4, color=colors)
    axes[0].set_title("Mean Step Time")
    axes[0].set_ylabel("s")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(labels, var_means, yerr=var_stds, capsize=4, color=colors)
    axes[1].set_title("Pressure Variance")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_seed_scatter(path: Path, runs_rows: list[dict[str, str]]) -> None:
    if not runs_rows:
        return

    solvers = sorted(
        {str(r.get("solver", "")).lower().strip() for r in runs_rows},
        key=_solver_order_key,
    )
    if not solvers:
        return

    solver_to_x = {solver: idx for idx, solver in enumerate(solvers)}

    x_time: list[float] = []
    y_time: list[float] = []
    x_var: list[float] = []
    y_var: list[float] = []

    for row in runs_rows:
        solver = str(row.get("solver", "")).lower().strip()
        base_x = float(solver_to_x[solver])
        seed = _to_float(row.get("seed"), 0.0)
        jitter = (seed % 7.0 - 3.0) * 0.015
        x = base_x + jitter

        x_time.append(x)
        y_time.append(_to_float(row.get("mean_step_time")))
        x_var.append(x)
        y_var.append(_to_float(row.get("pressure_variance")))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].scatter(x_time, y_time, c="#0f766e", alpha=0.9)
    axes[0].set_xticks(list(solver_to_x.values()), [s.upper() for s in solvers])
    axes[0].set_title("Run-level Step Time")
    axes[0].set_ylabel("s")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].scatter(x_var, y_var, c="#b45309", alpha=0.9)
    axes[1].set_xticks(list(solver_to_x.values()), [s.upper() for s in solvers])
    axes[1].set_title("Run-level Pressure Variance")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _copy_or_placeholder(src: Path, dst: Path, title: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dst)
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Missing artifact:\\n{src}",
        ha="center",
        va="center",
        fontsize=11,
        color="#b91c1c",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(dst, dpi=200)
    plt.close(fig)


def _plot_sweep_sensitivity(path: Path, sweep_rows: list[dict[str, str]]) -> None:
    if not sweep_rows:
        _copy_or_placeholder(src=Path("__missing__"), dst=path, title="Batch-size Sensitivity")
        return

    ordered = sorted(sweep_rows, key=lambda r: int(_to_float(r.get("batch_size"), 0.0)))
    p = [int(_to_float(r.get("batch_size"), 0.0)) for r in ordered]
    step = [_to_float(r.get("mean_step_time_s")) for r in ordered]
    var_ratio = [_to_float(r.get("variance_ratio_vs_pppm")) for r in ordered]
    rec_mask = [str(r.get("is_recommended", "0")).strip() in {"1", "1.0", "true", "True"} for r in ordered]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax2 = ax1.twinx()

    line1 = ax1.plot(p, step, marker="o", color="#0f766e", label="Mean step time")
    line2 = ax2.plot(p, var_ratio, marker="s", color="#b45309", label="Variance ratio vs PPPM")

    for idx, is_rec in enumerate(rec_mask):
        if not is_rec:
            continue
        ax1.scatter([p[idx]], [step[idx]], color="#be123c", s=80, zorder=5)
        ax1.annotate(
            f"Recommended P={p[idx]}",
            (p[idx], step[idx]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color="#be123c",
        )

    ax1.set_xlabel("Batch size P")
    ax1.set_ylabel("Mean step time (s)")
    ax2.set_ylabel("Variance ratio vs PPPM")
    ax2.set_yscale("log")
    ax1.grid(alpha=0.25)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_jit_ablation(path: Path, jit_on: dict[str, float] | None, jit_off: dict[str, float] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jit_on is None or jit_off is None:
        _copy_or_placeholder(src=Path("__missing__"), dst=path, title="JIT Ablation")
        return

    labels = ["JIT on", "JIT off"]
    step = [jit_on["mean_step_time"], jit_off["mean_step_time"]]
    var = [jit_on["pressure_variance"], jit_off["pressure_variance"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].bar(labels, step, color=["#0f766e", "#b45309"])
    axes[0].set_title("RBSOG Mean Step Time")
    axes[0].set_ylabel("s")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(labels, var, color=["#0f766e", "#b45309"])
    axes[1].set_title("RBSOG Pressure Variance")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX report assets from benchmark results")
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=Path("results/bench_batch4_smoke/benchmark_summary.csv"),
    )
    parser.add_argument(
        "--benchmark-runs",
        type=Path,
        default=Path("results/bench_batch4_smoke/benchmark_runs.csv"),
    )
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=Path("results/sweep_weighted_smoke/batch_sweep_summary.json"),
    )
    parser.add_argument(
        "--stability-summary",
        type=Path,
        default=Path("results/run_batch4_smoke/stability_summary.json"),
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("results/run_batch4_smoke/run_config.json"),
    )
    parser.add_argument(
        "--sweep-pareto",
        type=Path,
        default=Path("results/sweep_weighted_smoke/batch_sweep_pareto.png"),
    )
    parser.add_argument(
        "--stability-plot",
        type=Path,
        default=Path("results/run_batch4_smoke/stability_timeseries.png"),
    )
    parser.add_argument(
        "--jit-on-summary",
        type=Path,
        default=Path("results/ablation_jit_on_smoke/benchmark_summary.csv"),
    )
    parser.add_argument(
        "--jit-off-summary",
        type=Path,
        default=Path("results/ablation_jit_off_smoke/benchmark_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/report/generated"),
    )
    args = parser.parse_args()

    benchmark_rows = _read_csv_rows(args.benchmark_summary)
    benchmark_run_rows = _read_csv_rows(args.benchmark_runs)
    sweep_summary = _read_json(args.sweep_summary)
    stability_summary = _read_json(args.stability_summary)
    run_config = _read_json(args.run_config) if args.run_config.exists() else {}

    stats = _solver_stats(benchmark_run_rows)

    paper_table_rel = str(sweep_summary.get("paper_table_csv", "batch_sweep_paper_table.csv"))
    paper_table_path = args.sweep_summary.parent / paper_table_rel
    sweep_rows = _read_csv_rows(paper_table_path)

    jit_on = _read_jit_row(args.jit_on_summary)
    jit_off = _read_jit_row(args.jit_off_summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_solver_table(output_dir / "benchmark_table.tex", benchmark_rows)
    _write_solver_stats_table(output_dir / "benchmark_stats_table.tex", stats)
    _write_effect_size_table(output_dir / "effect_size_table.tex", benchmark_run_rows)
    _write_sweep_table(output_dir / "sweep_table.tex", sweep_rows)
    _write_config_table(output_dir / "config_table.tex", run_config)

    jit_speedup, jit_variance_ratio = _write_jit_ablation_table(
        output_dir / "jit_ablation_table.tex",
        jit_on=jit_on,
        jit_off=jit_off,
    )

    _write_macros(
        output_dir / "macros.tex",
        sweep_summary=sweep_summary,
        stability_summary=stability_summary,
        benchmark_summary_rows=benchmark_rows,
        stats=stats,
        jit_speedup=jit_speedup,
        jit_variance_ratio=jit_variance_ratio,
    )

    _plot_solver_comparison(output_dir / "solver_compare.png", stats)
    _plot_seed_scatter(output_dir / "seed_scatter.png", benchmark_run_rows)
    _plot_jit_ablation(output_dir / "jit_ablation.png", jit_on=jit_on, jit_off=jit_off)
    _plot_sweep_sensitivity(output_dir / "sweep_sensitivity.png", sweep_rows)

    _copy_or_placeholder(
        src=args.sweep_pareto,
        dst=output_dir / "sweep_pareto.png",
        title="Batch Sweep Pareto",
    )
    _copy_or_placeholder(
        src=args.stability_plot,
        dst=output_dir / "stability_timeseries.png",
        title="Stability Timeseries",
    )

    manifest = {
        "benchmark_summary": str(args.benchmark_summary),
        "benchmark_runs": str(args.benchmark_runs),
        "sweep_summary": str(args.sweep_summary),
        "stability_summary": str(args.stability_summary),
        "run_config": str(args.run_config),
        "paper_table_csv": str(paper_table_path),
        "jit_on_summary": str(args.jit_on_summary),
        "jit_off_summary": str(args.jit_off_summary),
        "generated_files": [
            "benchmark_table.tex",
            "benchmark_stats_table.tex",
            "effect_size_table.tex",
            "sweep_table.tex",
            "config_table.tex",
            "jit_ablation_table.tex",
            "macros.tex",
            "solver_compare.png",
            "seed_scatter.png",
            "sweep_pareto.png",
            "sweep_sensitivity.png",
            "stability_timeseries.png",
            "jit_ablation.png",
        ],
    }
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()