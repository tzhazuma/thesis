from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_report_assets import (
    _copy_or_placeholder,
    _escape_tex,
    _fmt,
    _fmt_sci,
    _path_for_manifest,
    _plot_jit_ablation,
    _plot_seed_scatter,
    _plot_solver_comparison,
    _plot_sweep_sensitivity,
    _read_csv_rows,
    _read_jit_row,
    _read_json,
    _solver_stats,
    _write_effect_size_table,
    _write_jit_ablation_table,
    _write_solver_stats_table,
    _write_solver_table,
    _write_sweep_table,
)


ROW = r"\\"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _solver_row(rows: list[dict[str, str]], solver_name: str) -> dict[str, str]:
    for row in rows:
        if str(row.get("solver", "")).strip().lower() == solver_name:
            return row
    raise ValueError(f"Missing solver row for {solver_name}")


def _bold(text: str, enabled: bool) -> str:
    return f"\\textbf{{{text}}}" if enabled else text


def _newcommand(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _load_suite_paths(suite_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest = _read_json(suite_root / "suite_manifest.json")
    paths = {
        "suite_manifest": suite_root / "suite_manifest.json",
        "benchmark_summary": suite_root / "main_benchmark_10seeds/benchmark_summary.csv",
        "benchmark_runs": suite_root / "main_benchmark_10seeds/benchmark_runs.csv",
        "sweep_summary": suite_root / "batch_sweep/batch_sweep_summary.json",
        "sweep_paper_table": suite_root / "batch_sweep/batch_sweep_paper_table.csv",
        "sweep_pareto": suite_root / "batch_sweep/batch_sweep_pareto.png",
        "jit_on_summary": suite_root / "jit_ablation_on/benchmark_summary.csv",
        "jit_off_summary": suite_root / "jit_ablation_off/benchmark_summary.csv",
        "fairness_summary": suite_root / "pppm_grid_fairness/pppm_grid_summary.csv",
        "fairness_plot": suite_root / "pppm_grid_fairness/pppm_grid_tradeoff.png",
        "accuracy_summary": suite_root / "accuracy_reference/accuracy_summary.csv",
        "accuracy_batch_plot": suite_root / "accuracy_reference/accuracy_force_vs_batch.png",
        "accuracy_terms_plot": suite_root / "accuracy_reference/accuracy_force_vs_sog_terms.png",
        "scaling_summary": suite_root / "scaling/scaling_summary.csv",
        "scaling_fit": suite_root / "scaling/scaling_exponent_table.csv",
        "scaling_ratio": suite_root / "scaling/scaling_slowdown_ratio.csv",
        "scaling_step_plot": suite_root / "scaling/scaling_step_time.png",
        "scaling_ratio_plot": suite_root / "scaling/scaling_slowdown_ratio.png",
        "long_stability_summary": suite_root / "long_stability/long_stability_summary.csv",
        "rolling_area_plot": suite_root / "long_stability/rolling_area.png",
        "rolling_thickness_plot": suite_root / "long_stability/rolling_thickness.png",
        "rolling_pressure_plot": suite_root / "long_stability/rolling_pressure.png",
        "profile_summary": suite_root / "profiling/profile_breakdown.csv",
        "profile_plot": suite_root / "profiling/profile_stacked_bar.png",
        "robust_stats_summary": suite_root / "robust_stats/robust_stats_summary.csv",
        "robust_stats_plot": suite_root / "robust_stats/bootstrap_ratio_plot.png",
        "ablation_performance": suite_root / "rbsog_ablation/ablation_performance_summary.csv",
        "ablation_errors": suite_root / "rbsog_ablation/ablation_error_runs.csv",
        "ablation_error_plot": suite_root / "rbsog_ablation/ablation_error_heatmap.png",
        "ablation_speed_plot": suite_root / "rbsog_ablation/ablation_speed_heatmap.png",
    }
    return paths, manifest


def _write_config_table(path: Path, *, chosen_grid: int, recommended_batch: int) -> None:
    lines = [
        r"\begin{tabular}{ll}",
        r"\toprule",
        f"Parameter & Value {ROW}",
        r"\midrule",
        f"Primary system & lipids=128, solvent=512 {ROW}",
        f"Simulation box & $(16, 16, 20)$ reduced units {ROW}",
        f"Main benchmark & 1000 steps, sample interval 20, 10 seeds {ROW}",
        f"PPPM grid fairness choice & ${chosen_grid}^3$ {ROW}",
        f"Recommended RBSOG batch & $P={recommended_batch}$ {ROW}",
        f"Short-range cutoff & 1.2 reduced units {ROW}",
        f"Default SOG terms & $M=12$ {ROW}",
        f"Neighbor list & skin=0.3, rebuild interval=10 {ROW}",
        f"Long stability study & 5000 steps, sample interval 50, 5 seeds {ROW}",
        f"Scaling study & $N=320,640,960,1280$ with 3 seeds each {ROW}",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write_lines(path, lines)


def _write_fairness_table(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        f"Grid & Step Time (s) & Pressure Var & Force RMSE & Pressure Error & Fairness Score & Recommended {ROW}",
        r"\midrule",
    ]
    for row in rows:
        recommended = float(row["recommended"]) > 0.5
        values = [
            _bold(f"{int(float(row['grid_size']))}$^3$", recommended),
            _bold(_fmt(float(row["mean_step_time"]), 6), recommended),
            _bold(_fmt_sci(float(row["pressure_variance"]), 3), recommended),
            _bold(_fmt(float(row["force_rmse_rel"]), 3), recommended),
            _bold(_fmt(float(row["pressure_error_rel"]), 3), recommended),
            _bold(_fmt(float(row["fairness_score"]), 3), recommended),
            _bold("yes" if recommended else "no", recommended),
        ]
        lines.append(" & ".join(values) + f" {ROW}")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_accuracy_batch_table(path: Path, rows: list[dict[str, str]]) -> None:
    systems = sorted({row["system"] for row in rows})
    batches = sorted({int(float(row["batch_size"])) for row in rows if row["family"] == "rbsog_batch"})
    lines = [
        "\\begin{tabular}{l" + "r" * (len(batches) + 1) + "}",
        r"\toprule",
        "System & PPPM & " + " & ".join([f"P={batch}" for batch in batches]) + f" {ROW}",
        r"\midrule",
    ]
    for system in systems:
        pppm = next(row for row in rows if row["system"] == system and row["family"] == "pppm")
        values = [_fmt(float(pppm["force_rmse_rel"]), 3)]
        for batch in batches:
            row = next(
                item for item in rows
                if item["system"] == system
                and item["family"] == "rbsog_batch"
                and int(float(item["batch_size"])) == batch
            )
            values.append(_fmt(float(row["force_rmse_rel"]), 3))
        lines.append(f"{_escape_tex(system)} & " + " & ".join(values) + f" {ROW}")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_accuracy_terms_table(path: Path, rows: list[dict[str, str]]) -> None:
    systems = sorted({row["system"] for row in rows})
    terms = sorted({int(float(row["sog_terms"])) for row in rows if row["family"] == "rbsog_terms"})
    lines = [
        "\\begin{tabular}{l" + "r" * (len(terms) + 1) + "}",
        r"\toprule",
        "System & PPPM & " + " & ".join([f"M={term}" for term in terms]) + f" {ROW}",
        r"\midrule",
    ]
    for system in systems:
        pppm = next(row for row in rows if row["system"] == system and row["family"] == "pppm")
        values = [_fmt(float(pppm["force_rmse_rel"]), 3)]
        for term in terms:
            row = next(
                item for item in rows
                if item["system"] == system
                and item["family"] == "rbsog_terms"
                and int(float(item["sog_terms"])) == term
            )
            values.append(_fmt(float(row["force_rmse_rel"]), 3))
        lines.append(f"{_escape_tex(system)} & " + " & ".join(values) + f" {ROW}")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_scaling_table(path: Path, rows: list[dict[str, str]], ratio_rows: list[dict[str, str]]) -> None:
    ratio_by_n = {int(float(row["n_particles"])): row for row in ratio_rows}
    pppm = {int(float(row["n_particles"])): row for row in rows if row["solver"] == "pppm"}
    rbsog = {int(float(row["n_particles"])): row for row in rows if row["solver"] == "rbsog"}
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        f"$N$ & PPPM Step (s) & RBSOG Step (s) & Slowdown {ROW}",
        r"\midrule",
    ]
    for n_particles in sorted(pppm.keys()):
        lines.append(
            f"{n_particles} & {_fmt(float(pppm[n_particles]['mean_step_time']), 6)} & "
            f"{_fmt(float(rbsog[n_particles]['mean_step_time']), 6)} & "
            f"{_fmt(float(ratio_by_n[n_particles]['slowdown_ratio']), 3)} {ROW}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_scaling_fit_table(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        f"Solver & Empirical Slope & $R^2$ {ROW}",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_tex(str(row['solver']).upper())} & {_fmt(float(row['slope']), 3)} & {_fmt(float(row['r_squared']), 3)} {ROW}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_long_stability_table(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        f"Solver & Step Time (s) & Pressure Var & Area Std & Thickness Std & Area Drift & Thickness Drift {ROW}",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_tex(str(row['solver']).upper())} & {_fmt(float(row['mean_step_time']), 6)} & "
            f"{_fmt_sci(float(row['pressure_variance']), 3)} & {_fmt(float(row['area_per_lipid_std']), 4)} & "
            f"{_fmt(float(row['thickness_proxy_std']), 4)} & {_fmt(float(row['area_per_lipid_drift_per_time']), 4)} & "
            f"{_fmt(float(row['thickness_proxy_drift_per_time']), 4)} {ROW}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_profiling_table(path: Path, rows: list[dict[str, str]]) -> None:
    filtered = [row for row in rows if not str(row["component"]).endswith("_total_time")]
    ordered = sorted(filtered, key=lambda row: (row["solver"], -float(row["share_of_solver_total"])))
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        f"Solver & Component & Mean Time (s) & Share (\\%) {ROW}",
        r"\midrule",
    ]
    for row in ordered:
        lines.append(
            f"{_escape_tex(str(row['solver']).upper())} & {_escape_tex(str(row['component']))} & "
            f"{_fmt(float(row['mean_time']), 6)} & {_fmt(100.0 * float(row['share_of_solver_total']), 2)} {ROW}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_robust_stats_table(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        f"Metric & Ratio & 95\\% CI & Permutation $p$ & Dispersion $p$ {ROW}",
        r"\midrule",
    ]
    for row in rows:
        ci = f"[{_fmt(float(row['ratio_ci_lower']), 3)}, {_fmt(float(row['ratio_ci_upper']), 3)}]"
        lines.append(
            f"{_escape_tex(str(row['metric']))} & {_fmt(float(row['ratio_point']), 3)} & {ci} & "
            f"{_fmt(float(row['perm_p_value']), 4)} & {_fmt(float(row['dispersion_p_value']), 4)} {ROW}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_lines(path, lines)


def _write_ablation_summary_table(
    path: Path,
    perf_rows: list[dict[str, str]],
    error_rows: list[dict[str, str]],
) -> None:
    error_summary: dict[tuple[int, int], list[float]] = {}
    for row in error_rows:
        key = (int(float(row["batch_size"])), int(float(row["sog_terms"])))
        error_summary.setdefault(key, []).append(float(row["force_rmse_rel"]))

    best_accuracy_key = min(error_summary, key=lambda key: sum(error_summary[key]) / len(error_summary[key]))
    best_accuracy = sum(error_summary[best_accuracy_key]) / len(error_summary[best_accuracy_key])
    fastest = min(perf_rows, key=lambda row: float(row["mean_step_time"]))
    lowest_var = min(perf_rows, key=lambda row: float(row["pressure_variance"]))

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        f"Focus & Config & Step Time (s) & Pressure Var & Force RMSE {ROW}",
        r"\midrule",
        f"Best accuracy & P={best_accuracy_key[0]}, M={best_accuracy_key[1]} & -- & -- & {_fmt(best_accuracy, 3)} {ROW}",
        (
            f"Fastest runtime & P={int(float(fastest['batch_size']))}, $r_c$={_fmt(float(fastest['cutoff_short']), 1)} & "
            f"{_fmt(float(fastest['mean_step_time']), 6)} & {_fmt_sci(float(fastest['pressure_variance']), 3)} & -- {ROW}"
        ),
        (
            f"Lowest variance & P={int(float(lowest_var['batch_size']))}, $r_c$={_fmt(float(lowest_var['cutoff_short']), 1)} & "
            f"{_fmt(float(lowest_var['mean_step_time']), 6)} & {_fmt_sci(float(lowest_var['pressure_variance']), 3)} & -- {ROW}"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write_lines(path, lines)


def _write_thesis_macros(
    path: Path,
    *,
    suite_manifest: dict[str, Any],
    benchmark_rows: list[dict[str, str]],
    run_stats: dict[str, dict[str, float]],
    batch_sweep: dict[str, Any],
    fairness_rows: list[dict[str, str]],
    scaling_fit_rows: list[dict[str, str]],
    scaling_ratio_rows: list[dict[str, str]],
    long_stability_rows: list[dict[str, str]],
    profile_rows: list[dict[str, str]],
    robust_rows: list[dict[str, str]],
    jit_speedup: float,
    jit_variance_ratio: float,
) -> None:
    pppm_row = _solver_row(benchmark_rows, "pppm")
    rbsog_row = _solver_row(benchmark_rows, "rbsog")
    pppm_long = _solver_row(long_stability_rows, "pppm")
    rbsog_long = _solver_row(long_stability_rows, "rbsog")
    step_row = next(row for row in robust_rows if row["metric"] == "step_time")
    var_row = next(row for row in robust_rows if row["metric"] == "pressure_variance")
    fairness_row = next(row for row in fairness_rows if float(row["recommended"]) > 0.5)
    scaling_ratio_max = max(scaling_ratio_rows, key=lambda row: float(row["n_particles"]))
    pppm_fit = next(row for row in scaling_fit_rows if row["solver"] == "pppm")
    rbsog_fit = next(row for row in scaling_fit_rows if row["solver"] == "rbsog")

    pppm_profile = {
        row["component"]: row
        for row in profile_rows
        if row["solver"] == "pppm" and not str(row["component"]).endswith("_total_time")
    }
    rbsog_profile = {
        row["component"]: row
        for row in profile_rows
        if row["solver"] == "rbsog" and not str(row["component"]).endswith("_total_time")
    }

    recommended_batch = batch_sweep.get("recommended_batch") or {}
    baseline_pppm = batch_sweep.get("baseline_pppm") or {}
    objective_weights = batch_sweep.get("objective_weights") or {}

    lines = [
        "% Auto-generated file. Do not edit manually.",
        _newcommand("ChosenPPPMGrid", str(int(suite_manifest["chosen_pppm_grid"]))),
        _newcommand("RecommendedBatch", str(int(float(recommended_batch.get("batch_size", 0.0))))),
        _newcommand("RecommendedSpeedup", _fmt(float(recommended_batch.get("speedup_vs_pppm", float("nan"))), 3)),
        _newcommand("RecommendedVarianceRatio", _fmt(float(recommended_batch.get("variance_ratio_vs_pppm", float("nan"))), 4)),
        _newcommand("ObjectiveTimeWeight", _fmt(float(objective_weights.get("time", float("nan"))), 3)),
        _newcommand("ObjectiveVarianceWeight", _fmt(float(objective_weights.get("variance", float("nan"))), 3)),
        _newcommand("BaselinePPPMStepTime", _fmt(float(baseline_pppm.get("mean_step_time", float("nan"))), 6)),
        _newcommand("BaselinePPPMVariance", _fmt_sci(float(baseline_pppm.get("pressure_variance", float("nan"))), 3)),
        _newcommand("RBSOGSlowdownVsPPPM", _fmt(float(rbsog_row["mean_step_time"]) / max(float(pppm_row["mean_step_time"]), 1e-12), 3)),
        _newcommand("RBSOGVarianceReduction", _fmt(float(pppm_row["pressure_variance"]) / max(float(rbsog_row["pressure_variance"]), 1e-12), 3)),
        _newcommand("RBSOGVarianceRatioVsPPPM", _fmt(float(rbsog_row["pressure_variance"]) / max(float(pppm_row["pressure_variance"]), 1e-12), 3)),
        _newcommand("PPPMStepCI", _fmt(float(run_stats["pppm"]["step_ci95"]), 6)),
        _newcommand("RBSOGStepCI", _fmt(float(run_stats["rbsog"]["step_ci95"]), 6)),
        _newcommand("AreaDrift", _fmt(float(rbsog_long["area_per_lipid_drift_per_time"]), 5)),
        _newcommand("ThicknessDrift", _fmt(float(rbsog_long["thickness_proxy_drift_per_time"]), 5)),
        _newcommand("JITSpeedup", _fmt(jit_speedup, 3)),
        _newcommand("JITVarianceRatio", _fmt(jit_variance_ratio, 4)),
        _newcommand("FairnessGrid", str(int(float(fairness_row["grid_size"])))),
        _newcommand("FairnessForceRMSE", _fmt(float(fairness_row["force_rmse_rel"]), 3)),
        _newcommand("FairnessPressureError", _fmt(float(fairness_row["pressure_error_rel"]), 3)),
        _newcommand("ScalingExponentPPPM", _fmt(float(pppm_fit["slope"]), 3)),
        _newcommand("ScalingExponentRBSOG", _fmt(float(rbsog_fit["slope"]), 3)),
        _newcommand("ScalingSlowdownAtMaxN", _fmt(float(scaling_ratio_max["slowdown_ratio"]), 3)),
        _newcommand("LongSlowdownVsPPPM", _fmt(float(rbsog_long["mean_step_time"]) / max(float(pppm_long["mean_step_time"]), 1e-12), 3)),
        _newcommand("BootstrapStepRatio", _fmt(float(step_row["ratio_point"]), 3)),
        _newcommand("BootstrapStepRatioLow", _fmt(float(step_row["ratio_ci_lower"]), 3)),
        _newcommand("BootstrapStepRatioHigh", _fmt(float(step_row["ratio_ci_upper"]), 3)),
        _newcommand("BootstrapVarianceRatio", _fmt(float(var_row["ratio_point"]), 3)),
        _newcommand("BootstrapVarianceRatioLow", _fmt(float(var_row["ratio_ci_lower"]), 3)),
        _newcommand("BootstrapVarianceRatioHigh", _fmt(float(var_row["ratio_ci_upper"]), 3)),
        _newcommand("PPPMFFTShare", _fmt(100.0 * float(pppm_profile["pppm_fft_solve_time"]["share_of_solver_total"]), 1)),
        _newcommand("RBSOGSamplingShare", _fmt(100.0 * float(rbsog_profile["rbsog_sampling_time"]["share_of_solver_total"]), 1)),
        _newcommand("RBSOGLongRangeShare", _fmt(100.0 * float(rbsog_profile["rbsog_long_range_time"]["share_of_solver_total"]), 1)),
        _newcommand("RBSOGFilterShare", _fmt(100.0 * float(rbsog_profile["rbsog_filter_time"]["share_of_solver_total"]), 1)),
    ]
    _write_lines(path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis assets from thesis suite outputs")
    parser.add_argument("--suite-root", type=Path, default=Path("results/thesis_suite"))
    parser.add_argument("--output-dir", type=Path, default=Path("thesis_template/generated"))
    args = parser.parse_args()

    suite_root = args.suite_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, suite_manifest = _load_suite_paths(suite_root)
    benchmark_rows = _read_csv_rows(paths["benchmark_summary"])
    benchmark_run_rows = _read_csv_rows(paths["benchmark_runs"])
    batch_sweep = _read_json(paths["sweep_summary"])
    batch_sweep_rows = _read_csv_rows(paths["sweep_paper_table"])
    fairness_rows = _read_csv_rows(paths["fairness_summary"])
    accuracy_rows = _read_csv_rows(paths["accuracy_summary"])
    scaling_rows = _read_csv_rows(paths["scaling_summary"])
    scaling_fit_rows = _read_csv_rows(paths["scaling_fit"])
    scaling_ratio_rows = _read_csv_rows(paths["scaling_ratio"])
    long_stability_rows = _read_csv_rows(paths["long_stability_summary"])
    profile_rows = _read_csv_rows(paths["profile_summary"])
    robust_rows = _read_csv_rows(paths["robust_stats_summary"])
    ablation_perf_rows = _read_csv_rows(paths["ablation_performance"])
    ablation_error_rows = _read_csv_rows(paths["ablation_errors"])
    run_stats = _solver_stats(benchmark_run_rows)
    jit_on = _read_jit_row(paths["jit_on_summary"])
    jit_off = _read_jit_row(paths["jit_off_summary"])

    _write_solver_table(output_dir / "benchmark_table.tex", benchmark_rows)
    _write_solver_stats_table(output_dir / "benchmark_stats_table.tex", run_stats)
    _write_effect_size_table(output_dir / "effect_size_table.tex", benchmark_run_rows)
    _write_sweep_table(output_dir / "sweep_table.tex", batch_sweep_rows)
    _write_config_table(
        output_dir / "config_table.tex",
        chosen_grid=int(suite_manifest["chosen_pppm_grid"]),
        recommended_batch=int(suite_manifest["recommended_batch"]),
    )
    jit_speedup, jit_variance_ratio = _write_jit_ablation_table(
        output_dir / "jit_ablation_table.tex",
        jit_on=jit_on,
        jit_off=jit_off,
    )
    _write_fairness_table(output_dir / "fairness_table.tex", fairness_rows)
    _write_accuracy_batch_table(output_dir / "accuracy_batch_table.tex", accuracy_rows)
    _write_accuracy_terms_table(output_dir / "accuracy_terms_table.tex", accuracy_rows)
    _write_scaling_table(output_dir / "scaling_table.tex", scaling_rows, scaling_ratio_rows)
    _write_scaling_fit_table(output_dir / "scaling_fit_table.tex", scaling_fit_rows)
    _write_long_stability_table(output_dir / "long_stability_table.tex", long_stability_rows)
    _write_profiling_table(output_dir / "profiling_table.tex", profile_rows)
    _write_robust_stats_table(output_dir / "robust_stats_table.tex", robust_rows)
    _write_ablation_summary_table(output_dir / "ablation_summary_table.tex", ablation_perf_rows, ablation_error_rows)
    _write_thesis_macros(
        output_dir / "macros.tex",
        suite_manifest=suite_manifest,
        benchmark_rows=benchmark_rows,
        run_stats=run_stats,
        batch_sweep=batch_sweep,
        fairness_rows=fairness_rows,
        scaling_fit_rows=scaling_fit_rows,
        scaling_ratio_rows=scaling_ratio_rows,
        long_stability_rows=long_stability_rows,
        profile_rows=profile_rows,
        robust_rows=robust_rows,
        jit_speedup=jit_speedup,
        jit_variance_ratio=jit_variance_ratio,
    )

    _plot_solver_comparison(output_dir / "solver_compare.png", run_stats)
    _plot_seed_scatter(output_dir / "seed_scatter.png", benchmark_run_rows)
    _plot_sweep_sensitivity(output_dir / "sweep_sensitivity.png", batch_sweep_rows)
    _plot_jit_ablation(output_dir / "jit_ablation.png", jit_on=jit_on, jit_off=jit_off)

    copies = {
        "sweep_pareto.png": paths["sweep_pareto"],
        "pppm_grid_tradeoff.png": paths["fairness_plot"],
        "accuracy_force_vs_batch.png": paths["accuracy_batch_plot"],
        "accuracy_force_vs_sog_terms.png": paths["accuracy_terms_plot"],
        "scaling_step_time.png": paths["scaling_step_plot"],
        "scaling_slowdown_ratio.png": paths["scaling_ratio_plot"],
        "rolling_area.png": paths["rolling_area_plot"],
        "rolling_thickness.png": paths["rolling_thickness_plot"],
        "rolling_pressure.png": paths["rolling_pressure_plot"],
        "profile_stacked_bar.png": paths["profile_plot"],
        "bootstrap_ratio_plot.png": paths["robust_stats_plot"],
        "ablation_error_heatmap.png": paths["ablation_error_plot"],
        "ablation_speed_heatmap.png": paths["ablation_speed_plot"],
    }
    for name, src in copies.items():
        _copy_or_placeholder(src=src, dst=output_dir / name, title=name.replace("_", " ").title())

    manifest = {
        "suite_root": _path_for_manifest(suite_root),
        "sources": {
            "suite_manifest": _path_for_manifest(paths["suite_manifest"]),
            "benchmark_summary": _path_for_manifest(paths["benchmark_summary"]),
            "benchmark_runs": _path_for_manifest(paths["benchmark_runs"]),
            "sweep_summary": _path_for_manifest(paths["sweep_summary"]),
            "fairness_summary": _path_for_manifest(paths["fairness_summary"]),
            "accuracy_summary": _path_for_manifest(paths["accuracy_summary"]),
            "scaling_summary": _path_for_manifest(paths["scaling_summary"]),
            "long_stability_summary": _path_for_manifest(paths["long_stability_summary"]),
            "profile_summary": _path_for_manifest(paths["profile_summary"]),
            "robust_stats_summary": _path_for_manifest(paths["robust_stats_summary"]),
            "ablation_performance": _path_for_manifest(paths["ablation_performance"]),
            "ablation_errors": _path_for_manifest(paths["ablation_errors"]),
        },
        "generated_files": sorted(
            [
                "ablation_error_heatmap.png",
                "ablation_speed_heatmap.png",
                "ablation_summary_table.tex",
                "accuracy_batch_table.tex",
                "accuracy_force_vs_batch.png",
                "accuracy_force_vs_sog_terms.png",
                "accuracy_terms_table.tex",
                "asset_manifest.json",
                "benchmark_stats_table.tex",
                "benchmark_table.tex",
                "bootstrap_ratio_plot.png",
                "config_table.tex",
                "effect_size_table.tex",
                "fairness_table.tex",
                "jit_ablation.png",
                "jit_ablation_table.tex",
                "long_stability_table.tex",
                "macros.tex",
                "pppm_grid_tradeoff.png",
                "profile_stacked_bar.png",
                "profiling_table.tex",
                "robust_stats_table.tex",
                "rolling_area.png",
                "rolling_pressure.png",
                "rolling_thickness.png",
                "scaling_fit_table.tex",
                "scaling_slowdown_ratio.png",
                "scaling_step_time.png",
                "scaling_table.tex",
                "seed_scatter.png",
                "solver_compare.png",
                "sweep_pareto.png",
                "sweep_sensitivity.png",
                "sweep_table.tex",
            ]
        ),
    }
    (output_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
