from __future__ import annotations

import argparse
from pathlib import Path

from rbsog_md.analysis import export_stability_artifacts
from rbsog_md.benchmark import create_solver, run_batch_size_sweep, run_benchmark
from rbsog_md.simulation import SimulationConfig, run_simulation, write_records_csv
from rbsog_md.system import build_membrane_proxy_system
from rbsog_md.utils import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rbsog-md",
        description="Minimal RBSOG vs PPPM molecular dynamics benchmark",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one simulation")
    _add_common_args(run_parser)
    run_parser.add_argument("--solver", choices=["direct", "pppm", "rbsog"], default="rbsog")

    bench_parser = sub.add_parser("benchmark", help="Run multi-seed benchmark")
    _add_common_args(bench_parser)
    bench_parser.add_argument(
        "--solvers",
        default="pppm,rbsog",
        help="Comma-separated solver list (direct,pppm,rbsog)",
    )
    bench_parser.add_argument("--seeds", type=int, default=3, help="Number of consecutive seeds")

    sweep_parser = sub.add_parser("sweep-batch", help="Sweep RBSOG batch size and export Pareto")
    _add_common_args(sweep_parser)
    sweep_parser.add_argument("--seeds", type=int, default=3, help="Number of consecutive seeds")
    sweep_parser.add_argument(
        "--batch-sizes",
        default="50,100,200,400",
        help="Comma-separated batch sizes for RBSOG",
    )
    sweep_parser.add_argument(
        "--objective-time-weight",
        type=float,
        default=1.0,
        help="Weight for minimizing step time in recommended P selection",
    )
    sweep_parser.add_argument(
        "--objective-variance-weight",
        type=float,
        default=1.0,
        help="Weight for minimizing pressure variance in recommended P selection",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--sample-interval", type=int, default=10)

    parser.add_argument("--n-lipids", type=int, default=128)
    parser.add_argument("--n-solvent", type=int, default=512)
    parser.add_argument("--box", type=float, nargs=3, default=[16.0, 16.0, 20.0])

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pressure", type=float, default=1.0)
    parser.add_argument("--thermostat-tau", type=float, default=0.2)
    parser.add_argument("--barostat-tau", type=float, default=1.0)
    parser.add_argument("--compressibility", type=float, default=1e-3)

    parser.add_argument("--grid", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cutoff-short", type=float, default=1.2)
    parser.add_argument("--sog-terms", type=int, default=12)
    parser.add_argument("--neighbor-skin", type=float, default=0.3)
    parser.add_argument("--neighbor-rebuild-interval", type=int, default=10)
    parser.add_argument("--disable-numba", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/run"))


def _build_simulation_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        steps=args.steps,
        dt=args.dt,
        sample_interval=args.sample_interval,
        target_temperature=args.temperature,
        thermostat_tau=args.thermostat_tau,
        target_pressure=args.pressure,
        barostat_tau=args.barostat_tau,
        compressibility=args.compressibility,
    )


def _as_float3(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("Expected three float values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _as_int3(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("Expected three integer values")
    return (int(values[0]), int(values[1]), int(values[2]))


def _parse_int_list(raw: str) -> list[int]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer value")
    return [int(item) for item in items]


def run_single(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    box = _as_float3(args.box)
    grid = _as_int3(args.grid)

    system = build_membrane_proxy_system(
        n_lipids=args.n_lipids,
        n_solvent=args.n_solvent,
        box=box,
        temperature=args.temperature,
        seed=args.seed,
    )
    solver = create_solver(
        solver_name=args.solver,
        grid_shape=grid,
        batch_size=args.batch_size,
        cutoff_short=args.cutoff_short,
        sog_terms=args.sog_terms,
        neighbor_skin=args.neighbor_skin,
        neighbor_rebuild_interval=args.neighbor_rebuild_interval,
        use_numba=not args.disable_numba,
    )
    sim_config = _build_simulation_config(args)

    result = run_simulation(system=system, solver=solver, config=sim_config)

    write_records_csv(args.out / "metrics.csv", result["records"])
    save_json(args.out / "summary.json", result["summary"])
    stability_summary = export_stability_artifacts(
        records=result["records"],
        out_dir=args.out,
        title=f"{args.solver.upper()} seed={args.seed} stability",
    )
    save_json(
        args.out / "run_config.json",
        {
            "solver": args.solver,
            "seed": args.seed,
            "n_lipids": args.n_lipids,
            "n_solvent": args.n_solvent,
            "box": [float(x) for x in box],
            "sim_config": sim_config,
            "grid": [int(x) for x in grid],
            "batch_size": args.batch_size,
            "cutoff_short": args.cutoff_short,
            "sog_terms": args.sog_terms,
            "neighbor_skin": args.neighbor_skin,
            "neighbor_rebuild_interval": args.neighbor_rebuild_interval,
            "use_numba": (not args.disable_numba),
            "stability_summary": stability_summary,
        },
    )
    if hasattr(solver, "kernel_report"):
        save_json(args.out / "kernel_report.json", solver.kernel_report())

    print("Run finished")
    print(f"solver: {args.solver}")
    print(f"mean_step_time: {result['summary']['mean_step_time']:.6f} s")
    print(f"pressure_variance: {result['summary']['pressure_variance']:.6f}")


def run_multi_benchmark(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    sim_config = _build_simulation_config(args)
    box = _as_float3(args.box)
    grid = _as_int3(args.grid)

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    seeds = list(range(args.seed, args.seed + args.seeds))

    result = run_benchmark(
        out_dir=args.out,
        solver_names=solvers,
        seeds=seeds,
        sim_config=sim_config,
        n_lipids=args.n_lipids,
        n_solvent=args.n_solvent,
        box=box,
        init_temperature=args.temperature,
        grid_shape=grid,
        batch_size=args.batch_size,
        cutoff_short=args.cutoff_short,
        sog_terms=args.sog_terms,
        neighbor_skin=args.neighbor_skin,
        neighbor_rebuild_interval=args.neighbor_rebuild_interval,
        use_numba=not args.disable_numba,
    )

    print("Benchmark finished")
    for row in result["summary"]["by_solver"]:
        print(
            f"{row['solver']}: mean_step_time={row['mean_step_time']:.6f} s, "
            f"pressure_variance={row['pressure_variance']:.6f}"
        )


def run_batch_sweep(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    sim_config = _build_simulation_config(args)
    box = _as_float3(args.box)
    grid = _as_int3(args.grid)

    batch_sizes = _parse_int_list(args.batch_sizes)
    seeds = list(range(args.seed, args.seed + args.seeds))

    result = run_batch_size_sweep(
        out_dir=args.out,
        batch_sizes=batch_sizes,
        seeds=seeds,
        sim_config=sim_config,
        n_lipids=args.n_lipids,
        n_solvent=args.n_solvent,
        box=box,
        init_temperature=args.temperature,
        grid_shape=grid,
        cutoff_short=args.cutoff_short,
        sog_terms=args.sog_terms,
        neighbor_skin=args.neighbor_skin,
        neighbor_rebuild_interval=args.neighbor_rebuild_interval,
        use_numba=not args.disable_numba,
        objective_time_weight=args.objective_time_weight,
        objective_variance_weight=args.objective_variance_weight,
    )

    print("Batch sweep finished")
    print(
        "PPPM baseline: "
        f"mean_step_time={result['baseline_pppm']['mean_step_time']:.6f} s, "
        f"pressure_variance={result['baseline_pppm']['pressure_variance']:.6f}"
    )
    print(
        "Objective weights: "
        f"time={result['objective_weights']['time']:.3f}, "
        f"variance={result['objective_weights']['variance']:.3f}"
    )
    for row in result["rows"]:
        print(
            f"P={int(row['batch_size'])}: mean_step_time={row['mean_step_time']:.6f} s, "
            f"pressure_variance={row['pressure_variance']:.6f}, "
            f"speedup_vs_pppm={row['speedup_vs_pppm']:.3f}"
        )
    recommended = result.get("recommended_batch")
    if isinstance(recommended, dict):
        print(
            "Recommended P: "
            f"P={int(recommended['batch_size'])}, "
            f"utopia_score={recommended['utopia_score']:.4f}, "
            f"speedup_vs_pppm={recommended['speedup_vs_pppm']:.3f}, "
            f"variance_ratio_vs_pppm={recommended['variance_ratio_vs_pppm']:.4f}"
        )
    print(f"Paper table CSV: {args.out / 'batch_sweep_paper_table.csv'}")
    print(f"Pareto plot: {args.out / 'batch_sweep_pareto.png'}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_single(args)
        return
    if args.command == "benchmark":
        run_multi_benchmark(args)
        return
    if args.command == "sweep-batch":
        run_batch_sweep(args)
        return

    parser.error("Unknown command")
