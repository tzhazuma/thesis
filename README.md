# RBSOG-MD Minimal Prototype

This repository contains a minimal, reproducible Python prototype for benchmarking long-range electrostatics solvers in a membrane-inspired system.

Implemented solvers:

- Direct pairwise Coulomb reference (`O(N^2)`, small-system validation only)
- PPPM-like particle-mesh baseline (FFT-based)
- RBSOG (Random Batch + Sum-of-Gaussians long-range approximation)

## Goals

- Reproduce a fair PPPM vs RBSOG comparison on a personal computer.
- Track wall-clock performance, pressure variance, and basic membrane stability proxies.
- Keep architecture compatible with future migration to production engines.

## Quick start

```bash
cd /Users/azuma/Downloads/thesis
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: enable numba JIT acceleration for short-range neighbor solver
pip install -e ".[performance]"

# Single run
rbsog-md run --solver rbsog --steps 200 --batch-size 100 --out results/single_rbsog

# PPPM vs RBSOG benchmark
rbsog-md benchmark --steps 500 --seeds 3 --batch-size 100 --out results/bench_small

# Sweep RBSOG batch size and export Pareto front
rbsog-md sweep-batch --steps 500 --seeds 3 --batch-sizes 50,100,200,400 --out results/batch_sweep

# Weighted recommendation: prioritize runtime over variance (2:1)
rbsog-md sweep-batch \
	--steps 500 \
	--seeds 3 \
	--batch-sizes 50,100,200,400 \
	--objective-time-weight 2 \
	--objective-variance-weight 1 \
	--out results/batch_sweep_weighted
```

`sweep-batch` now performs automatic Pareto-front selection and reports a recommended batch size `P`.
You can bias recommendation behavior via `--objective-time-weight` and `--objective-variance-weight`.

See `docs/experiment_protocol.md` for a full benchmark workflow and `notebooks/rbsog_benchmark_template.ipynb` for result visualization.

## Notes

- The current system builder creates a membrane-inspired proxy model, not an all-atom DPPC force field.
- Units are reduced units for algorithm benchmarking.
- This code is a research prototype, not a production MD engine.

## Outputs

- `metrics.csv`: sampled trajectory metrics (pressure, energy, area-per-lipid, thickness proxy)
- `summary.json`: run-level aggregated statistics
- `stability_timeseries.csv`: structural stability time series export
- `stability_summary.json`: one-click statistics for membrane stability
- `stability_timeseries.png`: area/thickness time series chart
- `batch_sweep_summary.csv` + `batch_sweep_pareto.png`: batch-size scan and Pareto front
- `batch_sweep_summary.json`: includes `recommended_batch` with auto-selected `P`
- `batch_sweep_paper_table.csv`: paper-ready sweep table with Pareto/recommended markers

## LaTeX report

Generate report assets from current results and compile PDF:

```bash
cd /Users/azuma/Downloads/thesis
bash docs/report/build_report.sh
```

Output PDF:

- `docs/report/main.pdf`

Conference-style generated assets (auto-created under `docs/report/generated`):

- `benchmark_table.tex`
- `benchmark_stats_table.tex`
- `effect_size_table.tex`
- `config_table.tex`
- `sweep_table.tex`
- `jit_ablation_table.tex`
- `solver_compare.png`
- `seed_scatter.png`
- `sweep_pareto.png`
- `sweep_sensitivity.png`
- `stability_timeseries.png`
- `jit_ablation.png`

Optional JIT ablation data refresh before building report:

```bash
source .venv/bin/activate
rbsog-md benchmark --solvers rbsog --steps 400 --sample-interval 10 --seeds 2 --batch-size 100 --out results/ablation_jit_on_smoke
rbsog-md benchmark --solvers rbsog --steps 400 --sample-interval 10 --seeds 2 --batch-size 100 --disable-numba --out results/ablation_jit_off_smoke
bash docs/report/build_report.sh
```
