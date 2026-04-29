# RBSOG Personal-Computer Benchmark Protocol

This protocol is designed for a personal workstation (including Apple Silicon) to benchmark PPPM vs RBSOG in a membrane-inspired proxy system.

## 1. Environment setup

```bash
cd /Users/azuma/Downloads/thesis
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional (recommended): enable numba JIT acceleration
pip install -e ".[performance]"
```

## 2. Single-solver sanity run

Run a short simulation to verify the pipeline and output files.

```bash
source .venv/bin/activate
rbsog-md run \
  --solver rbsog \
  --steps 200 \
  --sample-interval 10 \
  --n-lipids 128 \
  --n-solvent 512 \
  --batch-size 100 \
  --out results/sanity_rbsog
```

Expected output files:

- `metrics.csv`
- `summary.json`
- `run_config.json`
- `kernel_report.json` (RBSOG only)
- `stability_timeseries.csv`
- `stability_summary.json`
- `stability_timeseries.png`

## 3. PPPM vs RBSOG benchmark

Use identical system and simulation settings for fair comparison.

```bash
source .venv/bin/activate
rbsog-md benchmark \
  --solvers pppm,rbsog \
  --steps 1000 \
  --sample-interval 20 \
  --seeds 3 \
  --n-lipids 128 \
  --n-solvent 512 \
  --batch-size 100 \
  --grid 32 32 32 \
  --out results/benchmark_main
```

## 4. Sweep RBSOG batch size

Evaluate speed-variance tradeoff.

```bash
source .venv/bin/activate
rbsog-md sweep-batch \
  --steps 1000 \
  --sample-interval 20 \
  --seeds 3 \
  --batch-sizes 50,100,200,400 \
  --objective-time-weight 2 \
  --objective-variance-weight 1 \
  --out results/batch_sweep_main
```

Key output files:

- `batch_sweep_summary.csv`
- `batch_sweep_summary.json`
- `batch_sweep_pareto.png`
- `batch_sweep_paper_table.csv`

The JSON file also stores an automatically selected recommendation:

- `recommended_batch.batch_size`
- `recommended_batch.utopia_score`
- `recommended_batch.speedup_vs_pppm`
- `recommended_batch.variance_ratio_vs_pppm`

Selection rule: choose from Pareto-front points and minimize distance to the normalized utopia point.
The two objective weights are normalized internally from:

- `objective_time_weight`
- `objective_variance_weight`

Use larger `objective_time_weight` to prefer faster runs, or larger `objective_variance_weight` to prefer lower-variance runs.

## 5. Core metrics to report

- Mean step wall-time (`mean_step_time`)
- Pressure variance (`pressure_variance`)
- Temperature mean and std
- Energy drift proxy (`energy_drift_per_time`)
- Membrane structural stability:
  - `area_per_lipid` time series and drift
  - `thickness_proxy` time series and drift

## 6. Notes and interpretation

- This prototype uses reduced units and a membrane-inspired proxy system.
- It is for algorithmic benchmarking, not full all-atom force-field validation.
- For publication-level claims, repeat with larger systems and longer trajectories.
- For figure generation, open `notebooks/rbsog_benchmark_template.ipynb`; it auto-discovers the latest benchmark, sweep, and stability outputs and renders publication-style plots.

## 7. Build LaTeX report

```bash
cd /Users/azuma/Downloads/thesis
bash docs/report/build_report.sh
```

Outputs:

- `docs/report/main.pdf`
- `docs/report/generated/benchmark_table.tex`
- `docs/report/generated/sweep_table.tex`
- `docs/report/generated/solver_compare.png`

Additional conference-style assets:

- `docs/report/generated/benchmark_stats_table.tex`
- `docs/report/generated/effect_size_table.tex`
- `docs/report/generated/config_table.tex`
- `docs/report/generated/jit_ablation_table.tex`
- `docs/report/generated/seed_scatter.png`
- `docs/report/generated/sweep_sensitivity.png`
- `docs/report/generated/jit_ablation.png`

## 8. Optional JIT ablation refresh

```bash
source .venv/bin/activate
rbsog-md benchmark \
  --solvers rbsog \
  --steps 400 \
  --sample-interval 10 \
  --seeds 2 \
  --batch-size 100 \
  --out results/ablation_jit_on_smoke

rbsog-md benchmark \
  --solvers rbsog \
  --steps 400 \
  --sample-interval 10 \
  --seeds 2 \
  --batch-size 100 \
  --disable-numba \
  --out results/ablation_jit_off_smoke

bash docs/report/build_report.sh
```
