#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${ROOT_DIR}/docs/report"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python3.12"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3.12"
fi

"${PYTHON_BIN}" "${REPORT_DIR}/generate_report_assets.py" \
  --benchmark-summary "${ROOT_DIR}/results/bench_batch4_smoke/benchmark_summary.csv" \
  --benchmark-runs "${ROOT_DIR}/results/bench_batch4_smoke/benchmark_runs.csv" \
  --sweep-summary "${ROOT_DIR}/results/sweep_weighted_smoke/batch_sweep_summary.json" \
  --stability-summary "${ROOT_DIR}/results/run_batch4_smoke/stability_summary.json" \
  --run-config "${ROOT_DIR}/results/run_batch4_smoke/run_config.json" \
  --sweep-pareto "${ROOT_DIR}/results/sweep_weighted_smoke/batch_sweep_pareto.png" \
  --stability-plot "${ROOT_DIR}/results/run_batch4_smoke/stability_timeseries.png" \
  --jit-on-summary "${ROOT_DIR}/results/ablation_jit_on_smoke/benchmark_summary.csv" \
  --jit-off-summary "${ROOT_DIR}/results/ablation_jit_off_smoke/benchmark_summary.csv" \
  --output-dir "${REPORT_DIR}/generated"

cd "${REPORT_DIR}"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
