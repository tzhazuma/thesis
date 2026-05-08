#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THESIS_DIR="${ROOT_DIR}/thesis_template"
SUITE_ROOT="${ROOT_DIR}/results/thesis_suite"

if [[ -f "${THESIS_DIR}/Tex/PrivateBuildConfig.tex" ]]; then
  printf 'Refusing public build while thesis_template/Tex/PrivateBuildConfig.tex exists.\n' >&2
  exit 1
fi

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python3.12"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/docs/report/generate_thesis_assets.py" \
  --suite-root "${SUITE_ROOT}" \
  --output-dir "${THESIS_DIR}/generated"

cd "${THESIS_DIR}"
xelatex -interaction=nonstopmode -halt-on-error Thesis.tex
bibtex Thesis
xelatex -interaction=nonstopmode -halt-on-error Thesis.tex
xelatex -interaction=nonstopmode -halt-on-error Thesis.tex
