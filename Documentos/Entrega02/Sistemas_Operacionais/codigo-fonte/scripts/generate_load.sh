#!/usr/bin/env bash
# Wrapper simples sobre app.collector.load_generator. Aceita o mesmo set de patterns.
#
# Uso: bash scripts/generate_load.sh [pattern] [duracao_segundos]

set -euo pipefail

PATTERN="${1:-mixed}"
DURATION="${2:-600}"

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[generate_load] pattern=${PATTERN} duration=${DURATION}s"
exec "${PYTHON_BIN}" -m app.collector.load_generator --pattern "${PATTERN}" --duration "${DURATION}"
