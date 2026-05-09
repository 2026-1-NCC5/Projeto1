#!/usr/bin/env bash
# Entrypoint parametrico. Chaveamento por $RUN_MODE.
#
# Modos suportados:
#   collector  -> roda coletor de metricas em loop infinito
#   dashboard  -> sobe o Streamlit
#   trainer    -> aceita comando arbitrario apos `--` (ex.: `python -m app.training.train_regression`)
#   shell      -> abre bash (debug)

set -euo pipefail

MODE="${RUN_MODE:-collector}"

echo "[entrypoint] RUN_MODE=${MODE}"

case "${MODE}" in
  collector)
    exec python -m app.collector.collect_metrics
    ;;
  dashboard)
    exec streamlit run app/dashboard/dashboard.py \
      --server.port "${DASHBOARD_PORT:-8501}" \
      --server.address "${DASHBOARD_HOST:-0.0.0.0}" \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  trainer)
    if [[ $# -eq 0 ]]; then
      echo "[entrypoint] modo trainer requer comando, ex.: 'python -m app.training.train_regression'"
      exit 64
    fi
    exec "$@"
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    echo "[entrypoint] RUN_MODE desconhecido: ${MODE}"
    exit 64
    ;;
esac
