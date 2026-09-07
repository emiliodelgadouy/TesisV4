#!/usr/bin/env bash
# Corre multirun.ipynb en una sesion Colab con GPU A100.
# El CONFIG y los modos viven solo en el notebook.
#
# Usa la misma autenticacion que `colab` (oauth2 por default en esta version).
# Override: COLAB_AUTH=adc ./run_multirun_a100.sh
#
# Ejemplos:
#   ./run_multirun_a100.sh
#   ./run_multirun_a100.sh --keep
#   ./run_multirun_a100.sh --session tesis-multirun-2
#   ./run_multirun_a100.sh --timeout 172800
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK="${ROOT}/multirun.ipynb"
SESSION="${COLAB_SESSION:-tesis-multirun}"
GPU="${GPU:-A100}"
AUTH="${COLAB_AUTH:-}"
# colab exec default es 30s por celda; la descarga del tar y el training
# necesitan horas. 24h alinea con el cap de keep-alive de Colab.
TIMEOUT="${COLAB_TIMEOUT:-86400}"
KEEP=0

usage() {
  cat <<EOF
Uso: $(basename "$0") [opciones]

  --session NAME   Nombre de la sesion Colab (default: ${SESSION})
  --gpu TYPE       GPU Colab (default: ${GPU})
  --auth MODE      oauth2 o adc. Default: el del CLI (oauth2)
  --timeout SEC    Timeout por celda de colab exec (default: ${TIMEOUT})
  --keep           No apaga la VM al terminar (sigue consumiendo compute)
  -h, --help       Esta ayuda

Ejecuta ${NOTEBOOK} celda por celda. El output queda en
multirun_output.ipynb junto al notebook.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --auth)
      AUTH="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --keep)
      KEEP=1
      shift
      ;;
    *)
      echo "Opcion desconocida: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${NOTEBOOK}" ]]; then
  echo "No existe ${NOTEBOOK}" >&2
  exit 1
fi

if ! command -v colab >/dev/null 2>&1; then
  echo "No esta el CLI 'colab' en PATH." >&2
  exit 1
fi

# google-colab-cli 0.6 llama KernelClient; jupyter-kernel-client 1.0 lo renombro.
# Si no chequeamos aca, se asigna la A100 y recien ahi explota.
if ! python -c "from jupyter_kernel_client import KernelClient" >/dev/null 2>&1; then
  echo "El CLI de Colab no puede ejecutar notebooks con jupyter-kernel-client>=1." >&2
  echo "Instala el pin compatible ANTES de pedir GPU:" >&2
  echo "  pip install 'jupyter-kernel-client==0.15.0'" >&2
  exit 1
fi

colab_cmd() {
  if [[ -n "${AUTH}" ]]; then
    colab --auth="${AUTH}" "$@"
  else
    colab "$@"
  fi
}

echo "[colab] sesion=${SESSION} gpu=${GPU} timeout=${TIMEOUT}s notebook=${NOTEBOOK}"
colab_cmd new -s "${SESSION}" --gpu "${GPU}"

stop_session() {
  if [[ "${KEEP}" -eq 0 ]]; then
    echo "[colab] apagando sesion ${SESSION}"
    colab_cmd stop -s "${SESSION}" || true
  else
    echo "[colab] sesion ${SESSION} sigue viva. Parala con: colab stop -s ${SESSION}"
  fi
}
trap stop_session EXIT

colab_cmd exec -s "${SESSION}" -f "${NOTEBOOK}" --timeout "${TIMEOUT}"
