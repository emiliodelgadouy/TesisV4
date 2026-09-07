#!/usr/bin/env bash
# Corre multirun.ipynb en una sesion Colab con GPU A100.
# El CONFIG y los modos viven en el notebook; --backbone pisa MODELS
# en una copia temporal (el ipynb original no se toca).
#
# Usa la misma autenticacion que `colab` (oauth2 por default en esta version).
# Override: COLAB_AUTH=adc ./run_multirun_a100.sh
#
# Ejemplos:
#   ./run_multirun_a100.sh
#   ./run_multirun_a100.sh --backbone chexnet
#   ./run_multirun_a100.sh --backbone chexnet,vgg19
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
BACKBONES=()
KNOWN_BACKBONES=(
  customtiny
  efficientnetb0 efficientnetb1 efficientnetb2 efficientnetb3
  efficientnetb4 efficientnetb5 efficientnetb6 efficientnetb7
  efficientnetv2b0 efficientnetv2b1 efficientnetv2b2 efficientnetv2b3
  efficientnetv2s efficientnetv2m efficientnetv2l
  chexnet vgg16 vgg19
)
TMP_NB_DIR=""

usage() {
  cat <<EOF
Uso: $(basename "$0") [opciones]

  --session NAME     Nombre de la sesion Colab (default: ${SESSION})
  --gpu TYPE         GPU Colab (default: ${GPU})
  --auth MODE        oauth2 o adc. Default: el del CLI (oauth2)
  --timeout SEC      Timeout por celda de colab exec (default: ${TIMEOUT})
  --backbone NAME    Solo este backbone (repetible o separado por comas).
                     Default: el MODELS del notebook
  --keep             No apaga la VM al terminar (sigue consumiendo compute)
  -h, --help         Esta ayuda

Backbones: ${KNOWN_BACKBONES[*]}

Ejecuta ${NOTEBOOK} celda por celda. El output queda en
multirun_output.ipynb junto al notebook.
EOF
}

add_backbones() {
  local raw="$1"
  local item trimmed
  IFS=',' read -ra item <<< "${raw}"
  for trimmed in "${item[@]}"; do
    trimmed="${trimmed#"${trimmed%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    trimmed="${trimmed,,}"
    if [[ -z "${trimmed}" ]]; then
      continue
    fi
    BACKBONES+=("${trimmed}")
  done
}

is_known_backbone() {
  local name="$1"
  local known
  for known in "${KNOWN_BACKBONES[@]}"; do
    if [[ "${known}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
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
    --backbone)
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "--backbone necesita un nombre (ej. --backbone chexnet)" >&2
        exit 1
      fi
      add_backbones "$2"
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

if [[ ${#BACKBONES[@]} -gt 0 ]]; then
  for name in "${BACKBONES[@]}"; do
    if ! is_known_backbone "${name}"; then
      echo "Backbone desconocido: ${name}" >&2
      echo "Opciones: ${KNOWN_BACKBONES[*]}" >&2
      exit 1
    fi
  done
fi

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

patch_notebook_backbones() {
  local src="$1"
  local dst="$2"
  local models_csv="$3"
  python3 - "${src}" "${dst}" "${models_csv}" <<'PY'
import json
import sys

src, dst, models_csv = sys.argv[1:4]
models = [name for name in models_csv.split(",") if name]
nb = json.loads(open(src, encoding="utf-8").read())
override = f"MODELS = {models!r}  # --backbone\n"
needle = "CONFIG = {"
found = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    text = "".join(cell.get("source") or [])
    if "MODELS = [" not in text or needle not in text:
        continue
    if override not in text:
        text = text.replace(needle, override + needle, 1)
        lines = text.split("\n")
        if text.endswith("\n"):
            cell["source"] = [line + "\n" for line in lines[:-1]]
        else:
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    found = True
    break
if not found:
    sys.exit("No se encontro la celda MODELS/CONFIG en el notebook")
open(dst, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
PY
}

if [[ ${#BACKBONES[@]} -gt 0 ]]; then
  TMP_NB_DIR="$(mktemp -d "${TMPDIR:-/tmp}/multirun.XXXXXX")"
  models_csv="$(IFS=','; echo "${BACKBONES[*]}")"
  patch_notebook_backbones "${NOTEBOOK}" "${TMP_NB_DIR}/multirun.ipynb" "${models_csv}"
  NOTEBOOK="${TMP_NB_DIR}/multirun.ipynb"
fi

colab_cmd() {
  if [[ -n "${AUTH}" ]]; then
    colab --auth="${AUTH}" "$@"
  else
    colab "$@"
  fi
}

if [[ ${#BACKBONES[@]} -gt 0 ]]; then
  echo "[colab] sesion=${SESSION} gpu=${GPU} timeout=${TIMEOUT}s notebook=${NOTEBOOK} backbones=${BACKBONES[*]}"
else
  echo "[colab] sesion=${SESSION} gpu=${GPU} timeout=${TIMEOUT}s notebook=${NOTEBOOK}"
fi
colab_cmd new -s "${SESSION}" --gpu "${GPU}"

stop_session() {
  if [[ "${KEEP}" -eq 0 ]]; then
    echo "[colab] apagando sesion ${SESSION}"
    colab_cmd stop -s "${SESSION}" || true
  else
    echo "[colab] sesion ${SESSION} sigue viva. Parala con: colab stop -s ${SESSION}"
  fi
}

cleanup() {
  local patched_output=""
  if [[ -n "${TMP_NB_DIR}" ]]; then
    patched_output="${TMP_NB_DIR}/multirun_output.ipynb"
    if [[ -f "${patched_output}" ]]; then
      mv -f "${patched_output}" "${ROOT}/multirun_output.ipynb"
      echo "[colab] output -> ${ROOT}/multirun_output.ipynb"
    fi
    rm -rf "${TMP_NB_DIR}"
  fi
  stop_session
}
trap cleanup EXIT

colab_cmd exec -s "${SESSION}" -f "${NOTEBOOK}" --timeout "${TIMEOUT}"
