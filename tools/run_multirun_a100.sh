#!/usr/bin/env bash
# Corre multirun.ipynb en una sesion Colab con GPU A100 high-RAM.
# El CONFIG y los modos viven en el notebook; --backbone pisa MODELS
# en una copia temporal (el ipynb original no se toca).
#
# Siempre pide machine shape high-RAM (colab new --high-mem, o shape=hm
# si el CLI instalado todavia no tiene el flag). Requiere Colab Pro/Pro+.
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
#   ./run_multirun_a100.sh --output multirun_output_chexnet.ipynb
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
# Vacio: se arma un nombre unico por corrida (sesion + backbones + timestamp + pid).
OUTPUT="${COLAB_OUTPUT:-}"
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
                     Siempre high-RAM (A100 ~80 GB; Pro/Pro+)
  --auth MODE        oauth2 o adc. Default: el del CLI (oauth2)
  --timeout SEC      Timeout por celda de colab exec (default: ${TIMEOUT})
  --backbone NAME    Solo este backbone (repetible o separado por comas).
                     Default: el MODELS del notebook
  --output PATH      Notebook de salida. Default: un archivo unico
                     multirun_output_<sesion>[_backbones]_<fecha>_<pid>.ipynb
                     Si PATH es un directorio, el archivo unico va ahi.
                     Override: COLAB_OUTPUT=...
  --keep             No apaga la VM al terminar (sigue consumiendo compute)
  -h, --help         Esta ayuda

Backbones: ${KNOWN_BACKBONES[*]}

Ejecuta ${NOTEBOOK} celda por celda. colab exec siempre escribe
<notebook>_output.ipynb al lado del ipynb que corre; este script copia
a un temp y mueve el resultado a --output para que dos corridas no
se pisen el mismo archivo.
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

# Nombre de archivo unico: no choca si se lanza el script dos veces.
default_output_basename() {
  local stamp parts
  stamp="$(date +%Y%m%d_%H%M%S)"
  parts="${SESSION}"
  if [[ ${#BACKBONES[@]} -gt 0 ]]; then
    parts="${parts}_$(IFS=-; echo "${BACKBONES[*]}")"
  fi
  printf 'multirun_output_%s_%s_%s.ipynb' "${parts}" "${stamp}" "$$"
}

resolve_output_path() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    printf '%s/%s' "${ROOT}" "$(default_output_basename)"
    return
  fi
  if [[ "${path}" != /* ]]; then
    path="${ROOT}/${path}"
  fi
  if [[ -d "${path}" ]]; then
    printf '%s/%s' "${path%/}" "$(default_output_basename)"
    return
  fi
  printf '%s' "${path}"
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
    --output)
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "--output necesita un path (archivo o directorio)" >&2
        exit 1
      fi
      OUTPUT="$2"
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

# Siempre copiar a temp: colab exec escribe <stem>_output.ipynb al lado del
# notebook que ejecuta. Sin temp, dos corridas pisarian el mismo archivo.
TMP_NB_DIR="$(mktemp -d "${TMPDIR:-/tmp}/multirun.XXXXXX")"
if [[ ${#BACKBONES[@]} -gt 0 ]]; then
  models_csv="$(IFS=','; echo "${BACKBONES[*]}")"
  patch_notebook_backbones "${NOTEBOOK}" "${TMP_NB_DIR}/multirun.ipynb" "${models_csv}"
else
  cp -f "${NOTEBOOK}" "${TMP_NB_DIR}/multirun.ipynb"
fi
NOTEBOOK="${TMP_NB_DIR}/multirun.ipynb"
OUTPUT="$(resolve_output_path "${OUTPUT}")"
mkdir -p "$(dirname "${OUTPUT}")"

colab_cmd() {
  if [[ -n "${AUTH}" ]]; then
    colab --auth="${AUTH}" "$@"
  else
    colab "$@"
  fi
}

# google-colab-cli 0.6 no expone --high-mem; el backend acepta shape=hm
# (igual que la UI). CLI nuevos ya mandan eso con --high-mem.
enable_high_ram_assign() {
  local dir="$1"
  cat >"${dir}/sitecustomize.py" <<'PY'
"""Fuerza high-RAM en /tun/m/assign si el CLI no mando shape=hm."""
try:
    from colab_cli.client import Client
except Exception:
    pass
else:
    _orig = Client._build_assign_url

    def _build_assign_url(self, *args, **kwargs):
        url = _orig(self, *args, **kwargs)
        if url and "shape=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}shape=hm"
        return url

    Client._build_assign_url = _build_assign_url
PY
  export PYTHONPATH="${dir}${PYTHONPATH:+:${PYTHONPATH}}"
}

colab_new_supports_high_mem() {
  colab_cmd new --help 2>&1 | grep -Fq -- "--high-mem"
}

stop_session() {
  if [[ "${KEEP}" -eq 0 ]]; then
    echo "[colab] apagando sesion ${SESSION}"
    colab_cmd stop -s "${SESSION}" || true
  else
    echo "[colab] sesion ${SESSION} sigue viva. Parala con: colab stop -s ${SESSION}"
  fi
}

cleanup() {
  local exec_output="${TMP_NB_DIR}/multirun_output.ipynb"
  if [[ -n "${TMP_NB_DIR}" && -f "${exec_output}" ]]; then
    mv -f "${exec_output}" "${OUTPUT}"
    echo "[colab] output -> ${OUTPUT}"
  fi
  if [[ -n "${TMP_NB_DIR}" ]]; then
    rm -rf "${TMP_NB_DIR}"
  fi
  stop_session
}
trap cleanup EXIT

enable_high_ram_assign "${TMP_NB_DIR}"

new_args=(-s "${SESSION}" --gpu "${GPU}")
if colab_new_supports_high_mem; then
  new_args+=(--high-mem)
  high_mem_how="--high-mem"
else
  high_mem_how="shape=hm (CLI sin --high-mem)"
fi

log_bits="sesion=${SESSION} gpu=${GPU} high-ram=${high_mem_how} timeout=${TIMEOUT}s notebook=${NOTEBOOK} output=${OUTPUT}"
if [[ ${#BACKBONES[@]} -gt 0 ]]; then
  log_bits+=" backbones=${BACKBONES[*]}"
fi
echo "[colab] ${log_bits}"
colab_cmd new "${new_args[@]}"

colab_cmd exec -s "${SESSION}" -f "${NOTEBOOK}" --timeout "${TIMEOUT}"
