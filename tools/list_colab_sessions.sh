#!/usr/bin/env bash
# Lista sesiones Colab activas y si ya tienen las imagenes en disco.
# Usa `colab ls` (Contents API), no `colab exec`: no interrumpe un training.
#
# Rutas: DatasetConfig con cwd=/content
#   /content/mammo/raw/images/square_images.tar.gz
#   /content/mammo/raw/images/.extract_complete
#
# Override: COLAB_AUTH=adc ./list_colab_sessions.sh
set -euo pipefail

AUTH="${COLAB_AUTH:-}"
TAR_PATH="content/mammo/raw/images/square_images.tar.gz"
MARKER_PATH="content/mammo/raw/images/.extract_complete"

usage() {
  cat <<EOF
Uso: $(basename "$0") [opciones]

  --auth MODE    oauth2 o adc. Default: el del CLI (oauth2)
  -h, --help     Esta ayuda

Lista sesiones activas en el servidor y, para las que tienen estado local,
si el tar de imagenes esta descargado y extraido.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --auth)
      AUTH="$2"
      shift 2
      ;;
    *)
      echo "Opcion desconocida: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v colab >/dev/null 2>&1; then
  echo "No esta el CLI 'colab' en PATH." >&2
  exit 1
fi

colab_cmd() {
  if [[ -n "${AUTH}" ]]; then
    colab --auth="${AUTH}" "$@"
  else
    colab "$@"
  fi
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

# True si el Contents API ve ese path como archivo (no toca el kernel).
has_remote_file() {
  local session="$1"
  local path="$2"
  local base="${path##*/}"
  local out
  if ! out="$(colab_cmd ls -s "${session}" "${path}" 2>/dev/null)"; then
    return 1
  fi
  grep -Fq "${base}" <<<"${out}"
}

images_status() {
  local session="$1"
  if has_remote_file "${session}" "${MARKER_PATH}"; then
    printf 'si (extraidas)'
  elif has_remote_file "${session}" "${TAR_PATH}"; then
    printf 'tar (sin extraer)'
  else
    printf 'no'
  fi
}

sessions_out="$(colab_cmd sessions)"
if grep -Fq "No active sessions found on server" <<<"${sessions_out}"; then
  echo "No hay sesiones activas en el servidor."
  exit 0
fi

declare -A HW=()
declare -A VARIANT=()
declare -A STATUS=()
names=()

while IFS= read -r line; do
  [[ "${line}" =~ ^\[([^]]+)\] ]] || continue
  name="${BASH_REMATCH[1]}"
  names+=("${name}")
  if [[ "${line}" =~ Hardware:\ ([^|]+) ]]; then
    HW["${name}"]="$(trim "${BASH_REMATCH[1]}")"
  else
    HW["${name}"]="?"
  fi
  if [[ "${line}" =~ Variant:\ ([^|]+) ]]; then
    VARIANT["${name}"]="$(trim "${BASH_REMATCH[1]}")"
  else
    VARIANT["${name}"]="?"
  fi
done <<<"${sessions_out}"

if [[ ${#names[@]} -eq 0 ]]; then
  echo "No pude parsear sesiones. Salida cruda:"
  printf '%s\n' "${sessions_out}"
  exit 1
fi

status_out="$(colab_cmd status 2>/dev/null || true)"
while IFS= read -r line; do
  [[ "${line}" =~ ^\[([^]]+)\] ]] || continue
  name="${BASH_REMATCH[1]}"
  if [[ "${line}" =~ Status:\ (.+)$ ]]; then
    raw="$(trim "${BASH_REMATCH[1]}")"
    STATUS["${name}"]="${raw%% *}"
  fi
done <<<"${status_out}"

printf '%-22s %-8s %-8s %-10s %s\n' "SESION" "HW" "VARIANTE" "ESTADO" "IMAGENES"
printf '%-22s %-8s %-8s %-10s %s\n' "------" "--" "--------" "------" "--------"

for name in "${names[@]}"; do
  if [[ "${name}" == "?" ]]; then
    imgs="sin estado local"
    st="${STATUS[${name}]:--}"
  else
    imgs="$(images_status "${name}")"
    st="${STATUS[${name}]:--}"
  fi
  printf '%-22s %-8s %-8s %-10s %s\n' \
    "${name}" "${HW[${name}]}" "${VARIANT[${name}]}" "${st}" "${imgs}"
done
