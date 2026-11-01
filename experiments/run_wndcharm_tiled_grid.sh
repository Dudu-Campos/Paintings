#!/usr/bin/env bash
# Extracao tiled do WND-CHARM (protocolo Shamir et al. 2010) nos datasets que
# faltam: rijksdraw, food101, places_lt, nct_crc, minc2500. O multitask ja foi
# extraido antes; o oilpainting roda em processo separado (ja em andamento
# quando este driver foi escrito) -- este script NAO o inclui, pra nao
# duplicar trabalho no mesmo output-dir.
#
# Sequencial, um dataset por vez (evita 2x 96 workers disputando os mesmos
# nucleos). Resumivel: o extrator ja pula shard existente.
#
# Uso:
#   setsid nohup bash experiments/run_wndcharm_tiled_grid.sh \
#       > logs/wndcharm_tiled_grid_driver.log 2>&1 < /dev/null &

set -u
cd "$(dirname "$0")/.."

PY=.venv-rtx3090-py312/bin/python3
WORKERS=${WORKERS:-96}
LOG_DIR=logs
ATTEMPTS=3

say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*" | tee -a "$LOG_DIR/wndcharm_tiled_grid_driver.log"; }

# dataset -> --image-dir (mais --exclude-stems quando aplicavel)
dataset_args() {
  case "$1" in
    rijksdraw)   echo "--image-dir data/rijksmuseum/drawing --exclude-stems RP-T-1940-571" ;;
    food101)     echo "--image-dir data/food101/food-101/images" ;;
    places_lt)   echo "--image-dir data/places_lt/images" ;;
    nct_crc)     echo "--image-dir data/nct_crc/images" ;;
    minc2500)    echo "--image-dir data/minc2500/minc-2500/images" ;;
    *) echo "unknown dataset: $1" >&2; return 2 ;;
  esac
}

run_one() {
  local dataset=$1
  local args
  args=$(dataset_args "$dataset") || return $?
  local out="outputs/${dataset}_wndcharm_tiled"
  local marker="$out/features.npy"

  if [ -f "$marker" ]; then
    say "skip $dataset (ja tem features.npy)"
    return 0
  fi

  local log="$LOG_DIR/wndcharm_tiled_${dataset}.log"
  for attempt in $(seq 1 "$ATTEMPTS"); do
    say "start $dataset attempt=$attempt (workers=$WORKERS)"
    # shellcheck disable=SC2086
    "$PY" src/extractors/extract_wndcharm_tiled_features.py \
      --dataset "$dataset" --workers "$WORKERS" $args >> "$log" 2>&1
    if [ -f "$marker" ]; then
      say "ok $dataset"
      return 0
    fi
    say "failed $dataset attempt=$attempt (ver $log)"
    sleep 30
  done
  say "GAVE UP $dataset"
  return 1
}

main() {
  say "esperando oilpainting standalone e untar terminarem..."
  while pgrep -f "extract_wndcharm_tiled_features.py --dataset oilpainting" > /dev/null \
        || pgrep -f "tar xzf data/food101\|unzip -q data/nct_crc" > /dev/null; do
    sleep 30
  done
  say "livre -- comecando o grid"

  : > "$LOG_DIR/wndcharm_tiled_grid_failures.tsv"
  for dataset in rijksdraw food101 places_lt nct_crc minc2500; do
    run_one "$dataset" || echo "$dataset" >> "$LOG_DIR/wndcharm_tiled_grid_failures.tsv"
  done

  if [ -s "$LOG_DIR/wndcharm_tiled_grid_failures.tsv" ]; then
    say "FALHAS:"
    cat "$LOG_DIR/wndcharm_tiled_grid_failures.tsv" | tee -a "$LOG_DIR/wndcharm_tiled_grid_driver.log"
    return 1
  fi
  say "FIM -- todos os datasets extraidos"
}

main
