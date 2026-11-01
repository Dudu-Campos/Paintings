#!/usr/bin/env bash
# Food-101. Auto-restart: a absinto mata jobs longos por SIGKILL externo.
#
# SKIP_MULTISCALE=1: Multiscale foi aposentado em 24/ago (nao se paga no MLP e
# ainda impede o grafo de pagar), e custa ~60% do tempo de extracao. O marcador
# de conclusao e o Fourier, que e o ultimo passo global.
set -u
cd "$(dirname "$0")/.."
IMG=${1:-/tmp/prism_data/food-101/images}
export SKIP_MULTISCALE=1
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt" | tee -a logs/food101_extraction.log
  bash experiments/run_dataset_extraction.sh food101 "$IMG" >> logs/food101_driver.log 2>&1
  [ -f results/fourier_food101_features.parquet ] && {
    echo "=== [$(date +%H:%M:%S)] completo" | tee -a logs/food101_extraction.log; break; }
  sleep 15
done
