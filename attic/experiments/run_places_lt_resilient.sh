#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
IMG=${1:-/tmp/prism_data/places_lt/images}
export SKIP_MULTISCALE=1
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt" | tee -a logs/places_lt_extraction.log
  bash experiments/run_dataset_extraction.sh places_lt "$IMG" >> logs/places_lt_driver.log 2>&1
  [ -f results/fourier_places_lt_features.parquet ] && {
    echo "=== [$(date +%H:%M:%S)] completo" | tee -a logs/places_lt_extraction.log; break; }
  sleep 15
done
