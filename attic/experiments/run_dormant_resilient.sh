#!/usr/bin/env bash
set -u
cd /exp/duducampos/Paintings
run () {
  local tag=$1; shift
  for attempt in $(seq 1 20); do
    echo "=== [$(date +%H:%M:%S)] $tag tentativa $attempt" | tee -a logs/dormant_revisit.log
    .ABSvenv/bin/python3 experiments/evaluate_dormant_without_multiscale.py "$@" \
        >> logs/dormant_revisit.log 2>&1
    status=$?
    echo "=== [$(date +%H:%M:%S)] $tag saiu com código $status" | tee -a logs/dormant_revisit.log
    [ "$status" -eq 0 ] && return 0
    sleep 10
  done
}
run "sem Multiscale" 
run "com Multiscale" --with-multiscale
