#!/usr/bin/env bash
# Auto-restart para o cross-fit de peso (§1: a absinto mata job longo por
# SIGKILL externo). O script e idempotente: releu as unidades ja gravadas.
set -u
PY="${PY:-.ABSvenv/bin/python3}"
OUT="${OUT:-results/fusion_weight_crossfit.csv}"
for attempt in $(seq 1 12); do
    echo "=== tentativa $attempt ==="
    "$PY" experiments/evaluate_fusion_weight_crossfit.py --output "$OUT" "$@" && {
        echo "=== concluido na tentativa $attempt ==="; exit 0; }
    echo "=== saiu com $?; reiniciando em 10 s ==="
    sleep 10
done
echo "=== desistiu apos 12 tentativas ==="; exit 1
