#!/usr/bin/env bash
# Auto-restart (§1: a absinto mata job longo por SIGKILL externo sem traceback).
set -u
for attempt in $(seq 1 12); do
    echo "=== tentativa $attempt ==="
    bash experiments/run_fusion_text_posteriors.sh "$@" && {
        echo "=== concluido na tentativa $attempt ==="; exit 0; }
    echo "=== saiu com erro; reiniciando em 15 s ==="
    sleep 15
done
echo "=== desistiu apos 12 tentativas ==="; exit 1
