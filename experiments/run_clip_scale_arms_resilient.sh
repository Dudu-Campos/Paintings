#!/usr/bin/env bash
# Auto-restart para as duas escalas de CLIP (§1: jobs longos morrem por SIGKILL
# externo, sem traceback). O driver interno pula todo cache/JSON que ja existe,
# entao reiniciar nunca refaz passo concluido.
set -u
cd /exp/duducampos/Paintings
LOG=logs/clip_scale_arms.log
for t in $(seq 1 12); do
  echo "=== tentativa ${t} em $(date '+%H:%M:%S') ===" >> "$LOG"
  bash experiments/run_clip_scale_arms.sh >> "$LOG" 2>&1
  grep -q "^== fim" "$LOG" && { echo "=== concluido $(date '+%H:%M:%S') ===" >> "$LOG"; exit 0; }
  grep -qE "^FALHOU_" "$LOG" && sleep 20
  sleep 10
done
echo "=== DESISTIU ===" >> "$LOG"
