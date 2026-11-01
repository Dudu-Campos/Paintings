#!/usr/bin/env bash
# Auto-restart (§1: jobs longos morrem por SIGKILL sem traceback). O driver
# interno pula todo JSON que ja existe, entao reiniciar nao refaz trabalho.
set -u
cd /exp/duducampos/Paintings
LOG=logs/clip_probe_arms.log
for t in $(seq 1 12); do
  echo "=== tentativa ${t} em $(date '+%H:%M:%S') ===" >> "$LOG"
  bash experiments/run_clip_probe_arms.sh >> "$LOG" 2>&1
  grep -q "^== fim" "$LOG" && { echo "=== concluido $(date '+%H:%M:%S') ===" >> "$LOG"; exit 0; }
  sleep 15
done
echo "=== DESISTIU ===" >> "$LOG"
