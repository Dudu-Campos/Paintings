#!/usr/bin/env bash
# Wrapper de auto-restart para os controles de captioner (§1: jobs longos morrem
# por SIGKILL externo, sem traceback). O driver interno ja e idempotente por
# braco -- ele pula o que tem `<ds>_decisions.npy` em disco --, entao reiniciar
# nunca refaz trabalho concluido. Um braco morto no meio recomeca do fold 1,
# que e o custo aceito: o build_text_scores nao tem saida incremental por fold.
set -u
cd /exp/duducampos/Paintings
LOG=logs/captioner_controls.log
for tentativa in $(seq 1 12); do
  echo "=== tentativa ${tentativa} em $(date '+%H:%M:%S') ===" >> "$LOG"
  bash experiments/run_captioner_controls.sh >> "$LOG" 2>&1
  if grep -q "^== fim" "$LOG"; then
    echo "=== concluido em $(date '+%H:%M:%S') ===" >> "$LOG"
    exit 0
  fi
  sleep 10
done
echo "=== DESISTIU apos 12 tentativas ===" >> "$LOG"
