#!/usr/bin/env bash
# Espera o WL standalone terminar e dispara o teste de pipeline (marginal sobre
# o hand-8).
#
# GUARDA CONTRA DUPLO-LANCAMENTO. Em 30/set dois watchers dispararam o mesmo
# script ao mesmo tempo: 105 de 125 GB de RAM, os dois escrevendo no mesmo CSV
# (o hazard do §1). A causa foi eu concluir que um watcher tinha morrido a
# partir de um `grep` que nao casou -- §8, nona vez. Aqui o lockfile resolve
# independente de eu saber quem esta vivo.
set -u
cd "$(dirname "$0")/.."

LOCK=/tmp/prism_wl_pipeline.lock
STANDALONE_LOG=logs/wl_subtree_multitask.log
STANDALONE_CSV=results/multitask_wl_subtree.csv

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "=== lock $LOCK ja existe -- outra instancia esta no ar, saindo ==="
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

echo "=== [$(date '+%d/%b %H:%M:%S')] esperando o WL standalone terminar ==="
# Espera pelo MARCADOR, nao por PID: PID some num restart e so o marcador diz
# que terminou de verdade.
while ! grep -q "delta contra d0" "$STANDALONE_LOG" 2>/dev/null; do
  if ! pgrep -af evaluate_wl_subtree_multitask | grep -qv "pgrep\|grep"; then
    echo "=== standalone nao esta rodando e nao completou -- abortando ==="
    tail -5 "$STANDALONE_LOG"
    exit 1
  fi
  sleep 30
done

echo "=== [$(date '+%d/%b %H:%M:%S')] standalone completo ==="
tail -14 "$STANDALONE_LOG"

if pgrep -af evaluate_wl_pipeline_multitask | grep -qv "pgrep\|grep"; then
  echo "=== o pipeline JA esta rodando -- nao vou lancar outro ==="
  exit 0
fi

echo "=== [$(date '+%d/%b %H:%M:%S')] lancando o teste de pipeline ==="
.venv-rtx3090-py312/bin/python3 experiments/evaluate_wl_pipeline_multitask.py \
  --k 64 --folds 5
