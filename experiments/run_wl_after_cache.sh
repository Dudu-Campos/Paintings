#!/usr/bin/env bash
# Espera o cache de RAG terminar e dispara a avaliacao WL.
# Encadeamento em ARQUIVO, nao em `bash -c` inline -- §8: encadear inline ja
# custou tempo oito vezes neste projeto.
set -u
cd "$(dirname "$0")/.."

CACHE_LOG=logs/cache_rag_slic.log
CACHE_DIR=/tmp/prism_data/multitask_slic_rag

echo "=== [$(date '+%d/%b %H:%M:%S')] esperando o cache terminar ==="
# Espera pelo MARCADOR no log, nao por PID: o PID some se o processo for
# reiniciado e o marcador e o unico sinal de que o cache completou de verdade.
while ! grep -q "pronto:" "$CACHE_LOG" 2>/dev/null; do
  if ! pgrep -af cache_rag_compact | grep -qv "pgrep\|grep"; then
    echo "=== cache nao esta rodando e nao completou -- abortando ==="
    tail -5 "$CACHE_LOG"
    exit 1
  fi
  sleep 20
done

N_SHARDS=$(ls "$CACHE_DIR"/shard_*.npz 2>/dev/null | wc -l)
echo "=== [$(date '+%d/%b %H:%M:%S')] cache completo, $N_SHARDS shards ==="

exec .venv-rtx3090-py312/bin/python3 experiments/evaluate_wl_subtree_multitask.py \
  --k 64 --folds 5
