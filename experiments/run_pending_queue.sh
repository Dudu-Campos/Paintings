#!/usr/bin/env bash
# Fila sequencial do que ficou pendente depois que a GPU voltou:
#   1. places_lt: espera a extracao WND-CHARM, roda a avaliacao  (fecha o 7o dataset)
#   2. food101:   espera o cache de RAG, roda o marginal WL       (a extensao do §7.11b)
#
# SEQUENCIAL de proposito: o marginal do WL segura ~50 GB (contagens + rotulos
# de no) e a avaliacao do places_lt mais ~18 GB. Em paralelo arriscam OOM, e o
# §1 ja proibe dois processos sobre o mesmo output.
#
# Lock de diretorio: em 30/set dois watchers dispararam o mesmo script e
# consumiram 105 de 125 GB (§8, nona vez).
set -u
cd "$(dirname "$0")/.."

LOCK=/tmp/prism_pending_queue.lock
PY=.venv-rtx3090-py312/bin/python3

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "=== lock $LOCK existe -- outra instancia no ar, saindo ==="; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

wait_for() {  # $1 = marcador no log, $2 = log, $3 = padrao do processo
  while ! grep -q "$1" "$2" 2>/dev/null; do
    if ! pgrep -f "$3" | grep -qv "^$$\$"; then
      say "FALHOU: '$3' nao esta rodando e '$1' nao apareceu em $2"; tail -5 "$2"; return 1
    fi
    sleep 30
  done
}

# ---------- 1. places_lt ----------
say "esperando a extracao do places_lt"
if wait_for "gravado" logs/wndcharm_tiled_places_lt.log "extract_wndcharm_tiled_features.py --dataset places_lt"; then
  say "extracao do places_lt pronta -- avaliando"
  "$PY" experiments/evaluate_wndcharm_tiled_vs_hand8.py \
    --dataset places_lt --tiled-dir outputs/places_lt_wndcharm_tiled \
    --output results/wndcharm_tiled_vs_hand8_places_lt.csv \
    > logs/wndcharm_tiled_eval_places_lt.log 2>&1
  say "places_lt avaliado (saida em logs/wndcharm_tiled_eval_places_lt.log)"
else
  say "pulando a avaliacao do places_lt"
fi

# ---------- 2. food101 WL ----------
say "esperando o cache de RAG do food101"
if wait_for "pronto:" logs/cache_rag_food101.log "cache_rag_compact.py --dataset food101"; then
  say "cache do food101 pronto -- marginal WL"
  "$PY" experiments/evaluate_wl_pipeline_multitask.py \
    --dataset food101 --cache-dir /tmp/prism_data/food101_slic_rag \
    --k 64 --folds 5 \
    --output results/food101_wl_pipeline.csv \
    > logs/wl_pipeline_food101.log 2>&1
  say "food101 WL pronto (saida em logs/wl_pipeline_food101.log)"
else
  say "pulando o marginal WL do food101"
fi

say "FIM DA FILA"
