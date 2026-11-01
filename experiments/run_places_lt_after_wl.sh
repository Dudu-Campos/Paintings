#!/usr/bin/env bash
# Espera o marginal WL do food101 terminar e roda a avaliacao WND-CHARM do
# places_lt -- o 7o e ultimo dataset da tabela de comparacao.
#
# SEQUENCIAL de proposito: o WL segura dezenas de GB (contagens de subarvore) e
# a maquina ja esta em 65 de 125 GB. O conserto do memmap no avaliador reduziu
# o pico dele de 19,6 GB para ~1,2 GB, mas o §1 manda nao empilhar de qualquer
# forma numa maquina compartilhada.
#
# Lock de diretorio: §8, nona vez -- em 30/set dois watchers dispararam o mesmo
# script e consumiram 105 de 125 GB.
set -u
cd "$(dirname "$0")/.."

LOCK=/tmp/prism_places_lt_eval.lock
PY=.venv-rtx3090-py312/bin/python3
WL_LOG=logs/wl_pipeline_food101.log

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "=== lock $LOCK existe -- outra instancia no ar, saindo ==="; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

say() { echo "=== [$(date '+%d/%b %H:%M:%S')] $*"; }

say "esperando o marginal WL do food101"
# Espera pelo MARCADOR, nao por PID (§8): PID some num restart e so o marcador
# diz que terminou de verdade.
while ! grep -q "WL real contra o PLACEBO" "$WL_LOG" 2>/dev/null; do
  if ! pgrep -f "evaluate_wl_pipeline_multitask.py --dataset food101" \
       | grep -qv "^$$\$"; then
    say "o WL do food101 saiu sem completar -- seguindo para o places_lt assim mesmo"
    tail -5 "$WL_LOG"; break
  fi
  sleep 60
done

say "rodando a avaliacao WND-CHARM do places_lt"
"$PY" experiments/evaluate_wndcharm_tiled_vs_hand8.py \
  --dataset places_lt --tiled-dir outputs/places_lt_wndcharm_tiled \
  --output results/wndcharm_tiled_vs_hand8_places_lt.csv \
  > logs/wndcharm_tiled_eval_places_lt.log 2>&1
say "places_lt avaliado (saida em logs/wndcharm_tiled_eval_places_lt.log)"
