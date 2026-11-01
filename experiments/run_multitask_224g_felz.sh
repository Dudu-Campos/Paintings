#!/usr/bin/env bash
# Reextrai SO o Felzenszwalb do Multitask a 224 px com granularidade PRESERVADA.
#
# O §9.5 mediu que trocar os quatro blocos de segmentacao pela versao de 224 px
# custa -3,53pp, mas que a causa nao e pixel: o SLIC a 224 px EMPATA (+0,05,
# n.s.) porque `--segments 2048` mantem a contagem, e o Felz perde porque
# `felz_scale`/`min_size` estao em unidades de pixel e os segmentos colapsam de
# 1.398 para 134.
#
# Parametros calibrados em 120 imagens a 224 px contra o alvo de 1.398 segmentos
# (mediana da producao a 1024): `scale=4.0, min_size=8` -> mediana 1.405.
# A escala ingenua por area daria 4/4, que erra (mediana 2.982).
#
# Previsao registrada ANTES de rodar: isso deve recuperar a maior parte dos
# 3,53pp. Se recuperar, a comparacao com as congeladas passa a ser pareada em
# resolucao E o preditor de grafo do §9.4 pode operar em 224 px.
#
# SLIC nao precisa: ele ja empata a 224 px com os parametros de producao.
set -u
PY=.ABSvenv/bin/python3
IMG=data/multitask_painting100k/original/images
META=data/multitask_meta.csv
W=${1:-8}
D=multitask224g
LOG=logs/${D}_extraction.log
mkdir -p logs

run() {
  local name=$1 sentinel=$2 log=$3; shift 3
  if [ -f "$sentinel" ]; then
    echo "=== [$(date +%H:%M:%S)] $name JA FEITO" | tee -a "$LOG"; return 0
  fi
  echo "=== [$(date +%H:%M:%S)] $name" | tee -a "$LOG"
  local status=0
  "$@" >> "$log" 2>&1 || status=$?
  echo "=== [$(date +%H:%M:%S)] $name saiu com $status" | tee -a "$LOG"
}

for attempt in 1 2 3; do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt (workers=$W)" | tee -a "$LOG"

  run "$D/multigraph felzenszwalb" outputs/${D}_felz_features/combined_features.parquet logs/${D}_mg_felz.log \
    $PY src/extractors/segmentation_multigraph_features.py --method felzenszwalb \
      --image-dir "$IMG" --meta "$META" --output-dir outputs/${D}_felz_features \
      --export-rag-dir outputs/${D}_felz_full/graphs \
      --felz-scale 4.0 --felz-min-size 8 --felz-sigma 0.8 \
      --segments 2048 --max-side 224 --k 10 --sample-per-style 0 --workers "$W"

  run "$D/line_graph(felz)" results/line_graph_felz_${D}_features.parquet logs/${D}_lg_felz.log \
    $PY src/extractors/extract_line_graph_features.py \
      --sample outputs/${D}_felz_full/index.csv \
      --graph-dir outputs/${D}_felz_full/graphs \
      --output results/line_graph_felz_${D}_features.parquet --workers "$W"

  if [ -f results/line_graph_felz_${D}_features.parquet ]; then
    echo "=== [$(date +%H:%M:%S)] COMPLETO" | tee -a "$LOG"; break
  fi
  echo "=== [$(date +%H:%M:%S)] incompleto, retomando" | tee -a "$LOG"
done
