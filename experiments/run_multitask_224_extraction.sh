#!/usr/bin/env bash
# Extrai os dois segmentadores do Multitask em --max-side 224, pareando a
# resolucao de entrada com a da ResNet/ViT. Tudo o mais identico a producao
# (felz_scale 50, min_size 50, segments 2048, k 10), para que a UNICA variavel
# seja a resolucao. Saidas em diretorios proprios: nada de producao e tocado.
#
# Por que: o §7.1 defende o paper contra "voces veem mais pixels que a CNN" com
# uma mediana de 343 px que foi medida na copia local reduzida. A mediana real
# da coorte e 781 px e 30% passam de 1024, entao a defesa precisa de medicao.
set -u
PY=.ABSvenv/bin/python3
IMG=data/multitask_painting100k/original/images
META=data/multitask_meta.csv
W=${1:-8}
D=multitask224
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

  run "$D/multigraph slic" outputs/${D}_slic_features/combined_features.parquet logs/${D}_mg_slic.log \
    $PY src/extractors/segmentation_multigraph_features.py --method slic \
      --image-dir "$IMG" --meta "$META" --output-dir outputs/${D}_slic_features \
      --export-rag-dir outputs/${D}_slic_full/graphs \
      --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 224 --k 10 \
      --sample-per-style 0 --workers "$W"

  run "$D/multigraph felzenszwalb" outputs/${D}_felz_features/combined_features.parquet logs/${D}_mg_felz.log \
    $PY src/extractors/segmentation_multigraph_features.py --method felzenszwalb \
      --image-dir "$IMG" --meta "$META" --output-dir outputs/${D}_felz_features \
      --export-rag-dir outputs/${D}_felz_full/graphs \
      --felz-scale 50.0 --felz-min-size 50 --felz-sigma 0.8 \
      --segments 2048 --max-side 224 --k 10 --sample-per-style 0 --workers "$W"

  for pair in "slic:" "felz:_felz"; do
    method=${pair%%:*}; suffix=${pair##*:}
    run "$D/line_graph($method)" results/line_graph${suffix}_${D}_features.parquet logs/${D}_lg_${method}.log \
      $PY src/extractors/extract_line_graph_features.py \
        --sample outputs/${D}_${method}_full/index.csv \
        --graph-dir outputs/${D}_${method}_full/graphs \
        --output results/line_graph${suffix}_${D}_features.parquet --workers "$W"
  done

  if [ -f outputs/${D}_felz_features/combined_features.parquet ] \
     && [ -f results/line_graph_felz_${D}_features.parquet ]; then
    echo "=== [$(date +%H:%M:%S)] COMPLETO" | tee -a "$LOG"; break
  fi
  echo "=== [$(date +%H:%M:%S)] incompleto, retomando" | tee -a "$LOG"
done
