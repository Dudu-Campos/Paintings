#!/usr/bin/env bash
# Fusao PRISM + CLIP ViT-B/32 congelado nos sete corpora, pelo produtor da
# `*_final_table.csv` (BranchMLP estreito, 5 sementes). Saida SEPARADA em
# results/<ds>_clip_fusion.csv: as linhas `hand-8 + grafo` de referencia ja
# estao no final_table, no mesmo protocolo -- exceto places_lt, onde a coorte
# restrita (sem os 528 stems do §5.18d) exige rodar a base tambem.
set -u
cd /exp/duducampos/Paintings
PY=${PY:-.ABSvenv/bin/python3}
C="CLIP ViT-B/32"
COND=("$C" "hand-8 + $C" "hand-8 + grafo + $C")
for ds in oilpainting minc2500 rijksdraw food101 nct_crc multitask; do
  echo "=== $ds $(date)"
  $PY experiments/evaluate_preprocess_effect_oilpainting.py --dataset $ds --clip \
    --seeds 42 43 44 45 46 --only "${COND[@]}" --output results/${ds}_clip_fusion.csv
done
echo "=== places_lt $(date)"
$PY experiments/evaluate_preprocess_effect_oilpainting.py --dataset places_lt --clip \
  --restrict-to-clip --seeds 42 43 44 45 46 --only hand-8 "hand-8 + grafo" "${COND[@]}" \
  --output results/places_lt_clip_fusion.csv
echo "=== FIM $(date)"
