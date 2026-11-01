#!/usr/bin/env bash
# Bloco Multiscale (quadrantes) para os datasets externos.
#
# O Multiscale do WikiArt não é uma extração nova: é rodar os MESMOS
# extratores com `--region top_left|top_right|bottom_left|bottom_right` em vez
# de `global`. Quatro famílias por quadrante (descriptors, hilbert, fourier,
# slic), no layout `outputs/{ds}_quadrant_features/{region}/`, que é o que
# `final_tfidf_visual_mlp.load_dataset(quadrant_dir=...)` espera.
#
# Vale o custo: no WikiArt, Multiscale sozinho faz 37,62% na sonda LDA contra
# 30,44% de Descriptors global -- é o bloco isolado mais forte do pipeline.
# E o achado 11 mostrou que só a família Descriptors paga o próprio branch no
# MLP, então a configuração de produção (54,42%) depende destes arquivos.
#
# Retomável: pula o que já existe.
set -u
cd /exp/duducampos/Paintings
PY=.venv/bin/python
WORKERS=${WORKERS:-14}
LOG=logs/external_quadrants.log

run () {
  local name=$1 marker=$2 log=$3; shift 3
  if [ -e "$marker" ]; then
    echo "=== [$(date +%H:%M:%S)] $name já existe, pulando" | tee -a $LOG; return
  fi
  echo "=== [$(date +%H:%M:%S)] $name" | tee -a $LOG
  if "$@" >> "$log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] $name OK" | tee -a $LOG
  else
    echo "=== [$(date +%H:%M:%S)] $name FALHOU" | tee -a $LOG
  fi
}

for ds in pandora rijksmuseum; do
  if [ "$ds" = "pandora" ]; then IMG=data/pandora_flat; else IMG=rijksmuseum; fi
  META=data/${ds}_meta.csv
  for region in top_left top_right bottom_left bottom_right; do
    OUT=outputs/${ds}_quadrant_features/${region}
    mkdir -p "$OUT"

    run "$ds/$region descriptors" "$OUT/desc_full_features.npy" \
        logs/${ds}_quad_descriptors.log \
      $PY src/extractors/extract_descriptors_full.py --region "$region" \
        --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

    run "$ds/$region hilbert" "$OUT/hilbert_features.npy" \
        logs/${ds}_quad_hilbert.log \
      $PY src/extractors/extract_hilbert_features.py --region "$region" \
        --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

    run "$ds/$region fourier" "$OUT/fourier_full_features.parquet" \
        logs/${ds}_quad_fourier.log \
      $PY experiments/extract_fourier_full_features.py --region "$region" \
        --sample outputs/${ds}_slic_full/index.csv \
        --image-dir "$IMG" --meta "$META" \
        --output "$OUT/fourier_full_features.parquet" --workers "$WORKERS"

    # SLIC por quadrante sai de `slic_multigraph_features.py`, que tem
    # `--region`; o `segmentation_multigraph_features.py` (usado no bloco
    # global) não tem -- lá "region" são os superpixels, não os quadrantes.
    run "$ds/$region slic" "$OUT/slic/combined_features.parquet" \
        logs/${ds}_quad_slic.log \
      $PY src/extractors/slic_multigraph_features.py --region "$region" \
        --image-dir "$IMG" --meta "$META" --output-dir "$OUT/slic" \
        --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
        --sample-per-style 0 --workers "$WORKERS"
  done
done

echo "=== [$(date +%H:%M:%S)] quadrantes terminados" | tee -a $LOG
