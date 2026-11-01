#!/usr/bin/env bash
# Modalidades restantes dos datasets externos, em ordem de custo crescente.
#
# Descriptors, Hilbert e Fourier já tinham CLI (`--image_dir/--meta/--output`);
# só faltava chamá-los. SLIC e Felzenszwalb custam mais porque
# `segmentation_multigraph_features.py` **refaz a segmentação** e monta quatro
# grafos por imagem (RAG, kNN, MST, Delaunay) -- os RAGs já exportados servem
# ao line graph, que é operador diferente.
#
# Parâmetros do Felzenszwalb copiados do run.log do WikiArt
# (felz_scale=50, felz_min_size=50, felz_sigma=0.8) -- **não** são os defaults
# do script (100 e 20), e sem isso os dois datasets não seriam comparáveis.
#
# Retomável: cada etapa pula se a saída já existe.
set -u
cd /exp/duducampos/Paintings
PY=.venv/bin/python
WORKERS=${WORKERS:-14}
LOG=logs/external_features.log

run () {                       # run <nome> <marcador> <log> <comando...>
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

  run "$ds/descriptors" outputs/${ds}_descriptors/desc_full_features.npy \
      logs/${ds}_descriptors.log \
    $PY src/extractors/extract_descriptors_full.py \
      --image_dir "$IMG" --meta "$META" \
      --output outputs/${ds}_descriptors --workers "$WORKERS"

  run "$ds/hilbert" outputs/${ds}_hilbert/hilbert_features.npy \
      logs/${ds}_hilbert.log \
    $PY src/extractors/extract_hilbert_features.py \
      --image_dir "$IMG" --meta "$META" \
      --output outputs/${ds}_hilbert --workers "$WORKERS"

  run "$ds/fourier" results/fourier_${ds}_features.parquet \
      logs/${ds}_fourier.log \
    $PY experiments/extract_fourier_full_features.py \
      --sample outputs/${ds}_slic_full/index.csv \
      --image-dir "$IMG" --meta "$META" \
      --output results/fourier_${ds}_features.parquet --workers "$WORKERS"

  run "$ds/multigraph slic" outputs/${ds}_slic_features/combined_features.parquet \
      logs/${ds}_multigraph_slic.log \
    $PY src/extractors/segmentation_multigraph_features.py --method slic \
      --image-dir "$IMG" --meta "$META" --output-dir outputs/${ds}_slic_features \
      --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
      --sample-per-style 0 --workers "$WORKERS"

  run "$ds/multigraph felzenszwalb" outputs/${ds}_felz_features/combined_features.parquet \
      logs/${ds}_multigraph_felz.log \
    $PY src/extractors/segmentation_multigraph_features.py --method felzenszwalb \
      --image-dir "$IMG" --meta "$META" --output-dir outputs/${ds}_felz_features \
      --felz-scale 50.0 --felz-min-size 50 --felz-sigma 0.8 \
      --segments 2048 --max-side 1024 --k 10 \
      --sample-per-style 0 --workers "$WORKERS"
done

echo "=== [$(date +%H:%M:%S)] todas as modalidades externas terminadas" | tee -a $LOG
