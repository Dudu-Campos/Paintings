#!/usr/bin/env bash
# Extração das 8 modalidades nativas (sem Text) pro MultitaskPainting100k
# (~99.816 imagens, 125 estilos), adaptado dos três run_external_*.sh
# (pandora/rijksmuseum) pra esta máquina (absinto): .venv está quebrado
# aqui, usar .ABSvenv. Retomável -- cada etapa pula se a saída já existe.
set -u
cd /exp/duducampos/Paintings
PY=.ABSvenv/bin/python3
WORKERS=${WORKERS:-14}
IMG=data/multitask_painting100k/original/images
META=data/multitask_meta.csv
mkdir -p logs outputs

run () {                       # run <nome> <marcador> <log> <comando...>
  local name=$1 marker=$2 log=$3; shift 3
  if [ -e "$marker" ]; then
    echo "=== [$(date +%H:%M:%S)] $name já existe, pulando" | tee -a logs/multitask_extraction.log
    return
  fi
  echo "=== [$(date +%H:%M:%S)] $name" | tee -a logs/multitask_extraction.log
  if "$@" >> "$log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] $name OK" | tee -a logs/multitask_extraction.log
  else
    echo "=== [$(date +%H:%M:%S)] $name FALHOU (código $?)" | tee -a logs/multitask_extraction.log
  fi
}

# --- etapa 1: RAGs (SLIC/Felzenszwalb) + line graph ---
run "multitask/slic" outputs/multitask_slic_full/index.csv logs/multitask_slic.log \
  $PY src/extractors/extract_slic_rag_graphs.py \
    --image-dir "$IMG" --meta "$META" --sample-per-style 0 \
    --output-dir outputs/multitask_slic_full/graphs --workers "$WORKERS"

run "multitask/felzenszwalb" outputs/multitask_felz_full/index.csv logs/multitask_felz.log \
  $PY src/pipeline/extract_felzenszwalb_rag_graphs.py \
    --image-dir "$IMG" --meta "$META" --sample-per-style 0 \
    --output-dir outputs/multitask_felz_full/graphs --workers "$WORKERS"

run "multitask/line_graph(slic)" results/line_graph_multitask_features.parquet logs/multitask_linegraph.log \
  $PY src/extractors/extract_line_graph_features.py \
    --sample outputs/multitask_slic_full/index.csv \
    --graph-dir outputs/multitask_slic_full/graphs \
    --output results/line_graph_multitask_features.parquet --workers "$WORKERS"

run "multitask/line_graph(felz)" results/line_graph_felz_multitask_features.parquet logs/multitask_linegraph_felz.log \
  $PY src/extractors/extract_line_graph_features.py \
    --sample outputs/multitask_felz_full/index.csv \
    --graph-dir outputs/multitask_felz_full/graphs \
    --output results/line_graph_felz_multitask_features.parquet --workers "$WORKERS"

# --- etapa 2: Descriptors, Hilbert, Fourier, multigrafo global ---
run "multitask/descriptors" outputs/multitask_descriptors/desc_full_features.npy logs/multitask_descriptors.log \
  $PY src/extractors/extract_descriptors_full.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/multitask_descriptors --workers "$WORKERS"

run "multitask/hilbert" outputs/multitask_hilbert/hilbert_features.npy logs/multitask_hilbert.log \
  $PY src/extractors/extract_hilbert_features.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/multitask_hilbert --workers "$WORKERS"

run "multitask/fourier" results/fourier_multitask_features.parquet logs/multitask_fourier.log \
  $PY src/extractors/extract_fourier_full_features.py \
    --sample outputs/multitask_slic_full/index.csv \
    --image-dir "$IMG" --meta "$META" \
    --output results/fourier_multitask_features.parquet --workers "$WORKERS"

# Gabor: validado no Pandora hoje (LDA +0,25pp p=0,015, MLP +0,75pp p=0,020,
# 5/5 folds nos dois) -- achado 27 do CLAUDE.md apontava como pendente.
run "multitask/gabor" outputs/multitask_gabor/gabor_features.npy logs/multitask_gabor.log \
  $PY src/extractors/extract_gabor_features.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/multitask_gabor --workers "$WORKERS"

run "multitask/multigraph slic" outputs/multitask_slic_features/combined_features.parquet logs/multitask_multigraph_slic.log \
  $PY src/extractors/segmentation_multigraph_features.py --method slic \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/multitask_slic_features \
    --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
    --sample-per-style 0 --workers "$WORKERS"

run "multitask/multigraph felzenszwalb" outputs/multitask_felz_features/combined_features.parquet logs/multitask_multigraph_felz.log \
  $PY src/extractors/segmentation_multigraph_features.py --method felzenszwalb \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/multitask_felz_features \
    --felz-scale 50.0 --felz-min-size 50 --felz-sigma 0.8 \
    --segments 2048 --max-side 1024 --k 10 \
    --sample-per-style 0 --workers "$WORKERS"

# --- etapa 3: Multiscale (4 quadrantes x descriptors/hilbert/fourier/slic) ---
for region in top_left top_right bottom_left bottom_right; do
  OUT=outputs/multitask_quadrant_features/${region}
  mkdir -p "$OUT"

  run "multitask/$region descriptors" "$OUT/desc_full_features.npy" logs/multitask_quad_descriptors.log \
    $PY src/extractors/extract_descriptors_full.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

  run "multitask/$region hilbert" "$OUT/hilbert_features.npy" logs/multitask_quad_hilbert.log \
    $PY src/extractors/extract_hilbert_features.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

  run "multitask/$region fourier" "$OUT/fourier_full_features.parquet" logs/multitask_quad_fourier.log \
    $PY src/extractors/extract_fourier_full_features.py --region "$region" \
      --sample outputs/multitask_slic_full/index.csv \
      --image-dir "$IMG" --meta "$META" \
      --output "$OUT/fourier_full_features.parquet" --workers "$WORKERS"

  run "multitask/$region slic" "$OUT/slic/combined_features.parquet" logs/multitask_quad_slic.log \
    $PY src/extractors/slic_multigraph_features.py --region "$region" \
      --image-dir "$IMG" --meta "$META" --output-dir "$OUT/slic" \
      --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
      --sample-per-style 0 --workers "$WORKERS"
done

echo "=== [$(date +%H:%M:%S)] multitask: todas as 8 modalidades terminadas" | tee -a logs/multitask_extraction.log
