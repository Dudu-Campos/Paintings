#!/usr/bin/env bash
# Extração das 8 modalidades nativas (sem Text) pro MultitaskPainting100k
# (~99.816 imagens, 125 estilos), adaptado dos três run_external_*.sh
# (pandora/rijksmuseum) pra esta máquina (absinto): .venv está quebrado
# aqui, usar .ABSvenv. Retomável -- cada etapa pula se a saída já existe.
set -u
cd /exp/duducampos/Paintings
PY=.ABSvenv/bin/python3
WORKERS=${WORKERS:-14}
IMG=data/oilpainting/image
META=data/oilpainting_meta.csv
mkdir -p logs outputs

run () {                       # run <nome> <marcador> <log> <comando...>
  local name=$1 marker=$2 log=$3; shift 3
  if [ -e "$marker" ]; then
    echo "=== [$(date +%H:%M:%S)] $name já existe, pulando" | tee -a logs/oilpainting_extraction.log
    return
  fi
  echo "=== [$(date +%H:%M:%S)] $name" | tee -a logs/oilpainting_extraction.log
  if "$@" >> "$log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] $name OK" | tee -a logs/oilpainting_extraction.log
  else
    echo "=== [$(date +%H:%M:%S)] $name FALHOU (código $?)" | tee -a logs/oilpainting_extraction.log
  fi
}

# --- etapa 1: multigrafo SLIC ---
# Primeiro porque, além das 71 features, ele exporta o RAG e escreve o
# index.csv que o Fourier e os quadrantes consomem como coorte.
run "oilpainting/multigraph slic" outputs/oilpainting_slic_features/combined_features.parquet logs/oilpainting_multigraph_slic.log \
  $PY src/extractors/segmentation_multigraph_features.py --method slic \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/oilpainting_slic_features \
    --export-rag-dir outputs/oilpainting_slic_full/graphs \
    --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
    --sample-per-style 0 --workers "$WORKERS"

# --- etapa 2: Descriptors, Hilbert, Fourier, Gabor ---
run "oilpainting/descriptors" outputs/oilpainting_descriptors/desc_full_features.npy logs/oilpainting_descriptors.log \
  $PY src/extractors/extract_descriptors_full.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/oilpainting_descriptors --workers "$WORKERS"

run "oilpainting/hilbert" outputs/oilpainting_hilbert/hilbert_features.npy logs/oilpainting_hilbert.log \
  $PY src/extractors/extract_hilbert_features.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/oilpainting_hilbert --workers "$WORKERS"

run "oilpainting/fourier" results/fourier_oilpainting_features.parquet logs/oilpainting_fourier.log \
  $PY src/extractors/extract_fourier_full_features.py \
    --sample outputs/oilpainting_slic_full/index.csv \
    --image-dir "$IMG" --meta "$META" \
    --output results/fourier_oilpainting_features.parquet --workers "$WORKERS"

# Gabor: validado no Pandora hoje (LDA +0,25pp p=0,015, MLP +0,75pp p=0,020,
# 5/5 folds nos dois) -- achado 27 do CLAUDE.md apontava como pendente.
run "oilpainting/gabor" outputs/oilpainting_gabor/gabor_features.npy logs/oilpainting_gabor.log \
  $PY src/extractors/extract_gabor_features.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/oilpainting_gabor --workers "$WORKERS"

run "oilpainting/multigraph felzenszwalb" outputs/oilpainting_felz_features/combined_features.parquet logs/oilpainting_multigraph_felz.log \
  $PY src/extractors/segmentation_multigraph_features.py --method felzenszwalb \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/oilpainting_felz_features \
    --export-rag-dir outputs/oilpainting_felz_full/graphs \
    --felz-scale 50.0 --felz-min-size 50 --felz-sigma 0.8 \
    --segments 2048 --max-side 1024 --k 10 \
    --sample-per-style 0 --workers "$WORKERS"

run "oilpainting/line_graph(slic)" results/line_graph_oilpainting_features.parquet logs/oilpainting_linegraph.log \
  $PY src/extractors/extract_line_graph_features.py \
    --sample outputs/oilpainting_slic_full/index.csv \
    --graph-dir outputs/oilpainting_slic_full/graphs \
    --output results/line_graph_oilpainting_features.parquet --workers "$WORKERS"

run "oilpainting/line_graph(felz)" results/line_graph_felz_oilpainting_features.parquet logs/oilpainting_linegraph_felz.log \
  $PY src/extractors/extract_line_graph_features.py \
    --sample outputs/oilpainting_felz_full/index.csv \
    --graph-dir outputs/oilpainting_felz_full/graphs \
    --output results/line_graph_felz_oilpainting_features.parquet --workers "$WORKERS"

# --- etapa 3: Multiscale (4 quadrantes x descriptors/hilbert/fourier/slic) ---
for region in top_left top_right bottom_left bottom_right; do
  OUT=outputs/oilpainting_quadrant_features/${region}
  mkdir -p "$OUT"

  run "oilpainting/$region descriptors" "$OUT/desc_full_features.npy" logs/oilpainting_quad_descriptors.log \
    $PY src/extractors/extract_descriptors_full.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

  run "oilpainting/$region hilbert" "$OUT/hilbert_features.npy" logs/oilpainting_quad_hilbert.log \
    $PY src/extractors/extract_hilbert_features.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

  run "oilpainting/$region fourier" "$OUT/fourier_full_features.parquet" logs/oilpainting_quad_fourier.log \
    $PY src/extractors/extract_fourier_full_features.py --region "$region" \
      --sample outputs/oilpainting_slic_full/index.csv \
      --image-dir "$IMG" --meta "$META" \
      --output "$OUT/fourier_full_features.parquet" --workers "$WORKERS"

  run "oilpainting/$region slic" "$OUT/slic/combined_features.parquet" logs/oilpainting_quad_slic.log \
    $PY src/extractors/slic_multigraph_features.py --region "$region" \
      --image-dir "$IMG" --meta "$META" --output-dir "$OUT/slic" \
      --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
      --sample-per-style 0 --workers "$WORKERS"
done

echo "=== [$(date +%H:%M:%S)] oilpainting: todas as 8 modalidades terminadas" | tee -a logs/oilpainting_extraction.log
