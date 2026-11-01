#!/usr/bin/env bash
# Extração das modalidades para um dataset qualquer, agnóstico de máquina.
#
#   bash experiments/run_dataset_extraction.sh food101 /tmp/prism_data/food-101/images
#   SKIP_MULTISCALE=1 bash experiments/run_dataset_extraction.sh <ds> <dir> <workers>
#
# Por que existe em vez de mais um run_<dataset>_extraction.sh: os drivers
# anteriores foram derivados uns dos outros com `sed`, o que já custou um
# experimento inteiro (aspas simples dentro de f-string escapando a
# substituição, 16/ago). Um driver parametrizado evita a próxima cópia.
#
# **Imagens em disco local, saídas no NFS.** Medido em 24/ago na absinto:
# leitura de JPEG a 48.387 img/s no NVMe local contra 4.049 img/s no NFS (12x),
# e cada imagem é lida ~17 vezes ao longo do pipeline. Passe o `image_dir` já
# apontando para o disco local da máquina onde isto roda.
set -u
cd "$(dirname "$0")/.."

DATASET=${1:?uso: run_dataset_extraction.sh <dataset> <image_dir> [workers]}
IMG=${2:?falta o diretorio de imagens (use disco LOCAL)}
WORKERS=${3:-$(( $(nproc) - 4 ))}
META=data/${DATASET}_meta.csv

# SKIP_MULTISCALE=1 pula a etapa 3 (quadrantes). O achado de 24/ago e que
# Multiscale nao se paga no MLP, e ele custa ~60% do tempo de extracao.
SKIP_MULTISCALE=${SKIP_MULTISCALE:-0}

# Cap de BLAS/OMP herdado por todos os filhos: com N workers, cada um abriria
# um pool do tamanho de `nproc` e os processos brigariam por CPU. So o
# extrator de Fourier faz esse cap por conta propria.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# O venv difere por máquina: .venv na mojito (Python 3.12 do uv), .ABSvenv na
# absinto. Escolher pelo que existe evita editar o script ao trocar de máquina.
for candidate in .ABSvenv/bin/python3 .venv/bin/python3 .venv-rtx3090-py312/bin/python3; do
  [ -x "$candidate" ] || continue
  "$candidate" -c 'import numpy' >/dev/null 2>&1 || continue
  PY=$candidate && break
done
: "${PY:?nenhum venv utilizavel (testado: importa numpy?)}"

LOG=logs/${DATASET}_extraction.log
mkdir -p logs outputs
echo "=== [$(date +%H:%M:%S)] $DATASET | $PY | $WORKERS workers | $IMG" | tee -a "$LOG"
[ -f "$META" ] || { echo "metadado ausente: $META" | tee -a "$LOG"; exit 1; }

run () {                       # run <nome> <marcador> <log> <comando...>
  local name=$1 marker=$2 log=$3; shift 3
  if [ -e "$marker" ]; then
    echo "=== [$(date +%H:%M:%S)] $name já existe, pulando" | tee -a "$LOG"
    return
  fi
  echo "=== [$(date +%H:%M:%S)] $name" | tee -a "$LOG"
  local status=0
  "$@" >> "$log" 2>&1 || status=$?
  if [ "$status" = 0 ]; then
    echo "=== [$(date +%H:%M:%S)] $name OK" | tee -a "$LOG"
  else
    echo "=== [$(date +%H:%M:%S)] $name FALHOU (código $status)" | tee -a "$LOG"
  fi
}

D=$DATASET
# Etapa 1: multigrafo SLIC. Primeiro porque, além das 71 features, exporta o RAG
# e escreve o index.csv que Fourier e os quadrantes consomem como coorte.
run "$D/multigraph slic" outputs/${D}_slic_features/combined_features.parquet logs/${D}_mg_slic.log \
  $PY src/extractors/segmentation_multigraph_features.py --method slic \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/${D}_slic_features \
    --export-rag-dir outputs/${D}_slic_full/graphs \
    --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
    --sample-per-style 0 --workers "$WORKERS"

run "$D/multigraph felzenszwalb" outputs/${D}_felz_features/combined_features.parquet logs/${D}_mg_felz.log \
  $PY src/extractors/segmentation_multigraph_features.py --method felzenszwalb \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/${D}_felz_features \
    --export-rag-dir outputs/${D}_felz_full/graphs \
    --felz-scale 50.0 --felz-min-size 50 --felz-sigma 0.8 \
    --segments 2048 --max-side 1024 --k 10 --sample-per-style 0 --workers "$WORKERS"

for pair in "slic:" "felz:_felz"; do
  method=${pair%%:*}; suffix=${pair##*:}
  run "$D/line_graph($method)" results/line_graph${suffix}_${D}_features.parquet logs/${D}_lg_${method}.log \
    $PY src/extractors/extract_line_graph_features.py \
      --sample outputs/${D}_${method}_full/index.csv \
      --graph-dir outputs/${D}_${method}_full/graphs \
      --output results/line_graph${suffix}_${D}_features.parquet --workers "$WORKERS"
done

# Etapa 2: descritores globais
run "$D/descriptors" outputs/${D}_descriptors/desc_full_features.npy logs/${D}_desc.log \
  $PY src/extractors/extract_descriptors_full.py \
    --image_dir "$IMG" --meta "$META" --output outputs/${D}_descriptors --workers "$WORKERS"

run "$D/hilbert" outputs/${D}_hilbert/hilbert_features.npy logs/${D}_hilbert.log \
  $PY src/extractors/extract_hilbert_features.py \
    --image_dir "$IMG" --meta "$META" --output outputs/${D}_hilbert --workers "$WORKERS"

run "$D/gabor" outputs/${D}_gabor/gabor_features.npy logs/${D}_gabor.log \
  $PY src/extractors/extract_gabor_features.py \
    --image_dir "$IMG" --meta "$META" --output outputs/${D}_gabor --workers "$WORKERS"

run "$D/fourier" results/fourier_${D}_features.parquet logs/${D}_fourier.log \
  $PY src/extractors/extract_fourier_full_features.py \
    --sample outputs/${D}_slic_full/index.csv --image-dir "$IMG" --meta "$META" \
    --output results/fourier_${D}_features.parquet --workers "$WORKERS"

# Etapa 3: Multiscale (4 quadrantes). Fica por último por ser o mais caro e o
# menos usado -- o achado de 24/ago é que ele nao se paga no MLP.
if [ "$SKIP_MULTISCALE" = "1" ]; then
  echo "=== [$(date +%H:%M:%S)] $D: SKIP_MULTISCALE=1, pulando os quadrantes" | tee -a "$LOG"
fi
for region in top_left top_right bottom_left bottom_right; do
  [ "$SKIP_MULTISCALE" = "1" ] && break
  OUT=outputs/${D}_quadrant_features/${region}; mkdir -p "$OUT"
  run "$D/$region descriptors" "$OUT/desc_full_features.npy" logs/${D}_quad_desc.log \
    $PY src/extractors/extract_descriptors_full.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"
  run "$D/$region hilbert" "$OUT/hilbert_features.npy" logs/${D}_quad_hilbert.log \
    $PY src/extractors/extract_hilbert_features.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"
  run "$D/$region slic" "$OUT/slic/combined_features.parquet" logs/${D}_quad_slic.log \
    $PY src/extractors/slic_multigraph_features.py --region "$region" \
      --image-dir "$IMG" --meta "$META" --output-dir "$OUT/slic" \
      --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
      --sample-per-style 0 --workers "$WORKERS"
done

echo "=== [$(date +%H:%M:%S)] $DATASET: todas as modalidades terminadas" | tee -a "$LOG"
