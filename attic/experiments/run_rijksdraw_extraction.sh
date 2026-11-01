#!/usr/bin/env bash
# Extração completa das 8 modalidades visuais + ResNet50 congelada nos
# desenhos do Rijksmuseum (~43,5 mil imagens, 11 classes de meio-século).
#
# Este é o teste da **previsão registrada em 15/ago**: o padrão observado nos
# três datasets não é distância da tarefa ao ImageNet, é ter dado suficiente
# para a fusão de branches funcionar (3,9k -> perdemos; 14k -> ganhamos; 63k ->
# perdemos para transferência). Os desenhos põem o MESMO museu e a MESMA tarefa
# de período em ~35k de treino. Se a previsão vale, o sinal vira a nosso favor
# aqui sem que nada além do `n` tenha mudado.
#
# Ordem por dependência, não por custo: os exportadores de RAG gravam o
# `index.csv` que o line graph e o Fourier consomem como `--sample`.
#
# Parâmetros idênticos aos das pinturas (felz_scale=50, min_size=50,
# sigma=0.8 -- que **não** são os defaults do script). Trocar qualquer um
# quebraria a comparação entre os dois braços, que é o ponto do experimento.
#
# Retomável: cada etapa pula se a saída já existe; os exportadores pulam `.pt`
# já gravados. Pode matar e reexecutar à vontade.
set -u
cd /exp/duducampos/Paintings
PY=.venv/bin/python
WORKERS=${WORKERS:-20}
DS=rijksdraw
IMG=rijksmuseum
META=data/${DS}_meta.csv
LOG=logs/${DS}_extraction.log
mkdir -p logs outputs results

run () {                       # run <nome> <marcador> <log> <comando...>
  local name=$1 marker=$2 log=$3; shift 3
  if [ -e "$marker" ]; then
    echo "=== [$(date +%H:%M:%S)] $name já existe, pulando" | tee -a $LOG; return
  fi
  echo "=== [$(date +%H:%M:%S)] $name" | tee -a $LOG
  if "$@" >> "$log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] $name OK" | tee -a $LOG
  else
    # Não aborta: uma modalidade quebrada não pode custar as outras sete.
    echo "=== [$(date +%H:%M:%S)] $name FALHOU (código $?)" | tee -a $LOG
  fi
}

echo "=== [$(date +%H:%M:%S)] início, $(wc -l < $META) linhas em $META" | tee -a $LOG

# --- grafos: gatilho de tudo que vem depois -------------------------------
run "$DS/slic graphs" outputs/${DS}_slic_full/index.csv logs/${DS}_slic.log \
  $PY src/extractors/extract_slic_rag_graphs.py \
    --image-dir "$IMG" --meta "$META" --sample-per-style 0 \
    --output-dir outputs/${DS}_slic_full/graphs --workers "$WORKERS"

run "$DS/felz graphs" outputs/${DS}_felz_full/index.csv logs/${DS}_felz.log \
  $PY src/pipeline/extract_felzenszwalb_rag_graphs.py \
    --image-dir "$IMG" --meta "$META" --sample-per-style 0 \
    --output-dir outputs/${DS}_felz_full/graphs --workers "$WORKERS"

# --- baratas: minutos cada -------------------------------------------------
run "$DS/descriptors" outputs/${DS}_descriptors/desc_full_features.npy \
    logs/${DS}_descriptors.log \
  $PY src/extractors/extract_descriptors_full.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/${DS}_descriptors --workers "$WORKERS"

run "$DS/hilbert" outputs/${DS}_hilbert/hilbert_features.npy \
    logs/${DS}_hilbert.log \
  $PY src/extractors/extract_hilbert_features.py \
    --image_dir "$IMG" --meta "$META" \
    --output outputs/${DS}_hilbert --workers "$WORKERS"

run "$DS/fourier" results/fourier_${DS}_features.parquet \
    logs/${DS}_fourier.log \
  $PY experiments/extract_fourier_full_features.py \
    --sample outputs/${DS}_slic_full/index.csv \
    --image-dir "$IMG" --meta "$META" \
    --output results/fourier_${DS}_features.parquet --workers "$WORKERS"

run "$DS/line_graph(slic)" results/line_graph_${DS}_features.parquet \
    logs/${DS}_linegraph.log \
  $PY experiments/extract_line_graph_features.py \
    --sample outputs/${DS}_slic_full/index.csv \
    --graph-dir outputs/${DS}_slic_full/graphs \
    --output results/line_graph_${DS}_features.parquet --workers "$WORKERS"

run "$DS/line_graph(felz)" results/line_graph_felz_${DS}_features.parquet \
    logs/${DS}_linegraph_felz.log \
  $PY experiments/extract_line_graph_features.py \
    --sample outputs/${DS}_felz_full/index.csv \
    --graph-dir outputs/${DS}_felz_full/graphs \
    --output results/line_graph_felz_${DS}_features.parquet --workers "$WORKERS"

# --- multigrafo: refaz a segmentação e monta RAG+kNN+MST+Delaunay ----------
run "$DS/multigraph slic" outputs/${DS}_slic_features/combined_features.parquet \
    logs/${DS}_multigraph_slic.log \
  $PY src/extractors/segmentation_multigraph_features.py --method slic \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/${DS}_slic_features \
    --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
    --sample-per-style 0 --workers "$WORKERS"

run "$DS/multigraph felz" outputs/${DS}_felz_features/combined_features.parquet \
    logs/${DS}_multigraph_felz.log \
  $PY src/extractors/segmentation_multigraph_features.py --method felzenszwalb \
    --image-dir "$IMG" --meta "$META" --output-dir outputs/${DS}_felz_features \
    --felz-scale 50.0 --felz-min-size 50 --felz-sigma 0.8 \
    --segments 2048 --max-side 1024 --k 10 \
    --sample-per-style 0 --workers "$WORKERS"

# --- Multiscale: os mesmos extratores por quadrante ------------------------
# O SLIC por quadrante é a etapa mais cara do script inteiro (4x o multigrafo).
# Fica por último de propósito: se a noite acabar antes, as sete modalidades
# globais já dão o resultado principal, e o Multiscale entra depois.
for region in top_left top_right bottom_left bottom_right; do
  OUT=outputs/${DS}_quadrant_features/${region}
  mkdir -p "$OUT"

  run "$DS/$region descriptors" "$OUT/desc_full_features.npy" \
      logs/${DS}_quad_descriptors.log \
    $PY src/extractors/extract_descriptors_full.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

  run "$DS/$region hilbert" "$OUT/hilbert_features.npy" \
      logs/${DS}_quad_hilbert.log \
    $PY src/extractors/extract_hilbert_features.py --region "$region" \
      --image_dir "$IMG" --meta "$META" --output "$OUT" --workers "$WORKERS"

  run "$DS/$region fourier" "$OUT/fourier_full_features.parquet" \
      logs/${DS}_quad_fourier.log \
    $PY experiments/extract_fourier_full_features.py --region "$region" \
      --sample outputs/${DS}_slic_full/index.csv \
      --image-dir "$IMG" --meta "$META" \
      --output "$OUT/fourier_full_features.parquet" --workers "$WORKERS"

  run "$DS/$region slic" "$OUT/slic/combined_features.parquet" \
      logs/${DS}_quad_slic.log \
    $PY src/extractors/slic_multigraph_features.py --region "$region" \
      --image-dir "$IMG" --meta "$META" --output-dir "$OUT/slic" \
      --segments 2048 --compactness 10.0 --sigma 0.0 --max-side 1024 --k 10 \
      --sample-per-style 0 --workers "$WORKERS"
done

echo "=== [$(date +%H:%M:%S)] extração $DS terminada" | tee -a $LOG
