#!/usr/bin/env bash
# Extração dos grafos e features de grafo para os dois datasets externos.
#
# Objetivo: mostrar que "features hand-designed competem com CNN" não é
# específico do WikiArt. Pandora (18k, 18 estilos) tem a mesma tarefa; o
# Rijksmuseum (4.9k pinturas) tem tarefa **diferente** -- período de
# meio-século em vez de estilo -- o que é proposital: se a complementaridade
# entre representações também aparecer numa tarefa que não é estilo, o
# argumento deixa de depender do rótulo.
#
# Sequencial de propósito: cada extrator já paraleliza em WORKERS processos,
# e rodar dois ao mesmo tempo só faria os dois brigarem por CPU. Tudo é
# retomável -- os extratores pulam os `.pt` que já existem, então basta
# reexecutar este script se algo morrer.
set -u
cd /exp/duducampos/Paintings
PY=.venv/bin/python
WORKERS=${WORKERS:-14}
mkdir -p logs outputs

run () {                       # run <nome> <log> <comando...>
  local name=$1 log=$2; shift 2
  echo "=== [$(date +%H:%M:%S)] $name" | tee -a logs/external_overnight.log
  if "$@" >> "$log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] $name OK" | tee -a logs/external_overnight.log
  else
    # Não aborta: um dataset quebrado não pode custar a noite do outro.
    echo "=== [$(date +%H:%M:%S)] $name FALHOU (código $?)" | tee -a logs/external_overnight.log
  fi
}

for ds in pandora rijksmuseum; do
  if [ "$ds" = "pandora" ]; then IMG=data/pandora_flat; else IMG=rijksmuseum; fi
  META=data/${ds}_meta.csv

  run "$ds/slic" logs/${ds}_slic.log \
    $PY src/extractors/extract_slic_rag_graphs.py \
      --image-dir "$IMG" --meta "$META" --sample-per-style 0 \
      --output-dir outputs/${ds}_slic_full/graphs --workers "$WORKERS"

  run "$ds/felzenszwalb" logs/${ds}_felz.log \
    $PY src/pipeline/extract_felzenszwalb_rag_graphs.py \
      --image-dir "$IMG" --meta "$META" --sample-per-style 0 \
      --output-dir outputs/${ds}_felz_full/graphs --workers "$WORKERS"

  # Features de grafo de linha em cima dos dois RAGs -- é o operador que mais
  # rendeu no WikiArt (achado 14: +0,79pp somando L(G_felz) a L(G_slic)).
  # `--sample` tem default do WikiArt (3k stems); aqui vai o `index.csv` que o
  # próprio exportador grava, senão ele procura stems que não existem.
  run "$ds/line_graph(slic)" logs/${ds}_linegraph.log \
    $PY experiments/extract_line_graph_features.py \
      --sample outputs/${ds}_slic_full/index.csv \
      --graph-dir outputs/${ds}_slic_full/graphs \
      --output results/line_graph_${ds}_features.parquet --workers "$WORKERS"

  run "$ds/line_graph(felz)" logs/${ds}_linegraph_felz.log \
    $PY experiments/extract_line_graph_features.py \
      --sample outputs/${ds}_felz_full/index.csv \
      --graph-dir outputs/${ds}_felz_full/graphs \
      --output results/line_graph_felz_${ds}_features.parquet --workers "$WORKERS"
done

echo "=== [$(date +%H:%M:%S)] tudo terminado" | tee -a logs/external_overnight.log
