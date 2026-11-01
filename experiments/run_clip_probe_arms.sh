#!/usr/bin/env bash
# Probe linear nas DUAS escalas de CLIP, sobre a fonte reconstruida.
#
# Preenche as celulas Acc/macro-F1 que faltavam na tabela do §5.18g: a execucao
# de 09/set que produziu o 60,09 e ANTERIOR ao patch que fez o script emitir as
# tres metricas, e as reexecucoes de 18/set usaram --skip-probe.
#
# Roda sobre os caches `_orig` de proposito. O bloco ‡ da tabela compara ESCALAS
# de CLIP, e misturar fontes de imagem ali confundiria modelo com recompressao
# JPEG -- que e o que o controle do B/32 mediu valendo 0,84pp. Com isto o bloco
# fica inteiro numa fonte so: dois encoders x duas leituras (zero-shot, probe).
set -u
cd /exp/duducampos/Paintings
PY=.ABSvenv/bin/python3

probe () {   # apelido  modelo
  local apelido="$1" modelo="$2"
  local saida="results/multitask_clip_probe_${apelido}_orig.json"
  if [ -f "$saida" ]; then echo "== ${apelido}: ja existe"; return 0; fi
  echo "== probe ${apelido} (${modelo})"
  $PY experiments/evaluate_clip_text_arms.py --dataset multitask \
      --clip-model "$modelo" --clip-cache "outputs/multitask_clip_${apelido}_orig.npy" \
      --output "$saida" || { echo "FALHOU_${apelido}"; return 1; }
}

probe b32 openai/clip-vit-base-patch32  || exit 1
probe l14 openai/clip-vit-large-patch14 || exit 1
echo "== fim"
