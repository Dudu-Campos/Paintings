#!/usr/bin/env bash
# A ablacao de captioner do §5.18, pontuada por UM scorer so.
#
# O §5.18 registra CATR 8,98 e estudante 16,08 medidos por `build_text_scores.py`;
# o `evaluate_student_captions_multitask.py` mede o MESMO estudante em 14,99 com
# o `tfidf_classify` do 05_evaluate.py (hiperparametros diferentes: L1 C=0,1,
# min_df=10, max_df=0,90). Os dois sao validos e NAO sao comparaveis entre si --
# misturar as duas fontes numa tabela e o erro que este script existe para evitar.
#
# Aqui todo braco de LEGENDA passa por `build_text_scores.py`, que e o scorer das
# duas linhas ja publicadas. O braco do CLIP nao e de legenda (e o encoder cru) e
# sai do `evaluate_clip_text_arms.py`, declarado a parte.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.ABSvenv/bin/python3}
DS=${DS:-multitask}
# JSONL, nao CSV: o `build_text_scores.py` faz `json.loads(line)` por linha
# (linha 84). Os dois formatos existem lado a lado para todo corpus, e passar o
# CSV da JSONDecodeError no primeiro `stem,style,caption`. Sem cabecalho, entao
# a contagem de linhas e a contagem de legendas.
CAPS=results/${DS}_qwen_captions.jsonl

[ -f "$CAPS" ] || { echo "FALTA $CAPS -- o Qwen ainda nao terminou"; exit 1; }
# Modo de falha do cache PARCIAL: em 08/set um smoke test de 200 linhas foi
# reaproveitado em silencio e a rodada "terminou" com uma fracao da coorte.
N=$(wc -l < "$CAPS")
ESPERADO=$(wc -l < results/${DS}_student_captions.jsonl)
[ "$N" -eq "$ESPERADO" ] || { echo "PARCIAL: $N legendas Qwen contra $ESPERADO do estudante"; exit 1; }
echo "Qwen: $N legendas, coorte completa"

$PY experiments/build_text_scores.py --dataset "$DS" --captions "$CAPS" \
    --output-dir results/text_scores_qwen | tee logs/text_scores_qwen_${DS}.log
