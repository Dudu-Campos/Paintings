#!/usr/bin/env bash
# Completa as colunas do rijksdraw que nao dependem do fine-tuning (que roda na
# margarita, §9.8): grafo aditivo e canal de texto.
#
# O logit-add aqui e PREVISAO REGISTRADA: 11 classes dao bloco concatenado de
# 2C+32 = 54d, e a regra do §5.6 preve que o CONCATENADO vence (ele ja da +1,12).
# E o terceiro ponto da regra, num corpus que nao foi usado para deriva-la.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3

echo "=== [$(date '+%F %T')] logit-add: rijksdraw"
$PY experiments/evaluate_logit_add_multitask.py --dataset rijksdraw \
    --output results/rijksdraw_logit_add_metrics.csv || echo "FALHOU: logit-add"

echo "=== [$(date '+%F %T')] legendas do estudante: rijksdraw"
$PY experiments/evaluate_student_captions_multitask.py \
    --dataset rijksdraw --meta data/rijksdraw_meta.csv \
    --image-dir data/rijksmuseum/drawing || { echo "FALHOU: legendas"; exit 1; }

$PY - <<'PYEOF'
import json
from pathlib import Path
import pandas as pd
csv, jsonl = Path("results/rijksdraw_student_captions.csv"), Path("results/rijksdraw_student_captions.jsonl")
if not jsonl.exists() and csv.exists():
    frame = pd.read_csv(csv, dtype=str)
    col = "caption" if "caption" in frame else frame.columns[-1]
    with jsonl.open("w", encoding="utf-8") as out:
        for _, r in frame.iterrows():
            out.write(json.dumps({"stem": str(r["stem"]), "caption": str(r[col])}) + "\n")
    print(f"{len(frame):,} legendas -> {jsonl}")
PYEOF

echo "=== [$(date '+%F %T')] 2x2 grafo-aditivo x texto: rijksdraw"
$PY experiments/evaluate_logit_add_text_multitask.py --dataset rijksdraw \
    || echo "FALHOU: 2x2"
echo "=== [$(date '+%F %T')] FIM"
