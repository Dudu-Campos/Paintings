#!/usr/bin/env bash
# Retoma as legendas do rijksdraw depois da reextracao das congeladas.
# A primeira tentativa morreu em `RP-T-1940-571.jpg`, corrompido; o chamador
# agora legenda apenas a coorte do registry, que ja o exclui.
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3
wait_for() { while pgrep -af "$1" | grep python3 | grep -qv "bash -c"; do sleep 60; done; }

wait_for extract_cnn_full
echo "=== [$(date '+%F %T')] extracao terminou"

$PY experiments/evaluate_student_captions_multitask.py \
    --dataset rijksdraw --meta data/rijksdraw_meta.csv \
    --image-dir data/rijksmuseum/drawing || { echo "FALHOU: legendas"; exit 1; }

$PY - <<'PYEOF'
import json
from pathlib import Path
import pandas as pd
csv, jsonl = Path("results/rijksdraw_student_captions.csv"), Path("results/rijksdraw_student_captions.jsonl")
if not jsonl.exists() and csv.exists():
    f = pd.read_csv(csv, dtype=str)
    col = "caption" if "caption" in f else f.columns[-1]
    with jsonl.open("w", encoding="utf-8") as out:
        for _, r in f.iterrows():
            out.write(json.dumps({"stem": str(r["stem"]), "caption": str(r[col])}) + "\n")
    print(f"{len(f):,} legendas -> {jsonl}")
PYEOF

echo "=== [$(date '+%F %T')] 2x2 grafo x texto: rijksdraw"
$PY experiments/evaluate_logit_add_text_multitask.py --dataset rijksdraw || echo "FALHOU: 2x2"
echo "=== [$(date '+%F %T')] FIM"
