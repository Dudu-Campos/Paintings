#!/usr/bin/env bash
# Espera as legendas do oilpainting e roda o 2x2 (grafo aditivo x texto) nos dois
# corpora de arte. Encadeado, nao paralelo: GPU compartilhada (§1, §8).
set -u
cd "$(dirname "$0")/.."
PY=.ABSvenv/bin/python3

wait_for() {   # filtra por interpretador para nao casar o proprio shell (§8)
  while pgrep -af "$1" | grep python3 | grep -qv "bash -c"; do sleep 30; done
}

wait_for evaluate_student_captions
echo "=== [$(date '+%F %T')] legendas prontas"

# O gerador grava .csv; o consumidor le .jsonl. Converter se preciso.
$PY - <<'PYEOF'
import json
from pathlib import Path
import pandas as pd
for ds in ("oilpainting",):
    csv, jsonl = Path(f"results/{ds}_student_captions.csv"), Path(f"results/{ds}_student_captions.jsonl")
    if jsonl.exists() or not csv.exists():
        print(f"{ds}: jsonl={jsonl.exists()} csv={csv.exists()}, nada a fazer")
        continue
    frame = pd.read_csv(csv, dtype=str)
    col = "caption" if "caption" in frame else frame.columns[-1]
    with jsonl.open("w", encoding="utf-8") as out:
        for _, r in frame.iterrows():
            out.write(json.dumps({"stem": str(r["stem"]), "caption": str(r[col])}) + "\n")
    print(f"{ds}: {len(frame):,} legendas -> {jsonl}")
PYEOF

for ds in multitask oilpainting; do
  echo "=== [$(date '+%F %T')] 2x2 grafo-aditivo x texto: $ds"
  $PY experiments/evaluate_logit_add_text_multitask.py --dataset "$ds" \
    || echo "FALHOU: $ds"
done
echo "=== [$(date '+%F %T')] FIM"
