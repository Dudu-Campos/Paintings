#!/usr/bin/env bash
set -euo pipefail

cd /exp/duducampos/Paintings

while systemctl --user is-active --quiet paintings-qwen-sentence-ladder.service; do
    echo "[$(date '+%F %T')] waiting for paintings-qwen-sentence-ladder.service"
    sleep 60
done

/exp/duducampos/Paintings/.venv-rtx3090-py312/bin/python \
    /exp/duducampos/Paintings/src/affect/generate_catr_coco_wikiart.py \
    --device cuda \
    --batch-size 16 \
    --max-length 40 \
    --log-every 100

exec /exp/duducampos/Paintings/.venv-rtx3090-py312/bin/python \
    /exp/duducampos/Paintings/experiments/evaluate_catr_coco.py
