#!/usr/bin/env bash
set -euo pipefail

cd /exp/duducampos/Paintings
PYTHON=.venv-rtx3090-py312/bin/python
SAMPLE=paper/line_graph_full_sample.csv
FEATURES=paper/line_graph_full_features.parquet

"$PYTHON" paper/select_line_graph_full_sample.py
"$PYTHON" paper/extract_line_graph_features.py \
  --sample "$SAMPLE" --output "$FEATURES" --workers 32 --checkpoint-every 1000
"$PYTHON" paper/analyze_line_graph_redundancy.py \
  --sample "$SAMPLE" --line-features "$FEATURES" \
  --output paper/line_graph_full_redundancy.csv
"$PYTHON" paper/evaluate_line_graph_3k.py \
  --sample "$SAMPLE" --line-features "$FEATURES" \
  --output paper/line_graph_full_results.csv
"$PYTHON" paper/rank_line_graph_3k_features.py \
  --sample "$SAMPLE" --line-features "$FEATURES" \
  --output paper/line_graph_full_feature_ranking.csv

echo "line-graph full pipeline complete"
