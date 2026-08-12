#!/usr/bin/env bash
set -euo pipefail

cd /exp/duducampos/Paintings
PYTHON=.venv-rtx3090-py312/bin/python
SAMPLE=paper/super_resolution_3k_sample.csv
FEATURES=paper/square_graph_3k_features.parquet

"$PYTHON" paper/extract_square_graph_features.py \
  --sample "$SAMPLE" --output "$FEATURES" --workers 32 --checkpoint-every 1000
"$PYTHON" paper/analyze_line_graph_redundancy.py \
  --sample "$SAMPLE" --line-features "$FEATURES" \
  --output paper/square_graph_3k_redundancy.csv
"$PYTHON" paper/evaluate_square_graph_3k.py \
  --sample "$SAMPLE" --square-features "$FEATURES" \
  --line-features paper/line_graph_3k_features.parquet \
  --output paper/square_graph_3k_results.csv
"$PYTHON" paper/rank_line_graph_3k_features.py \
  --sample "$SAMPLE" --line-features "$FEATURES" \
  --output paper/square_graph_3k_feature_ranking.csv

echo "square-graph 3k pipeline complete"
