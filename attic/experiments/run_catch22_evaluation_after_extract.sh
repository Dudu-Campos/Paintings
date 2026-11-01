#!/usr/bin/env bash
# Start the paired MLP evaluation only once the resumable Catch22 extraction
# has written its validated final arrays.
set -euo pipefail

while [[ ! -f outputs/hilbert_catch22_full/features.npy ]]; do
    sleep 30
done

exec .venv/bin/python experiments/evaluate_multiscale_graph_catch22.py
