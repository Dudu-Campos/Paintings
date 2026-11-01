#!/usr/bin/env bash
# Start the HVG comparison only after its restart-safe feature export is complete.
set -euo pipefail

while [[ ! -f outputs/hilbert_visibility_graph_full/features.npy ]]; do
    sleep 30
done

exec .venv/bin/python experiments/evaluate_multiscale_graph_fourier_hvg.py
