#!/usr/bin/env bash
# Wait for the validated spectral array, then evaluate its added branch.
set -euo pipefail

while [[ ! -f outputs/hilbert_spectrum_1d_full/features.npy ]]; do
    sleep 30
done

exec .venv/bin/python experiments/evaluate_multiscale_graph_fourier_spectrum1d.py
