#!/usr/bin/env python3
"""Create a visual audit montage of the Fourier-spectrum representation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from slic_graph_features import resolve_wikiart_records  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_fourier_spectrum_features import (  # noqa: E402
    ANGLE,
    ANGULAR_BINS,
    ANGULAR_EDGES,
    HANN_WINDOW,
    RADIAL_BINS,
    RADIAL_EDGES,
    RADIUS,
    VALID_FREQUENCY,
    square_luminance,
)


OUTPUT = ROOT / "outputs/fourier_full/fourier_spectrum_examples.png"
STYLES = (
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Impressionism",
    "Minimalism",
    "Pointillism",
    "Pop_Art",
    "Realism",
    "Romanticism",
    "Symbolism",
)


def spectrum_views(luminance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = luminance.astype(np.float64) - float(luminance.mean())
    deviation = float(centered.std())
    if deviation > 1e-8:
        centered /= deviation
    spectrum = np.fft.fftshift(np.fft.fft2(centered * HANN_WINDOW))
    log_power = np.log1p(np.square(np.abs(spectrum)))
    low, high = np.percentile(log_power, (1, 99.7))
    display = np.clip((log_power - low) / max(high - low, 1e-12), 0, 1)
    polar, _, _ = np.histogram2d(
        RADIUS[VALID_FREQUENCY],
        ANGLE[VALID_FREQUENCY],
        bins=(RADIAL_EDGES, ANGULAR_EDGES),
        weights=np.square(np.abs(spectrum))[VALID_FREQUENCY],
    )
    polar /= max(float(polar.sum()), 1e-12)
    polar = np.log10(polar + 1e-12)
    return display, polar


def main() -> None:
    sample = pd.read_csv(ROOT / "results/line_graph_full_sample.csv")
    selected = (
        sample[sample["style"].isin(STYLES)]
        .sort_values("stem")
        .groupby("style", sort=False)
        .first()
        .reset_index()
    )
    selected["style"] = pd.Categorical(selected["style"], STYLES, ordered=True)
    selected = selected.sort_values("style")
    records = resolve_wikiart_records(
        ROOT / "data/artemis/wikiart_meta.csv", ROOT / "wikiart"
    )
    selected = selected.merge(
        records[["stem", "style", "path"]],
        on=["stem", "style"],
        validate="one_to_one",
    )

    figure, axes = plt.subplots(
        len(selected), 3, figsize=(10, 2.55 * len(selected)),
        gridspec_kw={"width_ratios": (1, 1, 1.05)},
    )
    for row_index, row in enumerate(selected.itertuples(index=False)):
        with Image.open(row.path) as image:
            luminance = square_luminance(image)
        spectrum, polar = spectrum_views(luminance)
        axes[row_index, 0].imshow(luminance, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].imshow(spectrum, cmap="magma", vmin=0, vmax=1)
        axes[row_index, 2].imshow(
            polar,
            cmap="magma",
            aspect="auto",
            origin="lower",
            extent=(0, 180, 0, RADIAL_BINS),
        )
        axes[row_index, 0].set_ylabel(str(row.style).replace("_", " "), fontsize=9)
        axes[row_index, 2].set_yticks((0, RADIAL_BINS - 1), ("baixa", "alta"))
        axes[row_index, 2].set_xticks((0, 45, 90, 135, 180))
        axes[row_index, 2].set_xlabel("orientacao (graus)", fontsize=8)
        for column in (0, 1):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 2].tick_params(labelsize=7)
    for axis, title in zip(
        axes[0],
        ("Luminancia 256 x 256", "Log-espectro 2D", "Energia polar: escala x orientacao"),
        strict=True,
    ):
        axis.set_title(title, fontsize=11, pad=8)
    figure.suptitle(
        "Representacao de Fourier usada no experimento",
        fontsize=14,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.992))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
