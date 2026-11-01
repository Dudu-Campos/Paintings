"""Compact Gram-style descriptors inspired by Gatys et al. (2015).

The related implementation ``tjwhitaker/a-neural-algorithm-of-artistic-style``
uses ImageNet VGG-19, inserts style losses after its first five convolutional
layers (``conv_1`` through ``conv_5``), and compares their Gram matrices.
Those full matrices total 610,560 values per image and are impractical for
WikiArt-scale cross-validation.  Here every activation is first projected to a
fixed 32-dimensional orthonormal channel subspace, then its normalized Gram
matrix's upper triangle is retained.  This gives 5 × 528 = 2,640 deterministic
compact second-order style features per image.

This is a classification adaptation of the style representation, not neural
style-transfer image optimisation.  VGG-19 is entirely frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import Compose, Normalize, ToTensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "extractors"))
from slic_graph_features import resolve_wikiart_records  # noqa: E402
sys.path.insert(0, str(ROOT / "src" / "baselines"))
from train_resnet_wikiart import Letterbox  # noqa: E402


STYLE_CONVOLUTIONS = (1, 2, 3, 4, 5)
SEED = 42


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class ImageDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform) -> None:
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        from PIL import Image
        row = self.records.iloc[index]
        with Image.open(row.path) as image:
            values = self.transform(image.convert("RGB"))
        return values, str(row.stem)


def make_projections(rank: int) -> dict[int, torch.Tensor]:
    """Fixed orthonormal channel projections indexed by VGG output width."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    result = {}
    for channels in (64, 128, 256):
        matrix = torch.randn(channels, rank, generator=generator)
        result[channels] = torch.linalg.qr(matrix, mode="reduced").Q.T.contiguous()
    return result


def compact_gram(values: torch.Tensor, projection: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
    """Projected, normalized Gram upper triangle for a batch of activations."""
    projected = torch.einsum("rc,bchw->brhw", projection, values)
    flattened = projected.flatten(2)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    gram /= projection.shape[0] * flattened.shape[-1]
    return gram[:, triangle[0], triangle[1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument(
        "--cohort-predictions", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp/predictions.csv",
        help="Final MLP predictions; fixes this baseline to its visual 79,207-image cohort.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/gatys_compact_gram_features")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.rank <= 64:
        parser.error("--rank must be in [1, 64]")

    predictions = pd.read_csv(args.cohort_predictions, dtype=str)
    cohort = predictions.loc[predictions.condition.eq("Visual GIST"), ["stem", "true_style"]]
    cohort = cohort.drop_duplicates("stem")
    records = resolve_wikiart_records(args.meta, args.image_dir).drop_duplicates("stem")
    records = records.merge(cohort, on="stem", how="inner", validate="one_to_one")
    transform = Compose([
        Letterbox(224), ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for VGG Gram extraction")
    loader = DataLoader(
        ImageDataset(records, transform), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    projections = {channels: values.to(device) for channels, values in make_projections(args.rank).items()}
    triangle = torch.triu_indices(args.rank, args.rank, device=device)
    n_per_layer = args.rank * (args.rank + 1) // 2
    feature_path = args.output_dir / "features.npy"
    stem_path = args.output_dir / "stems.npy"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32,
        shape=(len(records), len(STYLE_CONVOLUTIONS) * n_per_layer),
    )
    stems: list[str] = []
    offset, started, convolution = 0, time.perf_counter(), 0
    log(
        f"Compact Gatys Gram: {len(records):,} images, rank={args.rank}, "
        f"{len(STYLE_CONVOLUTIONS) * n_per_layer:,} features, device={device}"
    )
    with torch.inference_mode():
        for images, batch_stems in loader:
            values = images.to(device, non_blocking=True)
            pieces = []
            convolution = 0
            for layer in backbone:
                values = layer(values)
                if isinstance(layer, nn.Conv2d):
                    convolution += 1
                    if convolution in STYLE_CONVOLUTIONS:
                        pieces.append(compact_gram(values, projections[values.shape[1]], triangle))
                        if convolution == STYLE_CONVOLUTIONS[-1]:
                            break
            batch_features = torch.cat(pieces, dim=1).cpu().numpy().astype(np.float32, copy=False)
            features[offset:offset + len(batch_features)] = batch_features
            stems.extend(batch_stems)
            offset += len(batch_features)
            if offset % 5_000 < len(batch_features) or offset == len(records):
                rate = offset / (time.perf_counter() - started)
                log(f"{offset:,}/{len(records):,} ({rate:.1f} images/s)")
    features.flush()
    np.save(stem_path, np.asarray(stems))
    metadata = {
        "source": "tjwhitaker/a-neural-algorithm-of-artistic-style",
        "paper": "Gatys, Ecker, Bethge (2015), A Neural Algorithm of Artistic Style",
        "backbone": "VGG-19 ImageNet1K_V1; frozen",
        "style_convolutions": list(STYLE_CONVOLUTIONS),
        "input": "letterbox 224 with ImageNet normalization",
        "projection": "fixed orthonormal random channel projection before Gram",
        "rank": args.rank, "seed": SEED,
        "features_per_layer": n_per_layer, "feature_dim": features.shape[1],
        "n_images": len(records),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log(f"Saved compact Gram features to {args.output_dir}")


if __name__ == "__main__":
    main()
