"""Extract paired frozen VGG-19 pooled controls for the compact Gatys baseline.

Two outputs are stored in one forward pass on the 79,207-image visual cohort:

* ``conv5_gap``: global-average pooled activations after the fifth convolution,
  exactly where the related Gatys implementation takes its last style Gram;
* ``final_gap``: global-average pooled final VGG-19 feature map, the standard
  frozen VGG encoder control.

Neither output uses a Gram matrix or updates VGG weights.
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
from torch.utils.data import DataLoader
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import Compose, Normalize, ToTensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from slic_graph_features import resolve_wikiart_records  # noqa: E402
from extract_gatys_compact_gram import ImageDataset  # noqa: E402
from train_resnet_wikiart import Letterbox  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument(
        "--cohort-predictions", type=Path,
        default=ROOT / "outputs/final_tfidf_visual_mlp/predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/vgg_pool_features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

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
        raise RuntimeError("CUDA is required for VGG extraction")
    loader = DataLoader(
        ImageDataset(records, transform), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    early = np.lib.format.open_memmap(
        args.output_dir / "conv5_gap_features.npy", mode="w+", dtype=np.float32,
        shape=(len(records), 256),
    )
    final = np.lib.format.open_memmap(
        args.output_dir / "final_gap_features.npy", mode="w+", dtype=np.float32,
        shape=(len(records), 512),
    )
    stems: list[str] = []
    offset, started, convolution = 0, time.perf_counter(), 0
    log(f"Frozen VGG pooled controls: {len(records):,} images, device={device}")
    with torch.inference_mode():
        for images, batch_stems in loader:
            values = images.to(device, non_blocking=True)
            convolution = 0
            early_values = None
            for layer in model:
                values = layer(values)
                if isinstance(layer, nn.Conv2d):
                    convolution += 1
                    if convolution == 5:
                        early_values = values.mean(dim=(2, 3))
            if early_values is None:
                raise RuntimeError("VGG fifth convolution was not reached")
            final_values = values.mean(dim=(2, 3))
            size = len(images)
            early[offset:offset + size] = early_values.cpu().numpy()
            final[offset:offset + size] = final_values.cpu().numpy()
            stems.extend(batch_stems)
            offset += size
            if offset % 5_000 < size or offset == len(records):
                log(f"{offset:,}/{len(records):,} ({offset / (time.perf_counter() - started):.1f} images/s)")
    early.flush()
    final.flush()
    np.save(args.output_dir / "stems.npy", np.asarray(stems))
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "backbone": "VGG-19 ImageNet1K_V1; frozen", "input": "letterbox 224 with ImageNet normalization",
        "early": "global average pooling after VGG convolution #5", "final": "global average pooling after full VGG features",
        "n_images": len(records),
    }, indent=2) + "\n")
    log(f"Saved VGG pooled controls to {args.output_dir}")


if __name__ == "__main__":
    main()
