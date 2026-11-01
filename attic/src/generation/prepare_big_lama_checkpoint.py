#!/usr/bin/env python3
"""Extract a tensor-only Big-LaMa generator checkpoint.

The official checkpoint was produced by an old PyTorch Lightning release and
contains a serialized callback.  Runtime inference does not need that object.
This script loads the trusted official checkpoint once and writes only the
generator tensors plus provenance, so subsequent loads can use
``weights_only=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import types
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_lightning_checkpoint_stub() -> None:
    """Provide the single legacy class referenced by the official archive."""
    lightning = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    checkpoint_module = types.ModuleType(
        "pytorch_lightning.callbacks.model_checkpoint"
    )

    class ModelCheckpoint:
        pass

    ModelCheckpoint.__module__ = checkpoint_module.__name__
    checkpoint_module.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = checkpoint_module
    lightning.callbacks = callbacks
    sys.modules[lightning.__name__] = lightning
    sys.modules[callbacks.__name__] = callbacks
    sys.modules[checkpoint_module.__name__] = checkpoint_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--trust-official-checkpoint",
        action="store_true",
        help="Required because the one-time legacy load uses pickle.",
    )
    args = parser.parse_args()

    if not args.trust_official_checkpoint:
        parser.error("--trust-official-checkpoint is required")

    install_lightning_checkpoint_stub()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("generator."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("generator.")
    }
    if not state:
        raise RuntimeError("No generator tensors found in checkpoint")

    payload = {
        "format": "big-lama-generator-state-v1",
        "generator_state_dict": state,
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_commit": args.source_commit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved={args.output}")
    print(f"tensors={len(state)}")
    print(f"source_sha256={payload['source_checkpoint_sha256']}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
