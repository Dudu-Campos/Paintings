"""Quanto o ImageNet já resolve o corpus, sem treinar nada (§4.2).

Aplica a `fc` pré-treinada da ResNet50 sobre as features congeladas que já estão
em disco -- um matmul. Mede se a CNN está **dentro do domínio de treino dela**,
o que é diferente de sobreposição de rótulo: no Food-101 só 3 das 101 classes
têm nome no ImageNet, mas **92,12% dos top-1 caem em classes de comida/mesa**;
no WikiArt são 3,88%, e a resposta modal é `book jacket`.

É não-supervisionado e calculável antes de treinar qualquer coisa, então serve
para registrar previsão em vez de explicar resultado depois.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    from evaluate_external_modalities import load_block
    from torchvision.models import ResNet50_Weights, resnet50

    block = load_block(args.dataset, "ResNet50")
    if block is None:
        raise SystemExit(f"{args.dataset}: ResNet50 não extraída")
    values = block[[c for c in block.columns if c.startswith("ResNet50_")]].to_numpy(np.float32)

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    categories = weights.meta["categories"]
    logits = values @ model.fc.weight.detach().numpy().T + model.fc.bias.detach().numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), 1).numpy()
    top1 = logits.argmax(1)

    print(f"{args.dataset}: {len(values):,} imagens")
    print(f"  confiança média do top-1: {100 * probabilities.max(1).mean():.2f}%")
    print(f"  entropia média (bits):    {-(probabilities * np.log2(probabilities + 1e-12)).sum(1).mean():.2f} "
          f"(máximo 9,97)")
    counts = pd.Series([categories[i] for i in top1]).value_counts()
    print(f"  classes distintas previstas: {len(counts)} de 1000")
    print(f"  massa no top-{args.top} de classes previstas: "
          f"{100 * counts.head(args.top).sum() / len(top1):.2f}%")
    print(f"  classes mais previstas:")
    for name, n in counts.head(args.top).items():
        print(f"    {name:<28} {100 * n / len(top1):5.2f}%")


if __name__ == "__main__":
    main()
