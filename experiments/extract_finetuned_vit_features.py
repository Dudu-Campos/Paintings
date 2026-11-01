"""Penúltima camada de um ViT FINE-TUNED, para fusão em nível de features.

Contra backbones congelados medimos a fusão em nível de features (+4 a +5,4pp);
contra os fine-tuned estávamos presos a nível de decisão (+1,4 a +3,5pp), porque
só tínhamos os posteriors. Como o checkpoint por fold está salvo e as imagens
estão em disco local, dá para extrair as features e medir o número forte.

**O pré-processamento tem de casar com o do treino, e o config salvo não
registra `augment`.** Determinado empiricamente em 26/ago: rodando o modelo
carregado sobre o teste do fold 1, `Resize(256, bicubic) + CenterCrop(224)`
reproduz os 67,32% reportados ao centésimo, enquanto `Letterbox(224)` dá 62,41.
Isso fixa a transformação e valida o caminho de carregamento de uma vez -- e de
quebra mede que o pré-processamento sozinho vale **4,91pp** neste modelo.

**Ressalva obrigatória sobre vazamento.** O modelo foi fine-tunado no treino do
fold 1, então as features dele nas linhas de TREINO são de imagens que ele
memorizou (loss 0,637 com augmentação forte, mas ainda assim vistas). As de
TESTE são honestas. Treinar a cabeça de fusão sobre features super-ajustadas faz
ela confiar demais no ViT, e no teste isso custa -- ou seja, o viés é **contra**
a fusão, e o número que sair é conservador. Um caminho limpo exigiria OOF
aninhado com um ViT por sub-fold, que custa 2h por sub-fold.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines"))

from train_resnet_wikiart import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Images(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            return self.transform(Image.open(self.paths[index]).convert("RGB")), True
        except Exception:
            return torch.zeros(3, 224, 224), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/multitask_vit21k_recipe_acc/fold_01_best.pt")
    parser.add_argument("--timm-name", default="vit_base_patch16_224.orig_in21k")
    parser.add_argument("--image-dir", type=Path, default=Path("/tmp/prism_data/multitask/images"))
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--restrict-stems", type=Path,
                        default=ROOT / "data/multitask_25class_cohort_stems.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/multitask_vit21k_ft_features")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import timm
    blob = torch.load(args.checkpoint, map_location="cpu")
    model = timm.create_model(args.timm_name, pretrained=False, num_classes=blob["classes"])
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model = model.cuda().eval()
    log(f"checkpoint: fold {blob['fold']}, época {blob['best_epoch']}, "
        f"{blob['classes']} classes")

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    if args.restrict_stems.exists():
        keep = set(pd.read_csv(args.restrict_stems, dtype={"stem": str})["stem"])
        meta = meta[meta["stem"].isin(keep)]
    meta = meta.drop_duplicates("stem", keep="last").reset_index(drop=True)
    paths = [str(args.image_dir / f) for f in meta["filename"].to_numpy(str)]
    log(f"{len(paths):,} imagens")

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    loader = DataLoader(Images(paths, transform), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)

    features = np.zeros((len(paths), model.num_features), dtype=np.float32)
    failures, cursor, started = 0, 0, time.perf_counter()
    with torch.no_grad():
        for batch, ok in loader:
            out = model.forward_head(model.forward_features(batch.cuda(non_blocking=True)),
                                     pre_logits=True)
            features[cursor:cursor + len(out)] = out.cpu().numpy()
            failures += int((~ok).sum())
            cursor += len(out)
            if cursor % (args.batch_size * 100) == 0:
                rate = cursor / (time.perf_counter() - started)
                log(f"  {cursor:,}/{len(paths):,}  {rate:.0f} img/s  "
                    f"ETA {(len(paths) - cursor) / rate / 60:.1f} min")
    np.save(args.output / "cnn_full_features.npy", features)
    np.save(args.output / "cnn_full_stems.npy", meta["stem"].to_numpy(dtype=object))
    log(f"salvo: {features.shape}, {failures} falhas, "
        f"{(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
