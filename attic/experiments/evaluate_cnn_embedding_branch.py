"""O embedding da CNN treinada do zero como 13º branch.

Pergunta: a representação que a CNN aprendeu sozinha (42,3% de BAcc) soma алgo
ao pipeline hand-designed, ou as duas veem a mesma coisa?

**O cuidado que domina o desenho.** A CNN do zero foi treinada numa coorte de
79.902 imagens e o pipeline vive na canônica de 79.207. Como o `StratifiedKFold`
particiona sobre conjuntos diferentes, as duas partições concordam em **20,0%**
-- exatamente o acaso. Usar os folds canônicos com o embedding da CNN colocaria
**63.372 das 79.207 imagens de teste dentro do treino da CNN**, e o resultado
seria puro vazamento de rótulo.

Por isso este script usa a **partição da CNN** (lida de
`outputs/cnn_scratch/fold_0*_predictions.csv`) e **não usa o bloco Text**: Text
é o único bloco cujo valor depende de scores OOF calculados sobre a partição
canônica, e reaproveitá-lo sob outra partição vazaria pelo outro lado. Todos os
demais blocos são features label-free, então valem sob qualquer partição.

**Viés residual, declarado.** Para o fold k, o checkpoint k nunca viu os rótulos
do teste de k -- esse lado está limpo. Mas as features de *treino* saem do mesmo
checkpoint que treinou naquelas imagens, então são "boas demais" (memorizadas).
O efeito disso é a rede aprender a confiar demais no branch e cair no teste:
o viés é **contra** o branch da CNN, não a favor. Corrigir de verdade exigiria
OOF aninhado, ou seja 25 treinos de CNN -- fora de orçamento. Um resultado
positivo aqui é, portanto, conservador; um resultado nulo é ambíguo.

Saída: outputs/cnn_embedding_branch/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src/baselines"))
sys.path.insert(0, str(ROOT / "src/pipeline"))

from evaluate_branch_architecture import (BranchMLP, log, scaled,  # noqa: E402
                                          seed_everything, train_arm)
from evaluate_felzenszwalb_line_graph import load_felz_line  # noqa: E402
from evaluate_full_factorial_ablation_text import FACTORS, selected_blocks  # noqa: E402
from evaluate_multiscale_by_family import (BASELINE_MASK,  # noqa: E402
                                           family_quadrant_blocks)
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from final_tfidf_visual_mlp import SEED, load_dataset  # noqa: E402
from slic_graph_features import resolve_wikiart_records  # noqa: E402
from train_resnet_wikiart import IMAGENET_MEAN, IMAGENET_STD, Letterbox  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageSet(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def embed(checkpoint: Path, paths, n_classes: int, batch: int, workers: int):
    """2.048d do penúltimo estágio, com o pré-processamento do treino."""
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # `train_resnet_wikiart` grava um dicionário com metadados; os pesos ficam
    # em `model_state_dict`, ao lado de `classes`, `config`, `fold` etc.
    if isinstance(state, dict) and "model_state_dict" in state:
        log(f"    checkpoint fold {state.get('fold')}, época {state.get('best_epoch')}, "
            f"BAcc de parada {state.get('early_stop_balanced_accuracy', float('nan')):.4f}")
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.fc = nn.Identity()
    model = model.to(DEVICE).eval()

    transform = transforms.Compose([
        Letterbox(224), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    loader = DataLoader(ImageSet(paths, transform), batch_size=batch,
                        shuffle=False, num_workers=workers, pin_memory=True)
    features = np.zeros((len(paths), 2048), dtype=np.float32)
    offset, started = 0, time.perf_counter()
    with torch.no_grad():
        for images in loader:
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                output = model(images.to(DEVICE, non_blocking=True)).float().cpu().numpy()
            features[offset:offset + len(output)] = output
            offset += len(output)
            if offset % 20000 < batch:
                log(f"    {offset:,}/{len(paths):,} "
                    f"({offset / (time.perf_counter() - started):.0f} img/s)")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-dir", type=Path, default=ROOT / "outputs/cnn_scratch")
    parser.add_argument("--quadrant-dir", type=Path,
                        default=ROOT / "outputs/wikiart_quadrant_features")
    parser.add_argument("--line-features", type=Path,
                        default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--felz-line-features", type=Path,
                        default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--felz-features", type=Path,
                        default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--fourier-features", type=Path,
                        default=ROOT / "results/fourier_full_features.parquet")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "wikiart")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/cnn_embedding_branch")
    parser.add_argument("--folds", type=int, nargs="+", default=[1])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, groups = load_dataset(args.quadrant_dir)
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, args.line_features, args.felz_features)
    frame, fourier_columns = load_fourier(frame, args.fourier_features)
    frame, felzline_columns = load_felz_line(frame, args.felz_line_features)

    group_blocks = {name: frame[columns].to_numpy(dtype=np.float32)
                    for name, columns in groups.items()}
    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(dtype=str))
    n_classes = len(np.unique(labels))
    stems = frame["stem"].to_numpy(dtype=str)

    # Sem Text: o cache OOF dele é da partição canônica (ver docstring). O bit
    # é desligado na máscara em vez de filtrado depois, senão `selected_blocks`
    # exige a entrada "Text" no dicionário e quebra. BASELINE_MASK=127 tem os
    # sete fatores; Text é o bit 6.
    text_bit = 1 << FACTORS.index("Text")
    blocks, names = selected_blocks(BASELINE_MASK & ~text_bit, group_blocks, {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32)})
    quadrants, quadrant_names = family_quadrant_blocks("Descriptors", group_blocks)
    blocks = list(blocks) + quadrants + [
        frame[felzline_columns].to_numpy(dtype=np.float32)]
    names = list(names) + quadrant_names + ["Line graph (Felz)"]
    log(f"base: {sum(b.shape[1] for b in blocks)}d em {len(blocks)} branches "
        f"({', '.join(names)})")

    # Partição da CNN, lida das predições por fold.
    predictions = pd.concat([pd.read_csv(p) for p in
                             sorted(args.cnn_dir.glob("fold_*_predictions.csv"))])
    fold_of = dict(zip(predictions["stem"].astype(str), predictions["fold"]))
    assigned = np.array([fold_of.get(s, -1) for s in stems])
    if (assigned < 0).any():
        log(f"! {int((assigned < 0).sum())} stems sem fold da CNN; descartados")

    records = resolve_wikiart_records(args.meta, args.image_dir).drop_duplicates("stem")
    path_of = dict(zip(records["stem"].astype(str), records["path"]))
    paths = np.array([path_of[s] for s in stems])

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []

    # Embedding OOF: cada imagem é featurizada pelo checkpoint do fold em cujo
    # TESTE ela caiu, então nenhuma linha vem de uma CNN que viu o rótulo dela.
    # É a mesma construção do bloco Text (`fit_text_scores`).
    #
    # A versão ingênua -- featurizar tudo com um checkpoint só -- foi medida e
    # é inutilizável: a validação interna sai do treino, onde a CNN memorizou,
    # então a BAcc de validação vai a 0,849 e a parada antecipada passa a
    # otimizar exploração de memorização. O teste desabou 5,91pp com acurácia
    # bruta SUBINDO, assinatura de fuga para o prior. Ver fold_results.csv.
    oof_cache = args.output_dir / "embed_oof.npy"
    if oof_cache.exists():
        embedding = np.load(oof_cache)
        log(f"embedding OOF em cache {embedding.shape}")
    else:
        embedding = np.zeros((len(stems), 2048), dtype=np.float32)
        for fold in sorted(set(assigned[assigned > 0])):
            rows_k = np.where(assigned == fold)[0]
            checkpoint = args.cnn_dir / f"fold_{fold:02d}_best.pt"
            log(f"OOF: {len(rows_k):,} imagens do teste do fold {fold} "
                f"por {checkpoint.name}")
            embedding[rows_k] = embed(checkpoint, paths[rows_k], n_classes,
                                      args.batch, args.workers)
        np.save(oof_cache, embedding)
        log(f"embedding OOF salvo {embedding.shape}")

    for fold in args.folds:

        test = np.where(assigned == fold)[0]
        train = np.where((assigned != fold) & (assigned > 0))[0]
        log(f"fold {fold}: {len(train):,} treino / {len(test):,} teste "
            f"(partição da CNN)")

        rng = np.random.default_rng(SEED + fold)
        inner = rng.permutation(len(train))
        cut = int(0.9 * len(train))
        fit, validation = train[inner[:cut]], train[inner[cut:]]

        for condition, chosen in (("base", blocks),
                                  ("+ CNN embed", blocks + [embedding])):
            started = time.perf_counter()
            train_x, validation_x, test_x = scaled(chosen, fit, validation, test)
            seed_everything(SEED + fold)
            model = BranchMLP([b.shape[1] for b in chosen], n_classes).to(DEVICE)
            metrics = train_arm(model, train_x, labels[fit], validation_x,
                                labels[validation], test_x, labels[test],
                                n_classes, SEED + fold, args.epochs, args.patience)
            rows.append({"condition": condition, "fold": fold,
                         "dims": sum(b.shape[1] for b in chosen),
                         "branches": len(chosen), **metrics,
                         "elapsed": time.perf_counter() - started})
            pd.DataFrame(rows).to_csv(result_path, index=False, float_format="%.6f")
            log(f"  {condition:<12} BAcc={100 * metrics['bacc']:5.2f}%  "
                f"{rows[-1]['dims']}d  ({time.perf_counter() - started:.0f}s)")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby("condition", as_index=False)
               .agg(dims=("dims", "first"), branches=("branches", "first"),
                    bacc=("bacc", "mean"), std=("bacc", "std"),
                    macro_f1=("macro_f1", "mean")))
    summary[["bacc", "std", "macro_f1"]] *= 100
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.4f")
    log("\n=== embedding da CNN do zero como branch ===")
    log(summary.round(2).to_string(index=False))
    if detail["condition"].nunique() == 2:
        wide = detail.pivot(index="fold", columns="condition", values="bacc") * 100
        delta = wide["+ CNN embed"] - wide["base"]
        log(f"\ndelta {delta.mean():+.2f}pp  "
            f"({int((delta > 0).sum())}/{len(delta)} folds positivos)")


if __name__ == "__main__":
    main()
