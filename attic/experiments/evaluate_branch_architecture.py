"""As branches valem alguma coisa? Quatro arquiteturas sobre o mesmo vetor.

O pipeline codifica cada bloco num encoder próprio antes de concatenar, e essa
escolha carrega duas conclusões do projeto: o achado 17 (remover duplicatas
entre branches PIOROU) foi explicado por independência dos encoders, e o achado
16 manteve o MLP ingênuo como instrumento justamente por essa estrutura. Nenhum
dos dois se sustenta se a estrutura não fizer diferença.

Já existia um teste (`slic_branch_mlp_fusion.py`), mas na configuração de menor
poder: **SLIC sozinho**, onde os "branches" eram os quatro grafos da mesma
modalidade -- blocos homogêneos, mesma escala, mesma natureza. Deu empate
(45,78 flat contra 45,85 branch) com desvio 3x menor nas branches. Aqui o teste
é sobre os 12 branches heterogêneos da configuração de produção (1.154d), que é
onde a separação deveria importar.

O ponto que torna a comparação nítida: **branch é uma primeira camada
bloco-diagonal**, não uma arquitetura diferente. Com largura de fusão 1.536
(12 x 128), a densa equivalente tem 1.154x1.536 = 1.772.544 parâmetros contra
1.154x128 = 147.712 da bloco-diagonal -- 12x mais. Então "flat contra branch"
confunde estrutura com capacidade, e precisa de mais de dois braços:

    branch      bloco-diagonal 1154->1536, agrupamento semântico
    shuffled    IDÊNTICA em arquitetura e nº de parâmetros, colunas permutadas
                entre os blocos. Isola *agrupamento* de *esparsidade* -- se
                empatar com `branch`, a estrutura de bloco é regularização e
                não tem nada a ver com as modalidades significarem coisas.
    dense       densa 1154->1536, mesma largura de fusão, 12x os parâmetros.
                Isola *esparsidade* com a capacidade a favor da densa.
    narrow      densa 1154->128, o "flat MLP" ingênuo, com a cabeça recebendo
                128 em vez de 1.536. Menos parâmetros que todos.

Todos os braços compartilham o mesmo laço de treino, os mesmos folds, o mesmo
split interno de validação e o mesmo critério de parada -- o laço está copiado
aqui de propósito, para que uma mudança futura em `evaluate_robustness_mlp_aug`
não desloque um braço em relação aos outros no meio da comparação.

Nota: a configuração inclui o bloco Text, porque o alvo é a arquitetura da
config a que o número de 54,42% se refere. Text não entra na avaliação final do
paper, mas isso não muda a pergunta arquitetural.

Saída: outputs/branch_architecture/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_felzenszwalb_line_graph import load_felz_line  # noqa: E402
from evaluate_full_factorial_ablation_text import (load_text_block,  # noqa: E402
                                                   selected_blocks)
from evaluate_multiscale_by_family import (BASELINE_MASK,  # noqa: E402
                                           family_quadrant_blocks)
from evaluate_multiscale_graph_fourier import load_fourier  # noqa: E402
from evaluate_multiscale_graph_fusion import load_graph_blocks  # noqa: E402
from final_tfidf_visual_mlp import N_FOLDS, SEED, load_dataset  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_WIDTH = 128


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_head(width: int, n_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.35),
        nn.Linear(128, n_classes))


class BranchMLP(nn.Module):
    """Primeira camada bloco-diagonal: um encoder por bloco, depois concatena."""

    def __init__(self, branch_sizes: list[int], n_classes: int) -> None:
        super().__init__()
        self.branch_sizes = branch_sizes
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(size, ENCODER_WIDTH),
                          nn.BatchNorm1d(ENCODER_WIDTH), nn.GELU(), nn.Dropout(0.25))
            for size in branch_sizes])
        self.head = make_head(ENCODER_WIDTH * len(branch_sizes), n_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        pieces = torch.split(values, self.branch_sizes, dim=1)
        encoded = [encoder(piece) for encoder, piece in zip(self.encoders, pieces)]
        return self.head(torch.cat(encoded, dim=1))


class DenseMLP(nn.Module):
    """A mesma coisa sem a restrição bloco-diagonal, na largura que se pedir."""

    def __init__(self, n_features: int, n_classes: int, width: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, width), nn.BatchNorm1d(width),
            nn.GELU(), nn.Dropout(0.25))
        self.head = make_head(width, n_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(values))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scaled(blocks, fit, validation, test):
    """Padroniza por coluna com ajuste só no `fit`, e concatena."""
    pieces = []
    for block in blocks:
        scaler = StandardScaler().fit(block[fit])
        pieces.append((scaler.transform(block[fit]).astype(np.float32),
                       scaler.transform(block[validation]).astype(np.float32),
                       scaler.transform(block[test]).astype(np.float32)))
    return tuple(np.concatenate([p[i] for p in pieces], axis=1) for i in range(3))


def train_arm(model, train_x, train_y, validation_x, validation_y,
              test_x, test_y, n_classes, seed, max_epochs, patience):
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    weights = (1 / counts) / (1 / counts).mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).long()),
        batch_size=4096, shuffle=True, num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(seed))
    validation_tensor = torch.from_numpy(validation_x).to(DEVICE)
    test_tensor = torch.from_numpy(test_x).to(DEVICE)

    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = model(validation_tensor).argmax(1).cpu().numpy()
        score = balanced_accuracy_score(validation_y, prediction)
        if score > best_score:
            best_score, best_epoch, stale = float(score), epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(test_tensor).argmax(1).cpu().numpy()
    metrics = {"bacc": balanced_accuracy_score(test_y, prediction),
               "accuracy": accuracy_score(test_y, prediction),
               "macro_f1": f1_score(test_y, prediction, average="macro"),
               "validation_bacc": best_score, "best_epoch": best_epoch,
               "parameters": sum(p.numel() for p in model.parameters())}
    del model, validation_tensor, test_tensor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--text-cache-dir", type=Path,
                        default=ROOT / "outputs/tfidf_text_cache")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/branch_architecture")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
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
    fixed_extra = {
        "Line graph":   frame[line_columns].to_numpy(dtype=np.float32),
        "Felzenszwalb": frame[felz_columns].to_numpy(dtype=np.float32),
        "Fourier":      frame[fourier_columns].to_numpy(dtype=np.float32),
    }
    felzline_block = frame[felzline_columns].to_numpy(dtype=np.float32)
    descriptor_quadrants, quadrant_names = family_quadrant_blocks(
        "Descriptors", group_blocks)

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    result_path = args.output_dir / "fold_results.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(str(r["arm"]), int(r["fold"])) for r in rows}

    for fold, (outer_train, outer_test) in enumerate(splits, start=1):
        text_block = load_text_block(args.text_cache_dir, fold, frame, n_classes)
        blocks, names = selected_blocks(
            BASELINE_MASK, group_blocks, {**fixed_extra, "Text": text_block})
        blocks = blocks + descriptor_quadrants + [felzline_block]
        names = names + quadrant_names + ["Line graph (Felz)"]
        sizes = [block.shape[1] for block in blocks]
        total = sum(sizes)
        if fold == 1:
            log(f"{len(frame):,} imagens | {total}d | {len(blocks)} branches")
            log("  " + ", ".join(f"{n}:{s}" for n, s in zip(names, sizes)))

        fit, validation = train_test_split(
            outer_train, test_size=0.1, stratify=labels[outer_train],
            random_state=SEED + fold)
        train_x, validation_x, test_x = scaled(blocks, fit, validation, outer_test)

        # Permutação de colunas: mesmos tamanhos de bloco, agrupamento destruído.
        # Uma por fold, para a média não depender de uma partição sortuda.
        permutation = np.random.default_rng(SEED + fold).permutation(total)

        arms = {
            "branch":   ("branch", sizes, None),
            "shuffled": ("branch", sizes, permutation),
            "dense":    ("dense", ENCODER_WIDTH * len(sizes), None),
            "narrow":   ("dense", ENCODER_WIDTH, None),
        }
        for arm, (kind, spec, shuffle) in arms.items():
            if (arm, fold) in done:
                continue
            started = time.perf_counter()
            seed_everything(SEED + fold)
            if kind == "branch":
                model = BranchMLP(list(spec), n_classes).to(DEVICE)
                columns = permutation if shuffle is not None else slice(None)
            else:
                model = DenseMLP(total, n_classes, int(spec)).to(DEVICE)
                columns = slice(None)
            metrics = train_arm(
                model,
                np.ascontiguousarray(train_x[:, columns]), labels[fit],
                np.ascontiguousarray(validation_x[:, columns]), labels[validation],
                np.ascontiguousarray(test_x[:, columns]), labels[outer_test],
                n_classes, SEED + fold, args.epochs, args.patience)
            rows.append({"arm": arm, "fold": fold, "dims": total,
                         "branches": len(sizes) if kind == "branch" else 1,
                         **metrics, "elapsed": time.perf_counter() - started})
            pd.DataFrame(rows).to_csv(result_path, index=False, float_format="%.6f")
            log(f"  fold {fold} {arm:<9} BAcc={100 * metrics['bacc']:5.2f}%  "
                f"{metrics['parameters']:>9,} par  "
                f"({time.perf_counter() - started:.0f}s)")

    detail = pd.DataFrame(rows)
    summary = (detail.groupby("arm", as_index=False)
               .agg(parameters=("parameters", "first"),
                    bacc=("bacc", "mean"), std=("bacc", "std"),
                    accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                    epochs=("best_epoch", "mean")))
    summary[["bacc", "std", "accuracy", "macro_f1"]] *= 100
    summary = summary.sort_values("bacc", ascending=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.4f")
    log("\n=== arquiteturas sobre o mesmo vetor ===")
    log(summary.round(2).to_string(index=False))

    wide = detail.pivot(index="fold", columns="arm", values="bacc") * 100
    log("\n=== contra o branch real (pareado por fold) ===")
    comparisons = []
    for arm in [a for a in wide.columns if a != "branch"]:
        delta = wide[arm] - wide["branch"]
        comparisons.append({
            "arm": arm, "delta_pp": delta.mean(), "std": delta.std(),
            "p_paired": ttest_rel(wide[arm], wide["branch"])[1],
            "folds_positive": f"{int((delta > 0).sum())}/{len(delta)}"})
    table = pd.DataFrame(comparisons)
    table.to_csv(args.output_dir / "vs_branch.csv", index=False, float_format="%.4f")
    log(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
