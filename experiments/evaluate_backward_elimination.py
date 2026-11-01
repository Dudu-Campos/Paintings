"""Eliminação gulosa para trás: triagem barata antes do Shapley.

**Por que não usar LOO direto como critério de remoção.** LOO_i = v(N) − v(N\\i)
só é bem-posto para remover **um** bloco: ele mede a perda supondo que todo o
resto fica. Removendo vários de uma vez a aditividade quebra, e quebra
justamente onde há redundância -- dois blocos que se cobrem têm ambos LOO ~ 0
e remover os dois abre o buraco inteiro. Medido no fatorial atual: a soma dos
LOOs erra a perda real de pares em até **+0,76pp**, e o pior caso é
`Felzenszwalb + L(G_felz)` (soma 1,03, perda real 1,44). Com `Random walk` a
R²=0,953 de `Hilbert`, a regra ingênua apagaria o canal de série temporal
inteiro.

A eliminação para trás não tem esse problema porque **recalcula a cada passo**:
cada decisão é sempre "qual único bloco remover deste conjunto". É um caminho,
e por isso o relatório é a **curva inteira**, não só o ponto final -- onde ela
cai é a informação, e parar num limiar arbitrário jogaria fora essa leitura.

Custo: k + (k−1) + ... + 2 ajustes por fold, contra 2^k do fatorial. Com 12
blocos são 77 contra 4.095.

MLP **sem branches**: a estrutura por bloco daria a cada candidato um encoder
próprio, e a pergunta aqui é sobre a informação do bloco, não sobre como a
arquitetura a compartimenta.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_fusion_delta_across_corpora import SEED, VISUAL, clean, load_corpus  # noqa: E402
from evaluate_neighbour_attention import key  # noqa: E402
from evaluate_redundancy_selection_curve import fit as plain_fit  # noqa: E402

# Blocos que não estão no registro, carregados por caminho + stems.
EXTERNAL = {
    "Random walk": ("outputs/multitask_random_walk/rw_features.npy",
                    "outputs/multitask_random_walk/rw_stems.npy"),
    "HVG": ("outputs/multitask_hilbert_visibility/features.npy",
            "outputs/multitask_hilbert_visibility/stems.npy"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--extra", nargs="*", default=list(EXTERNAL))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--keep-at-least", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/backward_elimination.csv")
    args = parser.parse_args()

    blocks, raw_labels, _ = load_corpus("multitask", VISUAL + ["Gabor"])
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    mask = np.array([key(v) in wanted for v in raw_labels])
    if len(np.unique(raw_labels[mask])) != len(wanted):
        raise ValueError("casamento de classes do benchmark incompleto")
    blocks = {n: clean(b[mask]) for n, b in blocks.items()}
    labels = LabelEncoder().fit_transform(raw_labels[mask])
    n_classes = len(np.unique(labels))

    meta = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})
    # `meta.style` é o acessor Styler do pandas, não a coluna (§8).
    reference = meta[[key(str(v)) in wanted for v in meta["style"]]].stem.to_numpy()
    if len(reference) != len(labels):
        raise ValueError(f"coorte do registro ({len(labels)}) != metadado ({len(reference)})")
    for name in args.extra:
        values_path, stems_path = EXTERNAL[name]
        values = np.load(ROOT / values_path)
        stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
        index = pd.Series(np.arange(len(stems)), index=stems).reindex(reference)
        if index.isna().any():
            raise ValueError(f"{name}: {int(index.isna().sum())} stems ausentes")
        blocks[name] = clean(values[index.to_numpy(dtype=int)])

    order = [n for n in VISUAL + ["Gabor"] if n in blocks] + list(args.extra)
    log(f"{len(labels):,} imagens, {n_classes} classes, {len(order)} blocos: {order}")
    log(f"orçamento: {sum(range(args.keep_at_least, len(order) + 1))} ajustes por fold")

    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels)), labels))[:args.folds]
    scaled = {}
    for fold, (train, _) in enumerate(splits, start=1):
        scaled[fold] = {n: StandardScaler().fit(b[train]).transform(b).astype(np.float32)
                        for n, b in blocks.items()}

    def score(names, fold, train, test):
        stacked = np.hstack([scaled[fold][n] for n in names])
        fit_idx, val_idx = train_test_split(
            train, test_size=0.1, stratify=labels[train], random_state=SEED + fold)
        out = plain_fit(stacked[fit_idx], labels[fit_idx], stacked[val_idx], labels[val_idx],
                        stacked[test], labels[test], n_classes, seed=SEED + fold)
        return out["bacc"]

    rows = []
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
    done = {(int(r["step"]), str(r["removed"]), int(r["fold"])) for r in rows}

    remaining = list(order)
    step = 0
    while len(remaining) > args.keep_at_least:
        # Nível de referência do conjunto atual (removed = "-").
        for fold, (train, test) in enumerate(splits, start=1):
            if (step, "-", fold) in done:
                continue
            value = score(remaining, fold, train, test)
            rows.append({"step": step, "removed": "-", "fold": fold,
                         "n_blocks": len(remaining), "remaining": "+".join(remaining),
                         "dimensions": sum(scaled[fold][n].shape[1] for n in remaining),
                         "bacc": value})
            pd.DataFrame(rows).to_csv(args.output, index=False)
        frame = pd.DataFrame(rows)
        base = frame[frame.step.eq(step) & frame.removed.eq("-")].bacc.mean()
        log(f"passo {step}: {len(remaining)} blocos, BAcc {base:.2f}")

        for candidate in remaining:
            for fold, (train, test) in enumerate(splits, start=1):
                if (step, candidate, fold) in done:
                    continue
                subset = [n for n in remaining if n != candidate]
                value = score(subset, fold, train, test)
                rows.append({"step": step, "removed": candidate, "fold": fold,
                             "n_blocks": len(subset), "remaining": "+".join(subset),
                             "dimensions": sum(scaled[fold][n].shape[1] for n in subset),
                             "bacc": value})
                pd.DataFrame(rows).to_csv(args.output, index=False)
            frame = pd.DataFrame(rows)
            got = frame[frame.step.eq(step) & frame.removed.eq(candidate)].bacc.mean()
            log(f"    sem {candidate:<14} {got:5.2f}  ({got - base:+5.2f})")

        frame = pd.DataFrame(rows)
        options = frame[frame.step.eq(step) & frame.removed.ne("-")]
        victim = options.groupby("removed").bacc.mean().idxmax()
        log(f"  -> remove {victim}")
        remaining = [n for n in remaining if n != victim]
        step += 1

    frame = pd.DataFrame(rows)
    path = frame[frame.removed.eq("-")].groupby("step").agg(
        n_blocks=("n_blocks", "first"), bacc=("bacc", "mean"), remaining=("remaining", "first"))
    log("\ncaminho de eliminação (o relatório é a curva, não o ponto final):")
    log(path.round(2).to_string())


if __name__ == "__main__":
    main()
