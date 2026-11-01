"""Layout do grafo k-NN entre pinturas -- figura ILUSTRATIVA do método.

Serve para mostrar o que o branch de grafo enxerga no regime **indutivo**: cada
obra a predizer (cinza) ligada aos seus k vizinhos **rotulados** (coloridos por
estilo), calculados só no treino do fold. A feature é a média das distribuições
de rótulo desses vizinhos -- por isso o desenho é o mecanismo, não uma metáfora.

**Não é figura de resultado**: nenhum número de desempenho sai daqui, e a
pureza por classe (que é resultado) foi removida de propósito para não
misturar os dois papéis.

**Layout determinístico, não dirigido por força.** Cada ego-rede é desenhada
como centro mais anel, e as estrelas são dispostas numa grade. `spring_layout`
é a ferramenta errada aqui: com componentes desconexos ele comprime cada
vizinhança numa bola e os nós se sobrepõem ao centro. O anel ordena os vizinhos
do mais próximo (topo, sentido horário) ao mais distante.

**As sementes são escolhidas, não sorteadas** -- e a legenda da figura precisa
dizer isso. Duas curadorias, ambas para o mecanismo aparecer: as obras a
predizer vêm dos estilos que a paleta cobre (com sementes aleatórias metade das
estrelas fica cinza, que é vizinho de estilo fora das seis fatias e não ilustra
nada); e são estratificadas por pureza, para a figura mostrar tanto a
vizinhança coerente quanto a ambígua em vez de só o caso favorável.

**Amostra por ego-rede, não sorteio simples.** Sortear 3.000 obras de 82.824 e
recalcular o k-NN entre elas daria um grafo diferente do que a produção usa:
com k=30 de 66 mil, duas obras sorteadas quase nunca são vizinhas, e o desenho
mostraria um grafo que não existe. Aqui sorteia-se um conjunto semente e
trazem-se os **vizinhos reais** de cada uma, calculados no espaço completo.

**Seis estilos coloridos, o resto em cinza.** 25 estilos excedem a paleta
categórica validada (máximo 8 fatias); gerar cores novas quebraria a separação
sob daltonismo. Os demais dobram em "outros", que é a regra do método.

O espaço é o de produção: 8 modalidades padronizadas no treino do fold 1, sem
Multiscale (aposentado em 24/ago).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))

from evaluate_external_modalities import BLOCKS, LABELS, load_block  # noqa: E402

SEED, N_FOLDS = 42, 5
# ordem categórica validada (ΔE OKLab adjacente >= 8 sob protan/deutan/tritan)
PALETTE = ["#eda100", "#2a78d6", "#eb6834", "#4a3aa7", "#e87ba4", "#1baf7a"]
OTHER = "#c8c8c4"
SURFACE, GRID, INK, MUTED = "#fbfbfa", "#dedede", "#0b0b0b", "#52514e"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=12)


def load_cohort(dataset: str, classes_file: Path | None, exclude: tuple[str, ...]):
    meta_path, label_column = LABELS[dataset]
    meta = pd.read_csv(ROOT / meta_path, dtype={"stem": str}).dropna(subset=[label_column])
    meta["label"] = meta[label_column].astype(str)

    available = {}
    for name in BLOCKS:
        if name in exclude:
            continue
        block = load_block(dataset, name)
        if block is not None:
            available[name] = block
    frame = meta[["stem", "label"]]
    for name, block in available.items():
        frame = frame.merge(block, on="stem", how="inner")

    if classes_file:
        def normalize(value: str) -> str:
            return value.replace("_", " ").replace("-", " ").replace("ï", "i").strip().lower()
        wanted = {normalize(line) for line in classes_file.read_text().splitlines() if line.strip()}
        by_norm = {normalize(label): label for label in frame["label"].unique()}
        frame = frame[frame["label"].isin([by_norm[n] for n in wanted])]

    frame = frame.reset_index(drop=True)
    blocks = []
    for name in available:
        values = frame[[c for c in frame.columns if c.startswith(f"{name}_")]].to_numpy(np.float32)
        bad = ~np.isfinite(values)
        if bad.any():
            medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
            values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
        blocks.append(values)
    return frame, blocks, list(available)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask", choices=list(LABELS))
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--exclude", nargs="+", default=("ResNet50", "Multiscale"))
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=15,
                        help="obras a predizer; cada uma traz seus k vizinhos reais")
    parser.add_argument("--random-seeds", action="store_true",
                        help="sorteia em vez de curar (ver docstring)")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="stems específicos a usar como obras a predizer, "
                             "na ordem dada. Para a figura de abertura, onde a "
                             "escolha é editorial e não amostral.")
    parser.add_argument("--top-styles", type=int, default=6)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/knn_graph_layout.png")
    args = parser.parse_args()

    frame, blocks, names = load_cohort(args.dataset, args.classes_file, tuple(args.exclude))
    labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
    class_names = np.array(sorted(frame["label"].unique()))
    log(f"{len(frame):,} imagens, {len(class_names)} classes, blocos: {names}")

    # Espaço de produção: padronizado no treino do fold 1, como o grafo real.
    train, test = next(iter(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                            .split(np.zeros(len(labels)), labels)))
    scalers = [StandardScaler().fit(b[train]) for b in blocks]
    space = np.hstack([s.transform(b[train]) for s, b in zip(scalers, blocks)])
    query = np.hstack([s.transform(b[test]) for s, b in zip(scalers, blocks)])
    train_labels, test_labels = labels[train], labels[test]
    log(f"espaço do grafo: {space.shape[1]}d | {len(train):,} rotuladas, "
        f"{len(test):,} a predizer")

    started = time.perf_counter()
    engine = NearestNeighbors(n_neighbors=args.neighbors).fit(space)
    # Indutivo: o ponto de consulta nunca vê outro ponto de consulta -- os
    # vizinhos vêm todos do treino, que é o que a producao faz.
    _, indices = engine.kneighbors(query)
    log(f"k-NN pronto em {time.perf_counter() - started:.0f}s")

    purity = (train_labels[indices] == test_labels[:, None]).mean(axis=1)
    log(f"pureza média: {purity.mean():.3f} (acaso = {1 / len(class_names):.3f})")

    # --- amostra por ego-rede -------------------------------------------
    generator = np.random.default_rng(SEED)
    counts_all = pd.Series(train_labels).value_counts()
    palette_styles = set(counts_all.index[:args.top_styles])
    if args.stems:
        stems_all = frame["stem"].to_numpy(str)
        where = {stem: position for position, stem in enumerate(stems_all[test])}
        missing = [stem for stem in args.stems if stem not in where]
        if missing:
            raise ValueError(f"stems fora do conjunto de consulta do fold 1: {missing}")
        seeds = np.array([where[stem] for stem in args.stems])
        log("sementes explícitas: " + ", ".join(
            f"{stem} ({class_names[test_labels[i]]}, {100 * purity[i]:.0f}%)"
            for stem, i in zip(args.stems, seeds)))
    elif args.random_seeds:
        seeds = generator.choice(len(query), size=args.seeds, replace=False)
    else:
        # Estratificar por terço de pureza dentro dos estilos da paleta: a
        # figura mostra vizinhança coerente E ambígua, sem escolher só o caso
        # favorável.
        eligible = np.flatnonzero([label in palette_styles for label in test_labels])
        order = eligible[np.argsort(purity[eligible])]
        thirds = np.array_split(order, 3)
        per_third = args.seeds // 3
        seeds = np.concatenate([
            generator.choice(part, size=min(per_third, len(part)), replace=False)
            for part in thirds])
        log(f"sementes curadas: {len(seeds)} obras dos {args.top_styles} estilos da "
            f"paleta, estratificadas por pureza "
            f"({100 * purity[seeds].min():.0f}%–{100 * purity[seeds].max():.0f}%)")
    neighbour_nodes = np.unique(indices[seeds].ravel())
    position = {("train", n): i for i, n in enumerate(neighbour_nodes)}
    for offset, s_index in enumerate(seeds):
        position[("query", s_index)] = len(neighbour_nodes) + offset
    edges = [(position[("query", s_index)], position[("train", n)])
             for s_index in seeds for n in indices[s_index]]
    log(f"amostra: {len(seeds)} a predizer + {len(neighbour_nodes):,} rotuladas, "
        f"{len(edges):,} arestas")

    # Com sementes explícitas a paleta cobre os estilos DOS ALVOS: o que a
    # figura precisa mostrar é "os vizinhos compartilham o estilo da obra".
    # Colorir os estilos mais frequentes deixaria Ukiyo-e ou Northern
    # Renaissance em cinza, que é o oposto do argumento.
    if args.stems:
        top = list(dict.fromkeys(test_labels[seeds]))[:len(PALETTE)]
    else:
        top = list(counts_all.index[:args.top_styles])
    colour_of = {label: PALETTE[i] for i, label in enumerate(top)}

    columns = 3 if len(seeds) <= 6 else 5
    rows = int(np.ceil(len(seeds) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, args.neighbors, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    for axis, s_index in zip(axes, seeds):
        for (x, y) in ring:
            axis.plot([0, x], [0, y], color="#b8b8b6", linewidth=0.6, zorder=1)
        neighbours = indices[s_index]
        colours = [colour_of.get(train_labels[n], OTHER) for n in neighbours]
        axis.scatter(ring[:, 0], ring[:, 1], s=68, c=colours, linewidth=0.8,
                     edgecolor="white", zorder=3)
        axis.scatter([0], [0], s=210, color="#3a3a38", edgecolor="white",
                     linewidth=2.0, zorder=4)
        axis.set_xlim(-1.32, 1.32); axis.set_ylim(-1.32, 1.32)
        axis.set_aspect("equal"); axis.axis("off")
    for axis in axes[len(seeds):]:
        axis.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=11,
                          color="#3a3a38", markeredgecolor="white",
                          markeredgewidth=1.6, label="painting to predict")]
    handles += [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                           color=colour_of[label],
                           label=class_names[label].replace("_", " "))
                for label in top]
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                              color=OTHER, label="other styles"))
    figure.legend(handles=handles, frameon=False, fontsize=11.5, ncol=4,
                  loc="lower center", labelcolor=MUTED, handletextpad=0.35,
                  columnspacing=1.6, bbox_to_anchor=(0.5, -0.005))
    figure.suptitle(f"Each painting to predict and its {args.neighbors} nearest "
                    "labelled neighbours", fontsize=13.5, fontweight="bold",
                    color=INK, y=0.985)

    figure.tight_layout(rect=(0, 0.075, 1, 0.965))
    figure.savefig(args.output, dpi=220, facecolor="white")
    print(f"saved: {args.output}")

    # A pureza continua sendo calculada e salva -- ela é resultado e vai para
    # outra figura/tabela, não para esta, que é ilustrativa.
    per_class = pd.DataFrame({"label": class_names[test_labels], "purity": purity})
    summary = per_class.groupby("label")["purity"].agg(["mean", "size"]).sort_values("mean")
    print("\npureza por estilo (5 menores e 5 maiores):")
    print(pd.concat([summary.head(5), summary.tail(5)]).round(3).to_string())
    per_class.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
