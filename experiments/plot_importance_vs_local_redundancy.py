"""Rich feature importance/redundancy scatter with Text AND the global
similarity graph as extra groups (extends
plot_rich_importance_redundancy_top10_with_text_en.py)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

# `analyze_rich_lda_structure_importance` continua no attic/ (§7bis). O import e
# PREGUICOSO, dentro do ramo que recomputa: o `--from-csv` nao precisa dele, e um
# import de modulo quebraria o re-plot por causa de uma dependencia que ele nao
# usa. O attic entra no path para o ramo de calculo seguir funcionando.
sys.path.insert(0, str(ROOT / "attic" / "experiments"))
from analyze_visual_feature_importance_lda_xgb import make_lda  # noqa: E402
from fusion_curve_slic import N_FOLDS, SEED  # noqa: E402
from plot_rich_importance_redundancy_top10_with_text_en import (  # noqa: E402
    impute_and_standardize, load_text_scores, redundancy_metrics,
)


COLOURS = {
    "Region graphs": "#eb6834",
    "Image statistics": "#2a78d6",
    "Space-filling curve": "#1baf7a",
    "Spectral": "#4a3aa7",
    "Global similarity graph": "#cdae31",
    "Text": "#e94149",
}
MARKERS = {
    "Region graphs": "o",
    "Image statistics": "s",
    "Space-filling curve": "^",
    "Spectral": "D",
    "Global similarity graph": "P",
    "Text": "*",
}
GROUP_ORDER = tuple(COLOURS)
TEXT = "#202124"
MUTED = "#65676b"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_global_graph_scores(cache_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(cache_path)
    if frame["stem"].duplicated().any():
        raise ValueError("Duplicate stems in global graph cache")
    return frame


def plot_scatter(data: pd.DataFrame, output: Path, alpha: float, size: float,
                 legend: str = "right", font_scale: float = 1.0) -> None:
    # Um fator unico sobre os quatro tamanhos preserva a HIERARQUIA (titulo >
    # rotulo > tick ~ legenda); subir cada um na mao desregula a proporcao.
    f_titulo, f_rotulo = 16 * font_scale, 13 * font_scale
    f_tick, f_legenda = 11.5 * font_scale, 10 * font_scale
    y_cut = float(data["importance_pp"].quantile(0.75))
    y_max = float(data["importance_pp"].quantile(0.997) * 1.15)
    y_max = max(y_max, float(data["importance_pp"].max() * 1.08))
    x_cut = float(data["top10_mean_r2"].quantile(0.75))
    x_max = float(data["top10_mean_r2"].quantile(0.995) * 1.10)
    x_max = min(1.0, max(x_max, float(data["top10_mean_r2"].max() * 1.04)))

    # Com a chave FORA do eixo a figura alarga, senao o painel encolheria para
    # abrir espaco -- a area de plotagem tem de ficar igual ou maior, nao menor.
    # A chave alarga junto com a fonte dela, senao ela invade o painel.
    largura = (9.6 + 1.6 * (font_scale - 1.0)) if legend == "right" else 7.6
    figure, axis = plt.subplots(figsize=(largura, 5.9))
    axis.set_facecolor("#fbfbfa")
    axis.grid(color="#dedede", linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    handles = []
    highlight = {"Text", "Global similarity graph"}
    size_multiplier = {"Text": 3.6, "Global similarity graph": 2.4}
    for group in GROUP_ORDER:
        rows = data[data["group"].eq(group)]
        if rows.empty:
            continue
        colour = COLOURS[group]
        marker = MARKERS[group]
        axis.scatter(
            rows["top10_mean_r2"], rows["importance_pp"],
            s=size * size_multiplier.get(group, 1.0),
            color=colour, marker=marker, alpha=0.78 if group in highlight else alpha,
            linewidth=0, zorder=4 if group in highlight else 3,
        )
        handles.append(plt.Line2D([0], [0], marker=marker, linestyle="", color=colour,
                                  label=f"{group}", markersize=7 * font_scale))

    axis.axvline(x_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
    axis.axhline(y_cut, color="#8a8f98", linestyle=(0, (4, 3)), linewidth=1.0)
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, y_max)
    axis.set_xlabel("Mean r² of the 10 nearest neighbors", fontsize=f_rotulo)
    axis.set_ylabel("Normalized supervised importance (%)", fontsize=f_rotulo)
    axis.set_title("Feature importance vs. local redundancy", fontsize=f_titulo,
                   fontweight="bold", color=TEXT)
    axis.tick_params(axis="both", labelsize=f_tick)
    # A chave sai do eixo pela DIREITA (decisao do usuario, 07/out). No canto
    # superior esquerdo ela cobria as estrelas do `Text`, que e justamente o
    # grupo cujo argumento a figura carrega -- redundancia local BAIXA e
    # importancia ALTA. A direita de fora nao cobre ponto nenhum e a area de
    # plotagem fica inteira.
    if legend == "right":
        axis.legend(handles=handles, frameon=False, fontsize=f_legenda,
                    loc="upper left", bbox_to_anchor=(1.015, 1.0),
                    borderaxespad=0.0, labelspacing=0.9, handletextpad=0.6)
        figure.tight_layout(rect=(0, 0.015, 1, 0.97))
    else:
        axis.legend(handles=handles, frameon=False, fontsize=f_legenda, loc=legend)
        figure.tight_layout(rect=(0, 0.015, 1, 0.97))
    figure.savefig(output, dpi=230, facecolor="white")
    figure.savefig(output.with_name(output.stem + "_paper.png"), dpi=230, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/artemis/wikiart_meta.csv")
    parser.add_argument("--rich-dir", type=Path, default=ROOT / "outputs/wikiart_rich_features")
    parser.add_argument("--rich-fourier", type=Path, default=ROOT / "results/fourier_rich_features.parquet")
    parser.add_argument("--slic", type=Path, default=ROOT / "outputs/wikiart_slic_features/combined_features.parquet")
    parser.add_argument("--felzenszwalb", type=Path, default=ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    parser.add_argument("--line-graph", type=Path, default=ROOT / "results/line_graph_full_features.parquet")
    parser.add_argument("--line-graph-felz", type=Path, default=ROOT / "results/line_graph_felz_full_features.parquet")
    parser.add_argument("--text-stems", type=Path, default=ROOT / "results/tfidf_caption_style_stems.npy")
    parser.add_argument("--text-decisions", type=Path, default=ROOT / "results/tfidf_caption_style_oof_decisions.npy")
    parser.add_argument("--text-classes", type=Path, default=ROOT / "results/tfidf_caption_style_classes.npy")
    parser.add_argument("--global-cache", type=Path,
                        default=ROOT / "results/rich_feature_redundancy_r2_with_global_graph/global_similarity_graph_extended_features.parquet")
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="re-plota do CSV ja gravado, sem recomputar nada. "
                             "As entradas do WikiArt continuam em disco, mas o "
                             "calculo inteiro so para mudar layout nao paga.")
    parser.add_argument("--font-scale", type=float, default=1.3,
                        help="multiplica titulo, rotulos, ticks e legenda de uma "
                             "vez, preservando a hierarquia entre eles")
    parser.add_argument("--legend", default="right",
                        choices=["right", "upper left", "upper right", "lower right"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/importance_vs_local_redundancy.png")
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--size", type=float, default=14.0)
    args = parser.parse_args()

    # ATALHO DE RE-PLOT. O calculo inteiro (LDA por fold sobre 1.755 features +
    # a matriz de r2 par a par) existe so para produzir o CSV; mudar layout nao
    # precisa dele. O CSV gravado carrega as tres colunas que o plot usa
    # (`group`, `top10_mean_r2`, `importance_pp`).
    if args.from_csv is not None:
        data = pd.read_csv(args.from_csv)
        faltam = {"group", "top10_mean_r2", "importance_pp"} - set(data.columns)
        if faltam:
            raise SystemExit(f"{args.from_csv.name}: faltam colunas {sorted(faltam)}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plot_scatter(data, args.output, args.alpha, args.size, args.legend,
                     args.font_scale)
        log(f"re-plotado de {args.from_csv.name}: {args.output}")
        return


    # Import PREGUICOSO: so o ramo que recomputa precisa dele, e ele mora no
    # attic/ (§7bis). Deixa-lo no topo quebrava o `--from-csv`.
    from analyze_rich_lda_structure_importance import (  # noqa: E402
        load_rich_frame, structure_importance)

    frame, visual_features, visual_groups = load_rich_frame(args)
    text = load_text_scores(args.text_stems, args.text_decisions, args.text_classes)
    global_graph = load_global_graph_scores(args.global_cache)

    before = len(frame)
    frame = frame.merge(text, on="stem", how="inner", validate="one_to_one")
    frame = frame.merge(global_graph, on="stem", how="inner", validate="one_to_one")
    if len(frame) != before:
        log(f"merge changed cohort {before:,} -> {len(frame):,}")

    text_features = [c for c in text.columns if c != "stem"]
    graph_features = [c for c in global_graph.columns if c != "stem"]
    feature_names = np.asarray([*visual_features, *text_features, *graph_features], dtype=object)
    groups = np.asarray([*visual_groups.tolist(),
                        *["Text"] * len(text_features),
                        *["Global similarity graph"] * len(graph_features)], dtype=object)
    values = frame[feature_names].to_numpy(dtype=np.float32)

    standardized, keep = impute_and_standardize(values)
    feature_names, groups, values = feature_names[keep], groups[keep], values[:, keep]
    log(f"{len(frame):,} images x {values.shape[1]:,} features including Text + global graph")

    labels = LabelEncoder().fit_transform(frame["style"].to_numpy(dtype=str))
    splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels)
    rows = []
    for fold, (train, _test) in enumerate(splits, 1):
        model = make_lda(len(np.unique(labels)), values.shape[1])
        model.fit(values[train], labels[train])
        importance = structure_importance(model, values[train])
        normalized = importance / importance.sum()
        rows.extend({"fold": fold, "feature": feature, "importance": float(importance[i]),
                    "importance_normalized": float(normalized[i])}
                   for i, feature in enumerate(feature_names))
        log(f"fold={fold}/{N_FOLDS} importance complete")

    importance = (pd.DataFrame(rows).groupby("feature", as_index=False).agg(
        importance_mean=("importance", "mean"), importance_std=("importance", "std"),
        importance_normalized_mean=("importance_normalized", "mean"),
        importance_normalized_std=("importance_normalized", "std")))
    metrics, keep_for_metrics = redundancy_metrics(values)
    if not np.all(keep_for_metrics):
        raise RuntimeError("Unexpected constants after initial filtering")
    data = pd.DataFrame({"feature": feature_names, "group": groups}).join(metrics)
    data = data.merge(importance, on="feature", how="inner", validate="one_to_one")
    data["importance_pp"] = 100 * data["importance_normalized_mean"]
    data["group"] = pd.Categorical(data["group"], GROUP_ORDER, ordered=True)
    data = data.sort_values(["group", "importance_pp"], ascending=[True, False]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output.with_suffix(".csv"), index=False, float_format="%.8f")
    plot_scatter(data, args.output, args.alpha, args.size, args.legend,
                     args.font_scale)
    log(f"saved: {args.output}")
    log("top Global similarity graph features:\n" +
        data[data["group"].astype(str).eq("Global similarity graph")]
        .sort_values("importance_pp", ascending=False)[["feature", "importance_pp", "top10_mean_r2"]]
        .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
