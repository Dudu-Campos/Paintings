"""Taxonomia de canais a partir do padrão de erro, sem hipótese sobre canais.

A literatura de classificação de pintura acumulou dezenas de descritores, cada
um com sua justificativa própria de por que funciona -- passeio do turista para
textura, HOG para forma, LBP, Haralick, dimensão fractal. A pergunta é se essa
proliferação de *descritores* corresponde a uma proliferação de *informação*.

Duas maneiras erradas de responder, ambas já descartadas por medição neste
projeto:

  1. **Redundância entre features.** Não prevê contribuição e aqui aponta ao
     contrário: Hilbert tem R² 0,919 contra o resto e vale +3,76pp; o passeio
     do turista tem R² 0,161 -- o menos redundante de todos -- e vale zero.
     Redundância mora no espaço de variância, contribuição no discriminante.
  2. **Derrubar o canal suspeito e ver se o proxy revive.** Exige *escolher* o
     canal, e a escolha é justamente o que está em disputa: é confirmação de
     hipótese com a resposta embutida na pergunta.

O que este script faz é a terceira via, e ela não infere nada: mede o padrão de
**complementaridade na predição** e deixa a taxonomia emergir. Precedente
dentro do próprio projeto -- a fatorial revelou que Texto era diferente sem
nenhuma tese a respeito, e o desligamento de branch na inferência (achado 16)
mediu insubstituibilidade e produziu conclusão taxonômica que ninguém tinha
formulado.

Custo O(N), não O(2^N): uma sonda por representação, predições fora do fold, e
toda a estrutura par-a-par sai dessas predições sem re-treinar nada.

**A armadilha, e a correção.** Semelhança de predição é contaminada por
dificuldade e prior: duas representações inúteis que caem na classe majoritária
parecem idênticas, e duas fortes que acertam as imagens fáceis também. Sem
correção o dendrograma agrupa por acurácia, não por canal. Por isso o que entra
na distância é o **excesso de falta dupla sobre o esperado sob independência**,
dadas as acurácias marginais de cada uma -- a família de medidas de diversidade
de Kuncheva & Whitaker (2003), que tem a vantagem lateral de já ser vocabulário
estabelecido em Pattern Recognition.

Duas quantidades além do agrupamento, porque correlação é simétrica e a relação
de proxy não é:

    subsunção   P(A acerta | B erra) -- se A não acerta nada onde B erra, A
                não tem nada que B já não tenha
    perfil      acurácia por classe, para inspecionar o canal depois de
                descoberto (a interpretação é opcional; a taxonomia não
                depende dela)

**Validação sem rodar nada novo.** A taxonomia prevê que o marginal de
acrescentar A depende de o canal de A já estar ocupado. Cinco marginais já
medidos servem de teste: L(G_felz) +0,79 e Quickshift +0,67 (canal novo), HOG
−1,21, turista −0,02 e triângulos +0,06 (canal ocupado). O script reporta a
distância de cada representação ao canônico mais próximo, que é a quantidade
que deveria ordenar esses cinco.

Saída: results/channel_taxonomy_predictions.npz, results/channel_taxonomy.csv
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
SEED, N_FOLDS = 42, 5

# Representações fora do carregador canônico. Formato: (nome, tipo, path, stems)
EXTRA = (
    ("Random walk",        "npy",     "outputs/wikiart_features/rw_features.npy",
                                      "outputs/wikiart_features/rw_stems.npy"),
    ("Hybrid walk",        "npy",     "outputs/wikiart_features/hybrid_walk_features.npy",
                                      "outputs/wikiart_features/hybrid_walk_stems.npy"),
    ("Square graph",       "parquet", "results/square_graph_full_features.parquet", None),
    ("Hilbert catch22",    "npy",     "outputs/hilbert_catch22_full/features.npy",
                                      "outputs/hilbert_catch22_full/stems.npy"),
    ("Hilbert spectrum1d", "npy",     "outputs/hilbert_spectrum_1d_full/features.npy",
                                      "outputs/hilbert_spectrum_1d_full/stems.npy"),
    ("Hilbert visibility", "npy",     "outputs/hilbert_visibility_graph_full/features.npy",
                                      "outputs/hilbert_visibility_graph_full/stems.npy"),
    ("Hilbert wavelet",    "npy",     "outputs/hilbert_wavelet_pyramid_full/features.npy",
                                      "outputs/hilbert_wavelet_pyramid_full/stems.npy"),
    ("Tourist walk",       "npy",     "outputs/tourist_walk_backes/features.npy",
                                      "outputs/tourist_walk_backes/stems.npy"),
    ("Triangles SLIC",     "parquet", "outputs/triangle_features_slic/triangle_features.parquet", None),
    ("Triangles Felz",     "parquet", "outputs/triangle_features_felz/triangle_features.parquet", None),
    # Os dois moram no mesmo .npy: colunas 0-12 são Haralick, 13+ são HOG.
    # Convenção herdada de evaluate_lda_pca_curve_by_subset_hog_balanced.py:75.
    # Entram separados porque são descritores clássicos *distintos*, com
    # justificativas publicadas diferentes -- Haralick é co-ocorrência de nível
    # de cinza, HOG é gradiente orientado -- e a pergunta é justamente se caem
    # no mesmo canal apesar disso.
    ("Haralick",           "slice",   "outputs/haralick_hog_features/features.npy",
                                      "outputs/haralick_hog_features/stems.npy:0:13"),
    ("HOG",                "slice",   "outputs/haralick_hog_features/features.npy",
                                      "outputs/haralick_hog_features/stems.npy:13:"),
)

# As oito do pipeline canônico servem de referência: a distância ao canônico
# mais próximo é o que deve ordenar os marginais já medidos.
CANONICAL = ("SLIC", "Descriptors", "Hilbert", "Line graph", "Felzenszwalb",
             "Fourier", "Text", "L(G_felz)")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_frame():
    """Coorte canônica com todos os blocos disponíveis, e o mapa nome -> colunas."""
    from evaluate_felzenszwalb_line_graph import load_felz_line
    from evaluate_full_factorial_ablation_text import load_text_block
    from evaluate_multiscale_graph_fourier import load_fourier
    from evaluate_multiscale_graph_fusion import load_graph_blocks
    from final_tfidf_visual_mlp import load_dataset

    frame, groups = load_dataset(ROOT / "outputs/wikiart_quadrant_features")
    frame, line_columns, felz_columns = load_graph_blocks(
        frame, ROOT / "results/line_graph_full_features.parquet",
        ROOT / "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet")
    frame, fourier_columns = load_fourier(
        frame, ROOT / "results/fourier_full_features.parquet")
    frame, felzline_columns = load_felz_line(
        frame, ROOT / "results/line_graph_felz_full_features.parquet")

    blocks = {
        "SLIC":         list(groups["SLIC"]),
        "Descriptors":  list(groups["Descriptors"]),
        "Hilbert":      list(groups["Hilbert"]),
        "Line graph":   list(line_columns),
        "Felzenszwalb": list(felz_columns),
        "Fourier":      list(fourier_columns),
        "L(G_felz)":    list(felzline_columns),
    }
    try:
        frame, text_columns = load_text_block(frame)
        blocks["Text"] = list(text_columns)
    except Exception as error:                       # cache de TF-IDF ausente
        log(f"  ! Text indisponível ({error}); seguindo sem ele")

    for name, kind, path, stems_path in EXTRA:
        full = ROOT / path
        if not full.exists():
            log(f"  ! {name}: {path} não existe, pulando")
            continue
        prefix = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if kind == "slice":
            stems_file, begin, end = stems_path.rsplit(":", 2)
            values = np.asarray(np.load(full, mmap_mode="r"))
            values = values[:, int(begin):(int(end) if end else None)]
            stems = np.load(ROOT / stems_file, allow_pickle=True).astype(str)
            columns = [f"{prefix}_{i}" for i in range(values.shape[1])]
            block = pd.DataFrame(values, columns=columns)
            block.insert(0, "stem", stems)
        elif kind == "npy":
            values = np.asarray(np.load(full, mmap_mode="r"))
            stems = np.load(ROOT / stems_path, allow_pickle=True).astype(str)
            columns = [f"{prefix}_{i}" for i in range(values.shape[1])]
            block = pd.DataFrame(values, columns=columns)
            block.insert(0, "stem", stems)
        else:
            raw = pd.read_parquet(full)
            feature_columns = [c for c in raw if c not in {"stem", "style", "label"}]
            rename = {c: f"{prefix}_{c}" for c in feature_columns}
            block = raw[["stem"] + feature_columns].rename(columns=rename)
            columns = list(rename.values())
        block = block.drop_duplicates("stem")
        before = len(frame)
        frame = frame.merge(block, on="stem", how="inner", validate="one_to_one")
        if len(frame) != before:
            log(f"  {name}: coorte {before:,} -> {len(frame):,}")
        blocks[name] = columns
    return frame, blocks


def oof_predictions(values, labels, splits, n_classes, max_k):
    """Predições fora do fold de uma sonda LDA sobre PCA."""
    predicted = np.empty(len(labels), dtype=np.int16)
    for train, test in splits:
        scaler = StandardScaler().fit(values[train])
        train_x = scaler.transform(values[train])
        test_x = scaler.transform(values[test])
        k = min(max_k, train_x.shape[1], len(train) - 1)
        if k < train_x.shape[1]:
            pca = PCA(n_components=k, random_state=SEED).fit(train_x)
            train_x, test_x = pca.transform(train_x), pca.transform(test_x)
        model = LinearDiscriminantAnalysis(
            priors=np.ones(n_classes) / n_classes)
        model.fit(train_x, labels[train])
        predicted[test] = model.predict(test_x)
    return predicted


def excess_double_fault(correct_a, correct_b):
    """Falta dupla observada menos a esperada sob independência.

    Zero = os erros são independentes dadas as acurácias marginais, ou seja as
    duas representações acessam informação não relacionada. Positivo = erram
    juntas mais do que o acaso explica, a assinatura de canal compartilhado.
    Negativo = complementares além do independente.

    É a correção que impede o agrupamento de seguir acurácia em vez de canal:
    duas representações fracas erram muito as duas, mas o *esperado* também é
    alto, e o excesso desconta isso.
    """
    wrong_a, wrong_b = ~correct_a, ~correct_b
    observed = float((wrong_a & wrong_b).mean())
    expected = float(wrong_a.mean() * wrong_b.mean())
    return observed - expected


def oracle_curve(predictions, labels, order, n_classes):
    """Taxa de oráculo acumulada, com as duas correções que a tornam legível.

    Oráculo = a amostra conta como certa se **alguma** representação a acerta.
    Em acurácia balanceada, calculado por classe e depois promediado, senão a
    classe majoritária domina como sempre.

    Duas referências obrigatórias, porque o número cru é quase todo artefato:

      acaso        K sondas aleatórias sobre C classes já cobrem
                   1 − (1 − 1/C)^K. Com C=27 e K=19 isso é 51,2% -- ou seja,
                   o valor do nosso MLP. Sem essa linha, qualquer oráculo
                   parece impressionante.
      independência  1 − Π(1 − recall_i(c)) por classe, usando o recall real de
                   cada sonda. É o que se obteria se os erros fossem
                   descorrelacionados dadas as acurácias marginais.

    O **déficit contra a independência** é a quantidade que interessa: é a
    sobreposição de canais, agregada, e a mesma coisa que a matriz de falta
    dupla mede par-a-par.

    Ressalva de interpretação: oráculo é teto de **seleção** por amostra, não
    de fusão. Um classificador treinado na união das features não está
    formalmente limitado por ele.
    """
    classes = np.arange(n_classes)
    hit = {name: (predictions[name] == labels) for name in order}
    rows, covered = [], np.zeros(len(labels), dtype=bool)
    for step, name in enumerate(order, start=1):
        covered = covered | hit[name]
        per_class = [covered[labels == c].mean() for c in classes]
        # Independência: por classe, 1 − Π(1 − recall) das sondas já incluídas.
        independent = []
        for c in classes:
            mask = labels == c
            product = 1.0
            for other in order[:step]:
                product *= 1.0 - float(hit[other][mask].mean())
            independent.append(1.0 - product)
        rows.append({
            "k": step, "added": name,
            "oracle": 100 * float(np.mean(per_class)),
            "independent": 100 * float(np.mean(independent)),
            "chance": 100 * (1 - (1 - 1 / n_classes) ** step)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=256,
                        help="teto de componentes de PCA por representação. "
                             "Existe para que blocos de 1.032d e de 57d sejam "
                             "sondados com capacidade comparável -- sem isso a "
                             "taxonomia mistura canal com tamanho.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/channel_taxonomy.csv")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "results/channel_taxonomy_predictions.npz")
    args = parser.parse_args()

    frame, blocks = build_frame()
    # `load_dataset` nomeia a coluna de alvo como `label`, não `style`.
    labels = LabelEncoder().fit_transform(frame["label"].astype(str).to_numpy())
    n_classes = len(np.unique(labels))
    log(f"coorte {len(frame):,} | {len(blocks)} representações | {n_classes} classes")

    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(np.zeros(len(labels), dtype=np.uint8), labels))

    predictions, scores = {}, {}
    for name, columns in blocks.items():
        started = time.perf_counter()
        values = frame[columns].to_numpy(dtype=np.float32)
        values = np.where(np.isfinite(values), values, 0.0)
        predicted = oof_predictions(values, labels, splits, n_classes, args.max_k)
        predictions[name] = predicted
        scores[name] = balanced_accuracy_score(labels, predicted)
        log(f"  {name:<20} {len(columns):>5}d  BAcc={100 * scores[name]:5.2f}%  "
            f"({time.perf_counter() - started:.0f}s)")

    names = list(predictions)
    correct = {name: predictions[name] == labels for name in names}
    # Os stems entram junto: sem eles não há como alinhar estas predições com
    # nenhuma outra coorte a não ser por posição, o que é frágil e silencioso
    # quando a ordem difere.
    np.savez_compressed(args.predictions, labels=labels,
                        stems=frame["stem"].to_numpy(dtype=str),
                        **{name: predictions[name] for name in names})

    distance = pd.DataFrame(
        [[excess_double_fault(correct[a], correct[b]) for b in names] for a in names],
        index=names, columns=names)
    subsumption = pd.DataFrame(
        [[float(correct[a][~correct[b]].mean()) if (~correct[b]).any() else np.nan
          for b in names] for a in names], index=names, columns=names)

    log("\n=== excesso de falta dupla (x100; alto = mesmo canal) ===")
    log((distance * 100).round(2).to_string())
    log("\n=== P(linha acerta | coluna erra) (x100; baixo = linha é subsumida) ===")
    log((subsumption * 100).round(1).to_string())

    reference = [name for name in CANONICAL if name in distance.columns]
    rows = []
    for name in names:
        others = [r for r in reference if r != name]
        nearest = distance.loc[name, others].idxmax()
        rows.append({"representation": name, "dims": len(blocks[name]),
                     "bacc": 100 * scores[name],
                     "nearest_canonical": nearest,
                     "excess_double_fault": 100 * distance.loc[name, nearest],
                     "subsumed_by_nearest": 100 * subsumption.loc[name, nearest]})
    summary = pd.DataFrame(rows).sort_values("excess_double_fault", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, float_format="%.4f")
    distance.to_csv(args.output.with_name("channel_taxonomy_distance.csv"),
                    float_format="%.6f")

    log("\n=== proximidade ao canônico mais próximo ===")
    log("(a validação: os de marginal zero -- turista, triângulos -- devem "
        "aparecer no topo; L(G_felz), que valeu +0,79pp, no fundo)")
    log(summary.round(2).to_string(index=False))

    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        # Agrupamento sobre distância = −excesso, deslocada para ser não-negativa
        # e simetrizada; a diagonal zera por construção.
        matrix = -(distance.to_numpy() + distance.to_numpy().T) / 2
        matrix -= matrix.min()
        np.fill_diagonal(matrix, 0.0)
        tree = linkage(squareform(matrix, checks=False), method="average")
        order = dendrogram(tree, labels=names, no_plot=True)["ivl"]
        log("\n=== ordem do agrupamento hierárquico (canais emergentes) ===")
        log(" -> ".join(order))
        np.save(args.output.with_name("channel_taxonomy_linkage.npy"), tree)
    except Exception as error:
        log(f"agrupamento falhou: {error}")

    # Ordem de entrada no oráculo: acurácia isolada decrescente. É arbitrária
    # mas pré-especificada, e o valor final (todas incluídas) não depende dela.
    order = sorted(names, key=lambda n: -scores[n])
    oracle = oracle_curve(predictions, labels, order, n_classes)
    oracle.to_csv(args.output.with_name("channel_taxonomy_oracle.csv"),
                  index=False, float_format="%.4f")
    log("\n=== oráculo acumulado (BAcc, %) ===")
    log("observado | sob independência | de puro acaso")
    log(oracle.round(2).to_string(index=False))
    final = oracle.iloc[-1]
    log(f"\ndéficit contra independência: "
        f"{final['independent'] - final['oracle']:.1f}pp  "
        f"(= sobreposição de canais); acima do acaso: "
        f"{final['oracle'] - final['chance']:.1f}pp")

    log(f"\nSalvo: {args.output} e {args.predictions}")


if __name__ == "__main__":
    main()
