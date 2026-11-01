"""O gap do oráculo é irrecuperável ou é supressão? Decomposição por classe.

O oráculo das 19 representações dá 74,22% de BAcc contra 52,9% do MLP de
produção. Esse gap de ~21pp costuma ser lido como "teto distante", mas ele
junta duas coisas de natureza oposta:

    irredutível   classes que **nenhuma** representação acerta bem. Nada a
                  fazer com fusão, só com representação nova.
    supressão     classes que **alguma** representação acerta bem e a fusão
                  não aproveita. Recuperável, em princípio, com arquitetura
                  ou ponderação melhor.

Só a segunda é acionável, e a proporção entre elas nunca foi medida. Este
script separa as duas, por classe, e nomeia o especialista de cada uma.

A pergunta que motivou (do usuário): *talvez uma modalidade seja boa em algo e
esteja sendo suprimida pelas outras*. A hipótese é testável exatamente assim --
se existir classe onde uma representação isolada bate a fusão, a fusão está
descartando informação que tem.

Contexto que torna isso plausível: o achado 16 mediu que desligar um branch na
inferência derruba Text em 6,46pp e Descriptors em só 1,98pp **apesar do maior
Shapley** -- ou seja, a fusão já demonstrou tratar branches de forma muito
desigual, e por razões que não são a contribuição média.

Saída: results/class_specialists.csv e um resumo agregado no log.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/pipeline"))
from registry import load_blocks, log  # noqa: E402

SEED, N_FOLDS = 42, 5
VISUAL = ("Descriptors", "SLIC", "Felzenszwalb", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier", "Multiscale")


def fused_predictions(dataset: str, labels_text: np.ndarray, max_k: int) -> np.ndarray:
    """Predições fora do fold da concatenação, mesma sonda das individuais."""
    cohort = load_blocks(dataset, VISUAL, allow_external=False)
    values = np.hstack([cohort.blocks[name] for name in cohort.blocks])
    labels = LabelEncoder().fit_transform(cohort.labels)
    n_classes = len(np.unique(labels))
    splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    predicted = np.empty(len(labels), dtype=np.int16)
    for train, test in splits.split(np.zeros(len(labels), dtype=np.uint8), labels):
        scaler = StandardScaler().fit(values[train])
        train_x, test_x = scaler.transform(values[train]), scaler.transform(values[test])
        k = min(max_k, train_x.shape[1], len(train) - 1)
        reducer = PCA(n_components=k, svd_solver="randomized",
                      random_state=SEED).fit(train_x)
        model = LinearDiscriminantAnalysis(
            priors=np.ones(n_classes) / n_classes).fit(
            reducer.transform(train_x), labels[train])
        predicted[test] = model.predict(reducer.transform(test_x))
    return predicted, labels, cohort.stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "results/channel_taxonomy_predictions.npz")
    parser.add_argument("--dataset", default="wikiart")
    parser.add_argument("--max-k", type=int, default=1024)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/class_specialists.csv")
    args = parser.parse_args()

    stored = np.load(args.predictions, allow_pickle=True)
    labels = stored["labels"]
    names = [key for key in stored.files if key != "labels"]
    classes = np.arange(len(np.unique(labels)))
    log(f"{len(names)} representações | {len(labels):,} imagens | "
        f"{len(classes)} classes")

    # Recall por classe de cada representação: quanto daquela classe ela pega.
    recall = pd.DataFrame(
        {name: [float((stored[name][labels == c] == c).mean()) for c in classes]
         for name in names}, index=classes)

    # Alinhamento por stem, nunca por posição: as duas coortes têm 79.207
    # imagens mas ordens diferentes, e casar por índice daria um resultado
    # errado em silêncio. Execuções antigas do npz não têm `stems`; nesse caso
    # a ordem é reconstruída chamando o mesmo `build_frame` da taxonomia.
    if "stems" in stored.files:
        stored_stems = stored["stems"].astype(str)
    else:
        log("npz sem stems (execução antiga); reconstruindo a ordem via build_frame")
        sys.path.insert(0, str(ROOT / "experiments"))
        from evaluate_channel_taxonomy import build_frame
        stored_stems = build_frame()[0]["stem"].to_numpy(dtype=str)
        if len(stored_stems) != len(labels):
            log(f"! build_frame deu {len(stored_stems):,} contra {len(labels):,} "
                "nas predições — abortando")
            return

    log("calculando a referência de fusão (LDA sobre a concatenação)...")
    fused, fused_labels, fused_stems = fused_predictions(args.dataset, labels, args.max_k)
    position = {stem: index for index, stem in enumerate(fused_stems)}
    missing = [stem for stem in stored_stems if stem not in position]
    if missing:
        log(f"! {len(missing):,} stems das predições não estão na coorte da fusão "
            "— abortando")
        return
    order = np.array([position[stem] for stem in stored_stems])
    fused, fused_labels = fused[order], fused_labels[order]
    assert np.array_equal(fused_labels, labels), "rótulos divergem após alinhar por stem"
    fused_recall = np.array([float((fused[labels == c] == c).mean()) for c in classes])

    # Oráculo por classe: alguma representação acerta.
    hits = np.stack([stored[name] == labels for name in names])
    oracle_recall = np.array(
        [float(hits[:, labels == c].any(axis=0).mean()) for c in classes])

    best_value = recall.max(axis=1).to_numpy()
    best_owner = recall.idxmax(axis=1).to_numpy()
    table = pd.DataFrame({
        "class": classes,
        "support": [int((labels == c).sum()) for c in classes],
        "fused": 100 * fused_recall,
        "best_single": 100 * best_value,
        "specialist": best_owner,
        "oracle": 100 * oracle_recall,
        # Positivo = alguma representação sozinha bate a fusão naquela classe.
        "suppression": 100 * (best_value - fused_recall),
        # O que nem o melhor especialista alcança: precisa de representação nova.
        "irreducible": 100 * (oracle_recall - best_value)})
    table = table.sort_values("suppression", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.4f")

    log("\n=== por classe (ordenado por supressão) ===")
    log(table.round(2).to_string(index=False))

    # Em acurácia balanceada cada classe pesa 1/K, então médias simples das
    # colunas já são a decomposição do gap em pontos percentuais.
    log("\n=== decomposição do gap do oráculo (BAcc, pontos) ===")
    log(f"  fusão                      {table['fused'].mean():5.2f}")
    log(f"  melhor especialista/classe {table['best_single'].mean():5.2f}"
        f"   (+{table['suppression'].mean():.2f} = SUPRESSÃO, recuperável)")
    log(f"  oráculo                    {table['oracle'].mean():5.2f}"
        f"   (+{table['irreducible'].mean():.2f} = irredutível por seleção)")
    beaten = table[table["suppression"] > 0]
    log(f"\n{len(beaten)}/{len(table)} classes têm uma representação isolada que "
        f"bate a fusão")
    if len(beaten):
        log("especialistas mais frequentes: "
            + ", ".join(f"{name} ({count})" for name, count
                        in beaten["specialist"].value_counts().head(5).items()))
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
