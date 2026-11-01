"""Troca a view `Text` do dump de predições: embedding de legenda -> TF-IDF/SVC.

O `Text` que estava no dump é `caption_embed_features.npy` -- o **embedding** da
legenda (384d), definido em `evaluate_standalone_by_classifier.py:127`. O §5.9
mede esse bloco como ortogonal a todo o banco visual (r² máximo 0,068 contra
qualquer das 639) e com marginal de **−0,09pp**; ele é o contraexemplo da frase
"ortogonal não é valioso, é frequentemente ruído".

O canal de texto que **funciona** é outro objeto: TF-IDF -> LinearSVC -> escores
por classe (125d), com marginal de **+2,31pp** (§5.18). Os dois compartilhavam o
rótulo `Text` na figura de concordância, o que faz o leitor concluir que texto é
ortogonal e inútil quando o inútil é o embedding.

**Escolha de estimador, que a legenda tem de declarar.** As outras views entram
como "MLP de produção sobre o bloco". O canal de texto entra com a predição do
próprio LinearSVC (argmax dos escores OOF), porque a view *é* um pipeline de
texto -- pôr um MLP sobre os 125 escores seria um segundo estágio, e os escores
já são OOF por construção (`build_text_scores.py`). O docstring do
`dump_view_predictions.py` fixa "um classificador só" para isolar a view do
estimador; aqui a exceção é declarada em vez de silenciosa.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=ROOT / "results/view_predictions_multitask.npz")
    ap.add_argument("--scores", type=Path, default=ROOT / "results/text_scores_student")
    ap.add_argument("--name", default="Text (TF-IDF)")
    ap.add_argument("--drop", default="Text", help="view a substituir")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    args.output = args.output or args.input.with_name(args.input.stem + "_svctext.npz")

    data = np.load(args.input, allow_pickle=True)
    stems = data["stems"].astype(str)
    decisions = np.load(args.scores / "multitask_decisions.npy")
    classes = np.load(args.scores / "multitask_classes.npy", allow_pickle=True).astype(str)
    score_stems = np.load(args.scores / "multitask_stems.npy", allow_pickle=True).astype(str)

    order = pd.Index(score_stems).get_indexer(stems)
    missing = int((order < 0).sum())
    if missing:
        raise ValueError(f"{missing:,} de {len(stems):,} stems da coorte sem escore de texto")

    meta = pd.read_csv(ROOT / "data/multitask_meta.csv", dtype={"stem": str})
    lookup = dict(zip(meta["stem"].astype(str), meta["style"].astype(str)))
    expected = np.array([lookup[s] for s in stems], dtype=object)

    labels = data["labels"].astype(int)
    label_to_name: dict[int, str] = {}
    for label in np.unique(labels):
        names = np.unique(expected[labels == label])
        if len(names) != 1:
            raise ValueError(
                f"rótulo local {label} corresponde a {len(names)} nomes: "
                f"{', '.join(map(str, names[:5]))}"
            )
        label_to_name[int(label)] = str(names[0])

    name_to_label = {name: label for label, name in label_to_name.items()}
    missing_classes = sorted(set(label_to_name.values()) - set(classes))
    if missing_classes:
        raise ValueError("classes locais sem coluna nos escores de texto: "
                         + ", ".join(missing_classes[:10]))

    truth_name = np.array([label_to_name[int(v)] for v in labels], dtype=object)
    agree = float(np.mean(truth_name == expected))
    if agree < 0.999:
        raise ValueError(f"mapa índice->nome não confere: {100 * agree:.1f}% de acordo. "
                         "O conjunto de classes do npz e o dos escores diferem.")
    print(f"mapa índice->nome verificado em {100 * agree:.1f}% da coorte")

    predicted_names = classes[decisions[order].argmax(axis=1)]
    unknown = sorted(set(predicted_names) - set(name_to_label))
    if unknown:
        raise ValueError(
            "as predições de texto incluem classes fora desta coorte; "
            "use um arquivo de escores treinado no mesmo conjunto de classes. "
            f"Exemplos: {', '.join(map(str, unknown[:10]))}"
        )
    prediction = np.array([name_to_label[str(name)] for name in predicted_names],
                          dtype=labels.dtype)

    payload = {k: data[k] for k in data.files if k != args.drop}
    payload[args.name] = prediction
    np.savez_compressed(args.output, **payload)

    old = data[args.drop].astype(labels.dtype) if args.drop in data.files else None
    print(f"coorte {len(stems):,} | {len(classes)} classes")
    if old is not None:
        print(f"  {args.drop:18s} acurácia {100 * (old == labels).mean():5.2f}%")
    print(f"  {args.name:18s} acurácia {100 * (prediction == labels).mean():5.2f}%")
    print(f"\nsalvo: {args.output}")


if __name__ == "__main__":
    main()
