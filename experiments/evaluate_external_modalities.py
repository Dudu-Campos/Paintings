"""Modalidades hand-designed nos datasets externos, sonda LDA.

Mesmo instrumento do WikiArt (LDA com priors uniformes, 5 folds
estratificados, `SEED=42`) aplicado a Pandora e Rijksmuseum, para responder
duas perguntas que a tabela final precisa:

  1. cada modalidade sozinha vale quanto? -- dá a ordem de importância e
     permite ver se ela **replica** a do WikiArt (Descriptors no topo)
  2. juntas valem quanto contra a ResNet50 congelada? -- é o confronto que
     interessa hoje, já que a CNN treinada do zero foi aposentada como
     baseline (em 18 mil imagens ela seria artificialmente fraca)

Os dois datasets testam coisas diferentes de propósito: Pandora é a mesma
tarefa (estilo) mas **balanceado**, o que testa se os ganhos sobrevivem sem
as correções de desbalanceamento que o pipeline carrega; Rijksmuseum é tarefa
**diferente** (período de meio-século), o que testa se a complementaridade é
propriedade das representações ou do rótulo do WikiArt.

Blocos ausentes são pulados com aviso -- a extração é incremental.

Saída: results/external_modalities.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "PRISM"))
SEED, N_FOLDS = 42, 5
LABELS = {"pandora": ("data/pandora_meta.csv", "style"),
          "rijksmuseum": ("data/rijksmuseum_labels.csv", "period"),
          # Desenhos do Rijksmuseum: mesmo museu e mesma tarefa de meio-século
          # das pinturas, com ~9x os dados (38,6k de treino contra 3,9k). É o
          # teste da previsão registrada em 15/ago -- o padrão observado nos
          # três datasets é **dado suficiente para a fusão funcionar**, não
          # distância da tarefa ao ImageNet. Aqui só o `n` muda.
          "rijksdraw": ("data/rijksdraw_labels.csv", "period"),
          "multitask": ("data/multitask_meta.csv", "style"),
          # Chu & Wu, ACM MM 2016. 17 estilos que sao subconjunto quase
          # exato dos 25 do benchmark do Multitask, o que permite
          # comparacao controlada entre os dois corpora.
          "oilpainting": ("data/oilpainting_meta.csv", "style"),
          # Fora de arte, para testar generalizacao. Food-101 e balanceado
          # (1.000/classe) e de granularidade fina; Places-LT e o oposto
          # exato -- 996x de desbalanceamento, que estressa a correcao de
          # prior do pipeline num regime 7x pior que o WikiArt.
          "food101": ("data/food101_meta.csv", "style"),
          "places_lt": ("data/places_lt_meta.csv", "style"),
          "nct_crc": ("data/nct_crc_meta.csv", "style")}
# nome -> (caminho relativo, formato). Ordem = ordem de Shapley do WikiArt.
BLOCKS = {
    "Descriptors":  ("outputs/{ds}_descriptors/desc_full_features.npy", "npy"),
    "Felzenszwalb": ("outputs/{ds}_felz_features/combined_features.parquet", "parquet"),
    "SLIC":         ("outputs/{ds}_slic_features/combined_features.parquet", "parquet"),
    "Line graph":   ("results/line_graph_{ds}_features.parquet", "parquet"),
    "L(G_felz)":    ("results/line_graph_felz_{ds}_features.parquet", "parquet"),
    "Hilbert":      ("outputs/{ds}_hilbert/hilbert_features.npy", "npy"),
    "Fourier":      ("results/fourier_{ds}_features.parquet", "parquet"),
    "Gabor":        ("outputs/{ds}_gabor/gabor_features.npy", "npy"),
    "Multiscale":   ("outputs/{ds}_quadrant_features", "special"),
    "ResNet50":     ("outputs/{ds}_cnn_frozen/cnn_full_features.npy", "npy"),
}
# Blocos que existem para consulta por `load_block` mas **não** fazem parte do
# banco iterado. Oito scripts fazem `for name in BLOCKS`, e pôr coisas novas lá
# muda em silêncio o que todos eles medem -- em 27/ago isso fez a grade
# representação x dados subir de 9 para 10 degraus sem que nada avisasse.
EXTRA_BLOCKS = {
    # ViT-B/16 congelado, token CLS de 768d, pesos IMAGENET1K_V1: mesmo orçamento
    # de supervisão da ResNet50 congelada, família de arquitetura diferente.
    "ViT-B/16": ("outputs/{ds}_vit_frozen/cnn_full_features.npy", "npy"),
    # Embedding MiniLM (384d) das legendas CATR/COCO. Nenhum rótulo do corpus
    # entra nele, mas é supervisão EXTERNA: o captioner viu ~600k legendas do COCO.
    "Caption embed": ("outputs/{ds}_caption_embeddings/caption_embed_features.npy", "npy"),
}

NPY_STEMS = {"Descriptors": "desc_full_stems.npy", "Hilbert": "hilbert_stems.npy",
             "Gabor": "gabor_stems.npy", "ResNet50": "cnn_full_stems.npy",
             "ViT-B/16": "cnn_full_stems.npy",
             "Caption embed": "caption_embed_stems.npy"}
METADATA = {"stem", "style", "label", "period", "filename", "sample_id",
            "elapsed_seconds_all_graphs"}
# Os dois geradores de multigrafo nomeiam as colunas de parâmetro de forma
# diferente (`slic_*` num, `segmentation_*` no outro). As 71 features em si
# são idênticas -- conferido por diferença simétrica dos conjuntos de colunas.
# Deixar passar poria o externo com 74 colunas contra 71 do WikiArt, e uma
# delas (`segmentation_actual_segments`) é informativa: entraria como feature
# extra acidental e as linhas da tabela deixariam de ser comparáveis.
PARAMETER_PREFIXES = ("slic_", "segmentation")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


QUADRANTS = ("top_left", "top_right", "bottom_left", "bottom_right")


def load_multiscale(dataset: str) -> pd.DataFrame | None:
    """Multiscale = SLIC + Descriptors + Hilbert nos 4 quadrantes (1.032d).

    Mesma definição da grade do WikiArt: **três** famílias por quadrante, não
    quatro. Fourier por quadrante existe no disco mas fica de fora, senão o
    bloco não seria comparável com o do WikiArt.
    """
    merged = None
    for region in QUADRANTS:
        base = ROOT / f"outputs/{dataset}_quadrant_features/{region}"
        pieces = []
        for family, path, kind in (
                ("Descriptors", base / "desc_full_features.npy", "npy"),
                ("Hilbert", base / "hilbert_features.npy", "npy"),
                ("SLIC", base / "slic/combined_features.parquet", "parquet")):
            if not path.exists():
                return None
            if kind == "npy":
                values = np.load(path)
                stems = np.load(
                    path.parent / ("desc_full_stems.npy" if family == "Descriptors"
                                   else "hilbert_stems.npy"),
                    allow_pickle=True).astype(str)
                piece = pd.DataFrame(values, columns=[
                    f"Multiscale_{region}_{family}_{i}" for i in range(values.shape[1])])
                piece.insert(0, "stem", stems)
            else:
                table = pd.read_parquet(path)
                columns = [c for c in table.columns
                           if c not in METADATA
                           and not c.startswith(PARAMETER_PREFIXES)
                           and table[c].dtype.kind in "fiub"]
                piece = table[["stem"] + columns].rename(columns={
                    c: f"Multiscale_{region}_{family}_{c}" for c in columns})
            pieces.append(piece.drop_duplicates("stem"))
        block = pieces[0]
        for piece in pieces[1:]:
            block = block.merge(piece, on="stem", how="inner")
        merged = block if merged is None else merged.merge(block, on="stem", how="inner")
    return merged


def load_block(dataset: str, name: str) -> pd.DataFrame | None:
    if name == "Multiscale":
        return load_multiscale(dataset)
    template, kind = {**BLOCKS, **EXTRA_BLOCKS}[name]
    path = ROOT / template.format(ds=dataset)
    if not path.exists():
        return None
    if kind == "npy":
        values = np.load(path)
        stems = np.load(path.parent / NPY_STEMS[name], allow_pickle=True).astype(str)
        frame = pd.DataFrame(values, columns=[f"{name}_{i}" for i in range(values.shape[1])])
        frame.insert(0, "stem", stems)
    else:
        frame = pd.read_parquet(path)
        columns = [c for c in frame.columns
                   if c not in METADATA
                   and not c.startswith(PARAMETER_PREFIXES)
                   and frame[c].dtype.kind in "fiub"]
        frame = frame[["stem"] + columns].rename(
            columns={c: f"{name}_{c}" for c in columns})
    return frame.drop_duplicates("stem")


def k_grid(max_k: int) -> list[int]:
    grid = [k for k in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
                        768, 1024, 1280, 1536, 2048) if k <= max_k]
    if not grid or grid[-1] != max_k:
        grid.append(max_k)
    return grid


def fit_lda(blocks, labels, train, test, n_classes) -> dict:
    """Pico sobre k componentes de PCA, como na grade do WikiArt.

    Sem isso, condições de dimensão alta ficam artificialmente ruins em
    datasets pequenos por penalidade de dimensionalidade, e a tabela mede
    dimensionalidade em vez de conteúdo -- foi o que aconteceu no Rijksmuseum,
    onde somar Multiscale (1.639d para 3.893 amostras de treino) custou 4pp.
    A varredura vai nos dois braços, hand-designed e ResNet: aplicar só de um
    lado trocaria um viés por outro.

    O k é escolhido pelo BAcc do próprio fold de teste, o que é otimista --
    mas é o mesmo protocolo em todas as condições, então as linhas continuam
    comparáveis entre si e com a grade do WikiArt.
    """
    parts_train, parts_test = [], []
    for block in blocks:
        imputer = SimpleImputer(strategy="median").fit(block[train])
        scaler = StandardScaler().fit(imputer.transform(block[train]))
        parts_train.append(scaler.transform(imputer.transform(block[train])))
        parts_test.append(scaler.transform(imputer.transform(block[test])))
    train_x, test_x = np.hstack(parts_train), np.hstack(parts_test)

    max_k = min(train_x.shape[1], len(train) - n_classes)
    reducer = PCA(n_components=max_k, svd_solver="randomized",
                  random_state=SEED).fit(train_x)
    train_p, test_p = reducer.transform(train_x), reducer.transform(test_x)

    best = {"bacc": -1.0}
    for k in k_grid(max_k):
        model = LinearDiscriminantAnalysis(
            n_components=min(n_classes - 1, k),
            priors=np.ones(n_classes) / n_classes).fit(train_p[:, :k], labels[train])
        predicted = model.predict(test_p[:, :k])
        bacc = balanced_accuracy_score(labels[test], predicted)
        if bacc > best["bacc"]:
            best = {"bacc": bacc,
                    "accuracy": accuracy_score(labels[test], predicted),
                    "macro_f1": f1_score(labels[test], predicted, average="macro"),
                    "dimensions": train_x.shape[1], "k": k}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(LABELS), choices=list(LABELS))
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--top-classes", type=int, default=None,
                        help="restringe às N classes mais frequentes, em vez "
                             "de filtrar por --min-per-class (aproximação de "
                             "benchmarks de literatura com menos classes que "
                             "a taxonomia completa)")
    parser.add_argument("--classes-file", type=Path, default=None,
                        help="restringe a uma lista nomeada de classes (uma "
                             "por linha; casamento normalizado por espaço/"
                             "underscore/hífen/acento) -- para replicar um "
                             "benchmark específico da literatura")
    parser.add_argument("--classifier", choices=["lda", "mlp"], default="lda",
                        help="mlp usa um branch por modalidade em dimensão "
                             "nativa (forma de produção); a ResNet entra como "
                             "branch único de 2.048d, que é a forma dela")
    parser.add_argument("--folds", type=int, default=N_FOLDS,
                        help="quantos folds rodar; 1 para sondagem rápida")
    parser.add_argument("--only", nargs="+", default=None,
                        help="rodar só estas condições (nome exato)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="repassado pro MLP; achado do sweep de 21/ago: "
                             "~0.15 reduz overfitting de calibração sem "
                             "custar BAcc, num único fold")
    parser.add_argument("--official-split", action="store_true",
                        help="usa a coluna `split` do metadado (train/test) em vez "
                             "de StratifiedKFold. Necessário no Places-LT: os 996x "
                             "de desbalanceamento vivem SÓ no split de treino "
                             "(max 4.980 / min 5); val e test são balanceados por "
                             "desenho do benchmark, e no pool das 106.300 o "
                             "desbalanceamento cai para 40,8x -- mais brando que os "
                             "139,9x do WikiArt, o que mediria o oposto da razão "
                             "pela qual o dataset foi escolhido. O split `val` fica "
                             "de fora (não treinamos nele nem selecionamos por ele).")
    parser.add_argument("--output", type=Path, default=ROOT / "results/external_modalities.csv")
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        meta_path, label_column = LABELS[dataset]
        meta = pd.read_csv(ROOT / meta_path).dropna(subset=[label_column])
        meta["stem"] = meta["stem"].astype(str)
        meta["label"] = meta[label_column].astype(str)

        available = {}
        for name in BLOCKS:
            block = load_block(dataset, name)
            if block is None:
                log(f"{dataset}: {name} ainda não extraído, pulando")
            else:
                available[name] = block
        if not available:
            continue

        # Interseção de stems entre todos os blocos disponíveis: sem isso, um
        # bloco incompleto encolheria a coorte de forma silenciosa e diferente
        # por modalidade, e as linhas da tabela não seriam comparáveis.
        carry = ["stem", "label"] + (["split"] if "split" in meta.columns else [])
        frame = meta[carry]
        for name, block in available.items():
            frame = frame.merge(block, on="stem", how="inner")
        counts = frame["label"].value_counts()
        if args.classes_file:
            def normalize(value: str) -> str:
                return (value.replace("_", " ").replace("-", " ")
                        .replace("ï", "i").strip().lower())
            wanted = {normalize(line) for line in
                      args.classes_file.read_text().splitlines() if line.strip()}
            by_norm = {normalize(label): label for label in counts.index}
            missing = wanted - set(by_norm)
            if missing:
                raise ValueError(f"classes não encontradas em {dataset}: {missing}")
            frequent = [by_norm[name] for name in wanted]
        elif args.top_classes:
            frequent = counts.head(args.top_classes).index
        else:
            frequent = counts[counts >= args.min_per_class].index
        frame = frame[frame["label"].isin(frequent)]
        labels = LabelEncoder().fit_transform(frame["label"].to_numpy(str))
        n_classes = len(np.unique(labels))
        log(f"\n=== {dataset}: {len(frame):,} imagens, {n_classes} classes, "
            f"{len(available)} blocos: {', '.join(available)}")

        arrays = {name: frame[[c for c in frame.columns if c.startswith(f"{name}_")]]
                  .to_numpy(dtype=np.float32) for name in available}
        # Sanear NaN aqui, e não só dentro do `fit_lda`: o caminho do MLP passa
        # os arrays crus para `evaluate_mlp`, e **um** NaN basta para o
        # StandardScaler propagar, a loss virar NaN e a rede colapsar para
        # classe constante -- foi o que aconteceu nos desenhos (BAcc 9,09% =
        # 1/11 exato, desvio 0,00 nos cinco folds).
        #
        # Origem nos desenhos: 4 imagens de 48.295 com quadrante inferior
        # direito uniforme (papel em branco), o que zera a variância e deixa
        # assimetria e curtose em 0/0. É esperado em arte a traço, não é bug de
        # extração. Mesma convenção de `load_targets` no pipeline do WikiArt.
        for name, values in arrays.items():
            bad = ~np.isfinite(values)
            if bad.any():
                medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan),
                                       axis=0)
                values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
                log(f"  ! {name}: {int(bad.sum())} valores não-finitos em "
                    f"{int(bad.any(axis=1).sum())} imagens, imputados pela mediana")
        if args.official_split:
            if "split" not in frame.columns:
                raise ValueError(f"{dataset}: --official-split pedido, mas o "
                                 f"metadado não tem coluna `split`")
            where = frame["split"].to_numpy(str)
            train_ix = np.flatnonzero(where == "train")
            test_ix = np.flatnonzero(where == "test")
            if not len(train_ix) or not len(test_ix):
                raise ValueError(f"{dataset}: split oficial vazio "
                                 f"(train={len(train_ix)}, test={len(test_ix)})")
            splits = [(train_ix, test_ix)]
            tr = np.bincount(labels[train_ix], minlength=n_classes)
            te = np.bincount(labels[test_ix], minlength=n_classes)
            log(f"  split oficial: train={len(train_ix):,} (desbal "
                f"{tr.max() / max(tr[tr > 0].min(), 1):.1f}x), "
                f"test={len(test_ix):,} (desbal "
                f"{te.max() / max(te[te > 0].min(), 1):.1f}x), "
                f"val descartado={int((where == 'val').sum()):,}")
        else:
            splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                          .split(np.zeros(len(labels), dtype=np.uint8), labels))

        hand = [n for n in available if n != "ResNet50"]
        conditions = [(name, [name]) for name in available]
        if len(hand) > 1:
            conditions.append((f"todas hand-designed ({len(hand)})", hand))
        # Fusão ResNet + hand-designed: achado 24 do WikiArt deu +4,03pp sobre
        # qualquer um sozinho (5/5 folds, p=0,001) -- nunca testado nos datasets
        # externos ainda.
        if "ResNet50" in available and len(hand) > 1:
            conditions.append((f"ResNet50 + hand-designed ({len(hand)})", hand + ["ResNet50"]))
        if args.only:
            conditions = [c for c in conditions if c[0] in args.only]
        for name, names in conditions:
            for fold, (train, test) in enumerate(splits[:args.folds], start=1):
                if args.classifier == "mlp":
                    from evaluate_full_factorial_ablation import evaluate_mlp
                    # Um branch por modalidade em dimensão nativa: é a forma de
                    # produção, e é justamente o que a ResNet densa não tem.
                    metrics = evaluate_mlp([arrays[n] for n in names], labels,
                                           train, test, fold, 150, 15,
                                           label_smoothing=args.label_smoothing)
                    metrics = {k: metrics[k] for k in ("bacc", "accuracy", "macro_f1")}
                    metrics["dimensions"] = sum(arrays[n].shape[1] for n in names)
                    metrics["k"] = 0
                else:
                    metrics = fit_lda([arrays[n] for n in names], labels, train, test, n_classes)
                rows.append({"dataset": dataset, "condition": name, "fold": fold,
                             "n_images": len(frame), "n_classes": n_classes, **metrics})
            block = pd.DataFrame(rows)
            block = block[(block.dataset == dataset) & (block.condition == name)]
            log(f"  {name:<26} {int(block.dimensions.iloc[0]):>5}d  "
                f"BAcc={100 * block.bacc.mean():>5.2f} ± {100 * block.bacc.std():.2f}  "
                f"F1={100 * block.macro_f1.mean():>5.2f}")

    if not rows:
        return
    detail = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False, float_format="%.6f")
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
