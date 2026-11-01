"""Registro declarativo de representações — a fonte única de verdade.

Substitui as **quatro** tabelas de especificação que hoje descrevem os mesmos
arquivos (`CANDIDATES` em `evaluate_dormant_representations`, `EXTRA` em
`evaluate_channel_taxonomy`, `BLOCKS` em `evaluate_external_modalities`,
`load_targets` em `pretrain_cnn_on_handdesigned`), cada uma com seu carregador
e seu conjunto de gotchas.

Três problemas concretos que o registro existe para matar:

1. **Coorte moldada por modalidade.** `final_tfidf_visual_mlp.load_dataset()`
   faz `dropna(subset=["label","caption"])`, então as 79.207 imagens canônicas
   são "tem feature visual **e** tem legenda" -- 695 saem por falta de caption,
   não de feature. Aqui a coorte é **interseção explícita das representações
   pedidas**, sempre reportada, nunca efeito colateral.

2. **Máscaras incompatíveis entre scripts.** A ablação de 7 fatores e a de 8
   usam bits com significados diferentes (Multiscale é 64 numa e 128 na outra),
   e reusar número de máscara entre elas dá resultado errado em silêncio. Aqui
   as máscaras derivam da ordem do registro, então não há número mágico.

3. **Caminhos de dados divergentes.** `evaluate_external_modalities` tinha
   `SimpleImputer` no braço LDA e não no do MLP; nos desenhos, 33 valores
   não-finitos em 4 imagens fizeram o MLP colapsar para classe constante
   (BAcc 9,09% = 1/11 exato, desvio 0,00). Saneamento acontece **uma vez**, no
   carregamento, para todos os consumidores.

O campo `external` é o que transforma uma regra de disciplina em invariante de
código: o bloco Text depende de informação externa (legendas) e por isso não
entra no número principal do paper. `load_blocks(..., allow_external=False)`
torna impossível contaminar a avaliação final por descuido.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Colunas que nunca são feature, em qualquer parquet do projeto.
METADATA = frozenset({"stem", "style", "label", "period", "filename",
                      "sample_id", "elapsed_seconds_all_graphs"})
# Os dois geradores de multigrafo nomeiam os parâmetros de forma diferente
# (`slic_*` num, `segmentation_*` no outro); as 71 features são idênticas.
# Deixar passar poria um dataset com 74 colunas contra 71 do outro, e uma delas
# (`segmentation_actual_segments`) é informativa -- entraria como feature extra
# acidental e as linhas deixariam de ser comparáveis.
PARAMETER_PREFIXES = ("slic_", "segmentation")
# Variância zero em toda a coorte, por construção geométrica -- não é peculiaridade
# do WikiArt nem dado corrompido, então o filtro vale para qualquer dataset:
# `knn_n_leaf_frac`/`del_n_leaf_frac` medem fração de nós de grau 1, mas o grafo
# kNN (k=10) garante grau mínimo >=10 e a triangulação de Delaunay sobre
# centroides bem distribuídos praticamente nunca isola um vértice em grau 1.
# `log_power_r00-r02_a*` vêm de bins (raio, ângulo) da FFT 256x256 que não têm
# nenhum ponto de frequência discreto dentro -- confirmado contando a grade sem
# peso de imagem nenhuma: r00 tem 8 pontos espalhados por 12 setores, r01 tem 12,
# r02 tem 16. Testado: remover não muda BAcc do LDA (+0,02pp, p=0,22, 3/5 folds).
DEAD_COLUMNS = frozenset({
    "knn_n_leaf_frac", "del_n_leaf_frac",
    "log_power_r00_a01", "log_power_r00_a02", "log_power_r00_a04",
    "log_power_r00_a05", "log_power_r00_a07", "log_power_r00_a08",
    "log_power_r00_a10", "log_power_r00_a11",
    "log_power_r01_a02", "log_power_r01_a03", "log_power_r01_a05",
    "log_power_r01_a08", "log_power_r01_a09", "log_power_r01_a11",
    "log_power_r02_a02", "log_power_r02_a05", "log_power_r02_a08",
    "log_power_r02_a11",
})

QUADRANTS = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass(frozen=True)
class Representation:
    """Uma representação, descrita e não implementada.

    `path` e `stems` são gabaritos com `{dataset}`, o que faz o mesmo registro
    servir wikiart, pandora, rijksmuseum e rijksdraw sem código novo.
    """

    name: str
    kind: str                       # npy | parquet | slice | quadrant
    path: str
    family: str
    stems: str | None = None
    external: bool = False          # depende de informação fora da imagem
    archival: bool = False          # medida e aposentada; fora do caminho de reprodução
    columns: tuple[int, int] | None = None    # recorte, para kind="slice"
    datasets: tuple[str, ...] = ()  # vazio = disponível em todos
    note: str = ""
    # Sobrescritas por dataset, para os caminhos legados do WikiArt, que usam
    # `full` onde os datasets externos usam o nome do dataset. Preferir sempre
    # o gabarito; isto existe para não ter que mover 26 GB de arquivo.
    overrides: tuple[tuple[str, str], ...] = ()
    stem_overrides: tuple[tuple[str, str], ...] = ()

    def _pick(self, dataset: str, template: str | None,
              overrides: tuple[tuple[str, str], ...]) -> str | None:
        for name, path in overrides:
            if name == dataset:
                return path
        return template

    def resolve(self, dataset: str) -> Path:
        return ROOT / self._pick(dataset, self.path, self.overrides).format(
            dataset=dataset)

    def resolve_stems(self, dataset: str) -> Path | None:
        template = self._pick(dataset, self.stems, self.stem_overrides)
        return ROOT / template.format(dataset=dataset) if template else None

    def available(self, dataset: str) -> bool:
        if self.datasets and dataset not in self.datasets:
            return False
        if self.kind == "quadrant":
            return all((ROOT / self.path.format(dataset=dataset) / region).is_dir()
                       for region in QUADRANTS)
        return self.resolve(dataset).exists()


# Ordem = ordem de Shapley do WikiArt no MLP, que é a ordem canônica das
# progressões aninhadas. Mudar a ordem muda as máscaras: não reordenar sem
# reprocessar quem depende delas.
REGISTRY: tuple[Representation, ...] = (
    Representation("Descriptors", "npy",
                   "outputs/{dataset}_descriptors/desc_full_features.npy",
                   family="image",
                   stems="outputs/{dataset}_descriptors/desc_full_stems.npy",
                   overrides=(("wikiart", "outputs/wikiart_features/desc_full_features.npy"),),
                   stem_overrides=(("wikiart", "outputs/wikiart_features/desc_full_stems.npy"),),
                   note="130d; 90 são histograma Lab de 30 bins"),
    Representation("SLIC", "parquet",
                   "outputs/{dataset}_slic_features/combined_features.parquet",
                   family="segmentation"),
    Representation("Felzenszwalb", "parquet",
                   "outputs/{dataset}_felz_features/combined_features.parquet",
                   family="segmentation",
                   overrides=(("wikiart",
                               "outputs/wikiart_felzenszwalb_full_s50_m50/combined_features.parquet"),)),
    Representation("Line graph", "parquet",
                   "results/line_graph_{dataset}_features.parquet",
                   family="segmentation", note="L(G) sobre o RAG do SLIC",
                   overrides=(("wikiart", "results/line_graph_full_features.parquet"),)),
    Representation("L(G_felz)", "parquet",
                   "results/line_graph_felz_{dataset}_features.parquet",
                   family="segmentation",
                   overrides=(("wikiart", "results/line_graph_felz_full_features.parquet"),)),
    Representation("Hilbert", "npy",
                   "outputs/{dataset}_hilbert/hilbert_features.npy",
                   family="curve",
                   stems="outputs/{dataset}_hilbert/hilbert_stems.npy",
                   overrides=(("wikiart", "outputs/wikiart_features/hilbert_features.npy"),),
                   stem_overrides=(("wikiart", "outputs/wikiart_features/hilbert_stems.npy"),)),
    Representation("Fourier", "parquet",
                   "results/fourier_{dataset}_features.parquet",
                   family="spectral",
                   overrides=(("wikiart", "results/fourier_full_features.parquet"),)),
    Representation("Multiscale", "quadrant",
                   "outputs/{dataset}_quadrant_features",
                   family="multiscale",
                   note="SLIC+Descriptors+Hilbert nos 4 quadrantes, 1.032d", archival=True),
    # --- fora do pipeline canônico -------------------------------------
    Representation("ResNet50", "npy",
                   "outputs/{dataset}_cnn_frozen/cnn_full_features.npy",
                   family="cnn", external=True,
                   stems="outputs/{dataset}_cnn_frozen/cnn_full_stems.npy",
                   overrides=(("wikiart", "outputs/wikiart_features/cnn_full_features.npy"),),
                   stem_overrides=(("wikiart", "outputs/wikiart_features/cnn_full_stems.npy"),),
                   note="ImageNet: pesos treinados em 1,28M imagens rotuladas"),
    Representation("ResNet101", "npy",
                   "outputs/{dataset}_cnn101_frozen/cnn_full_features.npy",
                   family="cnn", external=True,
                   stems="outputs/{dataset}_cnn101_frozen/cnn_full_stems.npy",
                   note="2.048d pós-GAP, mesma dimensão da ResNet50: a "
                        "diferença entre as duas é profundidade, não largura"),
    Representation("ResNet50 random", "npy",
                   "outputs/{dataset}_cnn_random/cnn_full_features.npy",
                   family="cnn",
                   stems="outputs/{dataset}_cnn_random/cnn_full_stems.npy",
                   overrides=(("wikiart", "outputs/cnn_random_wikiart/cnn_full_features.npy"),),
                   stem_overrides=(("wikiart", "outputs/cnn_random_wikiart/cnn_full_stems.npy"),),
                   note="âncora de transfer zero; não é external, não usa nada de fora"),
    Representation("Gabor", "npy",
                   "outputs/{dataset}_gabor/gabor_features.npy",
                   family="texture",
                   stems="outputs/{dataset}_gabor/gabor_stems.npy",
                   datasets=("pandora", "multitask", "oilpainting", "rijksdraw",
                             "food101", "places_lt", "nct_crc", "nct_crc_val", "minc2500"),
                   note="32d, 4 orientacoes x 4 frequencias x (media,std); "
                        "achado 27 do CLAUDE.md apontava como pendente; "
                        "validado no Pandora 21/ago (LDA +0,25pp p=0,015, "
                        "MLP +0,75pp p=0,020, 5/5 folds nos dois)"),
    Representation("Haralick", "slice",
                   "outputs/haralick_hog_features/features.npy",
                   family="texture", columns=(0, 13),
                   stems="outputs/haralick_hog_features/stems.npy",
                   datasets=("wikiart",)),
    Representation("HOG", "slice",
                   "outputs/haralick_hog_features/features.npy",
                   family="texture", columns=(13, 1777),
                   stems="outputs/haralick_hog_features/stems.npy",
                   datasets=("wikiart",)),
    Representation("Tourist walk", "npy",
                   "outputs/tourist_walk_backes/features.npy",
                   family="texture",
                   stems="outputs/tourist_walk_backes/stems.npy",
                   datasets=("wikiart",),
                   note="assinatura de Backes, h(t+p); marginal nulo medido", archival=True),
    Representation("Triangles SLIC", "parquet",
                   "outputs/triangle_features_slic/triangle_features.parquet",
                   family="segmentation", datasets=("wikiart",), archival=True),
    Representation("Triangles Felz", "parquet",
                   "outputs/triangle_features_felz/triangle_features.parquet",
                   family="segmentation", datasets=("wikiart",), archival=True),
    # Família própria: é a única representação que mede estrutura **entre**
    # escalas. Todo o resto do pipeline fixa uma escala (SLIC em 2.048
    # segmentos, Felzenszwalb em scale=50) e descreve o que há nela.
    Representation("Percolation SLIC", "parquet",
                   "results/percolation_slic_{dataset}_features.parquet",
                   family="hierarchy", datasets=("wikiart",),
                   overrides=(("wikiart",
                               "results/percolation_slic_full_features.parquet"),),
                   note="persistência 0-dim do RAG do SLIC filtrado por cor", archival=True),
    Representation("Percolation Felz", "parquet",
                   "results/percolation_felz_{dataset}_features.parquet",
                   family="hierarchy", datasets=("wikiart",),
                   overrides=(("wikiart",
                               "results/percolation_felz_full_features.parquet"),),
                   note="idem sobre o RAG do Felzenszwalb", archival=True),
    Representation("Random walk", "npy",
                   "outputs/wikiart_features/rw_features.npy", family="curve",
                   stems="outputs/wikiart_features/rw_stems.npy",
                   datasets=("wikiart",),
                   note="duplicata de Hilbert: r² mediano 0,918 com alguma coluna"),
    Representation("Hybrid walk", "npy",
                   "outputs/wikiart_features/hybrid_walk_features.npy",
                   family="curve",
                   stems="outputs/wikiart_features/hybrid_walk_stems.npy",
                   datasets=("wikiart",), archival=True),
    Representation("Square graph", "parquet",
                   "results/square_graph_full_features.parquet",
                   family="segmentation", datasets=("wikiart",), archival=True),
    Representation("Hilbert catch22", "npy",
                   "outputs/hilbert_catch22_full/features.npy", family="curve",
                   stems="outputs/hilbert_catch22_full/stems.npy",
                   datasets=("wikiart",), archival=True),
    Representation("Hilbert spectrum1d", "npy",
                   "outputs/hilbert_spectrum_1d_full/features.npy", family="curve",
                   stems="outputs/hilbert_spectrum_1d_full/stems.npy",
                   datasets=("wikiart",), archival=True),
    Representation("Hilbert visibility", "npy",
                   "outputs/hilbert_visibility_graph_full/features.npy",
                   family="curve",
                   stems="outputs/hilbert_visibility_graph_full/stems.npy",
                   datasets=("wikiart",)),
    Representation("Hilbert wavelet", "npy",
                   "outputs/hilbert_wavelet_pyramid_full/features.npy",
                   family="curve",
                   stems="outputs/hilbert_wavelet_pyramid_full/stems.npy",
                   datasets=("wikiart",), archival=True),
)

BY_NAME = {representation.name: representation for representation in REGISTRY}

# Onde mora o rótulo de cada dataset, e qual coluna é o alvo.
LABELS = {
    "wikiart":     ("data/artemis/wikiart_meta.csv", "style", "stem"),
    "pandora":     ("data/pandora_meta.csv", "style", "stem"),
    "rijksmuseum": ("data/rijksmuseum_labels.csv", "period", "stem"),
    "rijksdraw":   ("data/rijksdraw_labels.csv", "period", "stem"),
    "multitask":   ("data/multitask_meta.csv", "style", "stem"),
    "oilpainting": ("data/oilpainting_meta.csv", "style", "stem"),
    "food101":     ("data/food101_meta.csv", "style", "stem"),
    "places_lt":   ("data/places_lt_meta.csv", "style", "stem"),
    "nct_crc":     ("data/nct_crc_meta.csv", "style", "stem"),
    "nct_crc_val": ("data/nct_crc_val_meta.csv", "style", "stem"),
    "minc2500":    ("data/minc2500_meta.csv", "style", "stem"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")


def strip_image_suffix(value: str) -> str:
    lowered = value.lower()
    for suffix in IMAGE_SUFFIXES:
        if lowered.endswith(suffix):
            return value[: -len(suffix)]
    return value


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns
            if column not in METADATA
            and column not in DEAD_COLUMNS
            and not column.startswith(PARAMETER_PREFIXES)
            and table[column].dtype.kind in "fiub"]


def _read(representation: Representation, dataset: str) -> pd.DataFrame:
    """Devolve um frame com `stem` mais as colunas da representação."""
    path = representation.resolve(dataset)
    prefix = representation.name

    if representation.kind in ("npy", "slice"):
        values = np.asarray(np.load(path, mmap_mode="r"))
        if representation.kind == "slice":
            begin, end = representation.columns
            values = values[:, begin:end]
        stems = np.load(representation.resolve_stems(dataset),
                        allow_pickle=True).astype(str)
        frame = pd.DataFrame(values.astype(np.float32),
                             columns=[f"{prefix}_{i}" for i in range(values.shape[1])])
        frame.insert(0, "stem", stems)
    elif representation.kind == "parquet":
        table = pd.read_parquet(path)
        columns = feature_columns(table)
        frame = table[["stem"] + columns].rename(
            columns={column: f"{prefix}_{column}" for column in columns})
    elif representation.kind == "quadrant":
        frame = _read_quadrants(path, prefix)
    else:
        raise ValueError(f"kind desconhecido: {representation.kind}")
    return frame.drop_duplicates("stem")


def _read_quadrants(base: Path, prefix: str) -> pd.DataFrame:
    """Multiscale = SLIC + Descriptors + Hilbert nos quatro quadrantes.

    Três famílias por quadrante, não quatro: Fourier por quadrante existe no
    disco mas fica de fora, senão o bloco deixa de ser comparável com o do
    WikiArt, onde ele foi definido assim.
    """
    merged = None
    for region in QUADRANTS:
        directory = base / region
        pieces = []
        for family, path, stems in (
                ("Descriptors", directory / "desc_full_features.npy",
                 directory / "desc_full_stems.npy"),
                ("Hilbert", directory / "hilbert_features.npy",
                 directory / "hilbert_stems.npy"),
                ("SLIC", directory / "slic/combined_features.parquet", None)):
            if stems is not None:
                values = np.load(path)
                piece = pd.DataFrame(values.astype(np.float32), columns=[
                    f"{prefix}_{region}_{family}_{i}" for i in range(values.shape[1])])
                piece.insert(0, "stem", np.load(stems, allow_pickle=True).astype(str))
            else:
                table = pd.read_parquet(path)
                columns = feature_columns(table)
                piece = table[["stem"] + columns].rename(columns={
                    c: f"{prefix}_{region}_{family}_{c}" for c in columns})
            pieces.append(piece.drop_duplicates("stem"))
        block = pieces[0]
        for piece in pieces[1:]:
            block = block.merge(piece, on="stem", how="inner")
        merged = block if merged is None else merged.merge(block, on="stem", how="inner")
    return merged


def sanitize(values: np.ndarray, name: str) -> np.ndarray:
    """Imputa não-finitos pela mediana da coluna, e diz que fez.

    Silêncio aqui já custou um experimento: um NaN basta para o
    StandardScaler propagar, a loss virar NaN e a rede colapsar para classe
    constante. Nos desenhos foram 33 valores em 4 imagens (quadrante uniforme
    -> variância zero -> assimetria e curtose em 0/0) e o MLP deu BAcc de
    9,09% = 1/11 exato com desvio 0,00 nos cinco folds.
    """
    bad = ~np.isfinite(values)
    if not bad.any():
        return values
    medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    values = values.copy()
    values[bad] = np.take(np.nan_to_num(medians), np.where(bad)[1])
    log(f"  ! {name}: {int(bad.sum())} não-finitos em "
        f"{int(bad.any(axis=1).sum())} imagens, imputados pela mediana")
    return values


@dataclass
class Cohort:
    """O que foi carregado, e a que custo em imagens."""

    dataset: str
    stems: np.ndarray
    labels: np.ndarray
    blocks: dict[str, np.ndarray]
    dropped: dict[str, int] = field(default_factory=dict)
    # Nomes das colunas, na ordem das do bloco. Existem porque várias perguntas
    # do projeto são sobre **sub-blocos** e não sobre blocos inteiros: restringir
    # Multiscale à família Descriptors (achado 11) ou medir só uma família de
    # colunas de uma representação. Sem isto o consumidor teria que recarregar o
    # parquet por fora, que é justamente a divergência de carregador que o
    # registro existe para matar.
    columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        return sum(block.shape[1] for block in self.blocks.values())

    def matrix(self, names=None) -> np.ndarray:
        chosen = list(self.blocks) if names is None else list(names)
        return np.hstack([self.blocks[name] for name in chosen])

    def select(self, name: str, predicate) -> np.ndarray:
        """Sub-bloco pelas colunas cujo nome satisfaz `predicate`.

        O nome vem sem o prefixo da representação, que o carregador carimba e
        não interessa a quem filtra.
        """
        prefix = f"{name}_"
        keep = [index for index, column in enumerate(self.columns[name])
                if predicate(column[len(prefix):] if column.startswith(prefix)
                             else column)]
        if not keep:
            raise ValueError(f"{name}: nenhuma coluna sobreviveu ao filtro")
        return self.blocks[name][:, keep]


def load_blocks(dataset: str, names=None, *, allow_external: bool = True,
                min_per_class: int = 0) -> Cohort:
    """Carrega as representações pedidas, com a coorte como interseção explícita.

    `allow_external=False` recusa blocos que dependem de informação fora da
    imagem (hoje só a ResNet do ImageNet e, quando entrar no registro, o Text).
    É a trava que impede o número principal do paper de ser contaminado por
    descuido -- a alegação é competir **sem supervisão externa**.
    """
    # O conjunto implícito serve o caminho de REPRODUÇÃO: pedir tudo devolve o
    # que o artigo usa, não o catálogo inteiro. Bloco aposentado continua
    # carregável pelo nome -- a §6 só vale enquanto o negativo for verificável.
    requested = list(names) if names else [r.name for r in REGISTRY if not r.archival]
    unknown = [name for name in requested if name not in BY_NAME]
    if unknown:
        raise KeyError(f"não estão no registro: {unknown}")

    chosen = []
    for name in requested:
        representation = BY_NAME[name]
        if representation.archival:
            log(f"  ! {name}: bloco APOSENTADO (§6), fora do caminho de reprodução")
        if representation.external and not allow_external:
            log(f"  - {name}: external, recusado por allow_external=False")
            continue
        if not representation.available(dataset):
            log(f"  - {name}: não extraído para {dataset}")
            continue
        chosen.append(representation)
    if not chosen:
        raise RuntimeError(f"nenhuma representação disponível para {dataset}")

    label_path, label_column, key = LABELS[dataset]
    meta = pd.read_csv(ROOT / label_path).dropna(subset=[label_column])
    # `keep="last"` não é arbitrário: 34 pinturas do WikiArt aparecem duas vezes
    # com estilos **diferentes** (o WikiArt atribui múltiplos estilos a algumas
    # obras e o CSV guarda as duas linhas). O carregador canônico
    # `final_tfidf_visual_mlp.load_dataset` resolve com `dict(zip(...))`, que
    # fica com a última ocorrência; ficar com a primeira mudaria o rótulo de 34
    # imagens e quebraria a comparabilidade com todos os resultados anteriores.
    # A ambiguidade é real e pequena (0,04% da coorte), mas a escolha tem que
    # ser a mesma em todo lugar.
    meta = meta.drop_duplicates(key, keep="last")
    frame = meta[[key, label_column]].rename(columns={key: "stem"})
    # Tirar extensão **só** quando ela é extensão: stems do WikiArt como
    # `a.y.-jackson_algoma-in-november-1935` contêm pontos, e `Path(v).stem`
    # cego os truncaria em `a.y`, colidindo milhares de linhas num só stem.
    frame["stem"] = frame["stem"].astype(str).map(strip_image_suffix)

    dropped, running = {}, len(frame)
    for representation in chosen:
        piece = _read(representation, dataset)
        frame = frame.merge(piece, on="stem", how="inner", validate="one_to_one")
        dropped[representation.name] = running - len(frame)
        running = len(frame)

    if min_per_class:
        counts = frame[label_column].value_counts()
        keep = counts[counts >= min_per_class].index
        removed = len(frame) - int(frame[label_column].isin(keep).sum())
        frame = frame[frame[label_column].isin(keep)]
        if removed:
            dropped[f"<{min_per_class} por classe"] = removed

    frame = frame.sort_values("stem").reset_index(drop=True)
    blocks, column_names = {}, {}
    for representation in chosen:
        columns = [c for c in frame.columns if c.startswith(f"{representation.name}_")]
        blocks[representation.name] = sanitize(
            frame[columns].to_numpy(dtype=np.float32), representation.name)
        column_names[representation.name] = columns

    cohort = Cohort(dataset, frame["stem"].to_numpy(dtype=str),
                    frame[label_column].to_numpy(dtype=str), blocks, dropped,
                    column_names)
    log(f"{dataset}: {len(cohort.stems):,} imagens | {cohort.dimensions}d em "
        f"{len(blocks)} blocos | {len(np.unique(cohort.labels))} classes")
    for name, count in dropped.items():
        if count:
            log(f"    −{count:,} ao intersectar {name}")
    return cohort


def mask_of(names) -> int:
    """Máscara de bits derivada da ordem do registro, não de número mágico."""
    order = {representation.name: index for index, representation in enumerate(REGISTRY)}
    return sum(1 << order[name] for name in names)


def names_of(mask: int) -> list[str]:
    return [representation.name for index, representation in enumerate(REGISTRY)
            if mask >> index & 1]
