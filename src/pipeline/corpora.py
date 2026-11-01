"""Nomes de blocos e o esquema do multigrafo: o vocabulário do banco.

**Por que este arquivo existe.** Três constantes que definem *o que é o banco de
representações* estavam cada uma dentro de um script de experimento diferente, e
duas delas atrás de cadeias de três imports:

    VISUAL          experiments/evaluate_fusion_delta_across_corpora.py
    HAND8           experiments/evaluate_minc2500_official.py
    FEATURE_COLUMNS experiments/compare_segmentations_pilot.py
                      <- experiments/compare_slic_multigraph.py  (RAG/EDGE_FEATURES)

`VISUAL` e `HAND8` **não são a mesma lista**, e confundi-las é um erro silencioso
que eu cometi em 06/set: `VISUAL` são as SETE modalidades comuns aos cinco
corpora (sem Gabor, que nunca foi extraído para o WikiArt), e `HAND8` são as
OITO de produção. Somar `"Fourier"` a `VISUAL` -- que já o contém -- faz
`load_blocks` duplicar colunas e renomear **sem erro nenhum** (§10.8), e o
hand-8 sai com nove blocos.

DÍVIDA DECLARADA: `load_corpus` e `load_graph_blocks` continuam em
`experiments/`. Os dois dependem de `load_block` e da configuração de caminhos
de `evaluate_external_modalities.py`, cujo lugar certo é `registry.py` -- que o
§2 já declara fonte única de verdade dos caminhos. Fundir os dois é uma
refatoração maior; fazê-la pela metade criaria uma segunda tabela de caminhos,
que é exatamente o que o §2 proíbe.
"""
from __future__ import annotations

import numpy as np

# As 7 modalidades presentes nos CINCO corpora. Gabor fica de fora: nunca foi
# extraído para o WikiArt e as imagens daquele corpus não estão mais em disco,
# então incluí-lo custaria a comparabilidade que é o ponto daquela medição.
VISUAL = ["Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
          "Hilbert", "Fourier"]

# As 8 de produção (§3). É `VISUAL + Gabor`, e é esta que o artigo reporta.
HAND8 = ("Descriptors", "Felzenszwalb", "SLIC", "Line graph", "L(G_felz)",
         "Hilbert", "Fourier", "Gabor")
assert set(HAND8) == set(VISUAL) | {"Gabor"} and len(HAND8) == 8

# --- esquema do bloco de multigrafo (71 dims por segmentação, §3) ------------
RAG_FEATURES = [
    "n_nodes", "n_edges", "density", "area_gini", "area_entropy",
    "area_max_frac", "area_mean", "area_std", "color_dist_mean",
    "color_dist_std", "color_dist_max", "spatial_dist_mean",
    "spatial_dist_std", "degree_mean", "degree_std", "degree_max",
    "hub_degree_ratio", "n_leaf_frac", "degree_gini",
    "area_second_max_frac", "area_cv", "spatial_density_cv",
    "spatial_density_max_block_frac", "L_mean_w", "L_std_w", "a_mean_w",
    "a_std_w", "b_mean_w", "b_std_w", "chroma_mean_w", "chroma_std_w",
    "warm_frac",
]
KNN_FEATURES = RAG_FEATURES[:21]

# Estas dependem do conjunto de arestas. As demais descrevem os nós SLIC
# compartilhados e seriam repetidas uma vez por família de grafo.
EDGE_FEATURES = [
    "n_edges", "density", "color_dist_mean", "color_dist_std",
    "color_dist_max", "spatial_dist_mean", "spatial_dist_std",
    "degree_mean", "degree_std", "degree_max", "hub_degree_ratio",
    "n_leaf_frac", "degree_gini",
]

FEATURE_COLUMNS = (
    RAG_FEATURES
    + [f"knn_{column}" for column in EDGE_FEATURES]
    + [f"del_{column}" for column in EDGE_FEATURES]
    + [f"mst_{column}" for column in EDGE_FEATURES]
)
assert len(FEATURE_COLUMNS) == 71

# Duas destas 71 são constantes na coorte e o `DEAD_COLUMNS` do registry as
# remove: `knn_n_leaf_frac` e `del_n_leaf_frac` (§3). São frações de folhas que
# dão constante -- por isso o banco emite 639d e o classificador usa 617d.
DEAD_MULTIGRAPH_COLUMNS = ("knn_n_leaf_frac", "del_n_leaf_frac")


def clean(block: np.ndarray) -> np.ndarray:
    """Não-finitos viram a mediana da coluna. Presente em Fourier e nos grafos."""
    bad = ~np.isfinite(block)
    if bad.any():
        median = np.nanmedian(np.where(np.isfinite(block), block, np.nan), axis=0)
        block[bad] = np.take(np.nan_to_num(median), np.where(bad)[1])
    return block
