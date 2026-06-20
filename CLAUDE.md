# CLAUDE.md — Pipeline Pinturas

Contexto para o Claude Code ao trabalhar neste repositório. Cobre apenas o pipeline C++/Python de percolação cromática de pinturas (branch Paintings da tese). Colocar este arquivo na raiz do repo (`/exp/duducampos/Paintings/CLAUDE.md`).

## Visão geral

Pipeline que converte pinturas em grafos via percolação cromática (componentes conexos sob threshold de distância Lab), extrai features estruturais desses grafos, e testa se essas features discriminam estilo/artista. Alvo: submissão ao CNA 2026 (Complex Networks), deadline ~2 de setembro.

## Ambiente e hardware

- GPU: A100 (64GB RAM) para experimentos Python.
- CPU: migrando para Ryzen Threadripper 3990X (64 cores) — pipeline de pinturas sendo reprocessado do zero nessa máquina.
- C++: OpenMP para paralelismo, AddressSanitizer para debug de memória.
- Python: pybind11 para bindings quando necessário, numpy structured arrays.
- Dataset: imagens estilo WikiArt + `meta.csv` (colunas `filename`, `artist`, `style`, `genre`).

## Estrutura de arquivos

- **`coarse2.cpp`** — motor de percolação. Union-find sobre grafo implícito de pixels 4-conectados, distância cromática em Lab (ΔE). Grade de thresholds: `{2,4,6,8,10,12,15,18,22,27,33,40}`. Dois modos de CLI: `--features` (extrai estatísticas agregadas por threshold) e `--graph --epsilon_star N` (materializa o grafo num ε específico).
- **`eps_star_ssim.cpp`** — busca binária de ε* por imagem via SSIM. Mediana saiu baixa (~1.66) — não é o critério operacional final, mantido só como referência/comparação.
- **`compute_eps_star_entropy.py`** — calcula ε*_entropy por imagem via detecção de joelho (`find_knee_decreasing`, Kneedle simplificado) na curva de entropia vs ε (decrescente). `min_eps_floor=6.0`: valores abaixo são marcados `is_degenerate` e substituídos por fallback (mediana dos casos válidos). Este é o critério de coarsening operacional atual, substituindo o antigo ε=15 fixo.
- **`run_coarsening2.py`** — orquestra materialização de grafos em disco a partir do lookup de ε*. `load_eps_star_lookup` detecta automaticamente qualquer coluna `eps_star_*` no parquet (genérico, não hardcoded para uma coluna específica) — emite aviso se houver mais de uma.
- **`lda_graph_features.py`** — extração de features de grafo (RAG + topológicas). Funções-chave:
  - `select_balanced_paths` — seleciona arquivos balanceados por estilo (min/max) antes da extração. Usar sempre `--balanced_min`/`--balanced_max` em qualquer comparação com o baseline de CNN, que foi avaliado em 104 estilos balanceados (min=31, max=100) — sem isso a comparação é inválida (a versão sem balanceamento usa só 21 estilos desbalanceados).
  - `read_coarse2_full` — lê a curva completa de percolação, incluindo `entropy` por threshold (a versão antiga `read_coarse2` descartava essa coluna).
  - `spatial_density_cv(nodes, grid_size=3)` — CV (std/mean) da contagem de nós por bloco numa grade 3×3.
  - `connected_components_stats(nodes, edges)` — via union-find: `n_components`, `giant_frac`, `giant_to_second_ratio`, `n_components_size_gt1`.
  - `remove_giant_component(nodes, edges, mode='dominant_node')` — remove só o nó de maior `area_frac`. **Nunca** usar um modo que remova o componente conexo inteiro (ver Gotchas).
  - `extract_graph_features_no_giant` — recalcula as 15 features básicas sobre o subgrafo sem o nó dominante.
- **`lda_knn_features.py`** / **`lda_combined_features.py`** — mesma lógica de balanceamento e remoção de nó dominante, para as representações kNN e combinada.
- **`xgboost_ablation.py`** — ablation com `compute_sample_weight('balanced', y_train)`, `scoring='balanced_accuracy'` no RandomizedSearchCV, grid restrito (`max_depth` 1–3, `n_estimators` ≤100, `min_child_weight` 3–10, `reg_alpha`/`reg_lambda`) para combater overfitting severo em classes com poucos exemplos.
- **`plot_sample_graphs_by_style.py`**, **`plot_spatial_density_violin.py`**, **`plot_connected_components_violin.py`**, **`plot_graph_structure_vs_epsilon.py`** — scripts de visualização standalone, sem dependências entre si.
- **`explore_features.ipynb`** — notebook exploratório com caminhos hardcoded: `FEATURES_PATH='./plots_v3/graph_features.parquet'`, `META_PATH='/exp/duducampos/Paintings/meta/meta.csv'` — ajustar se mudar de máquina ou de pasta de output.

## Como rodar (referência rápida)

```bash
# Materializar grafos com eps individual a partir do lookup eps_star_*
python run_coarsening2.py \
    --coarse_dir ./coarsening_output_v3 \
    --eps_lookup ./plots_v3/eps_star_entropy.parquet

# Extrair features kNN balanceadas por estilo
python lda_knn_features.py \
    --coarse_dir ./coarsening_output_v3/coarse \
    --meta /exp/duducampos/Paintings/meta/meta.csv \
    --output ./plots_v3 \
    --balanced_min 31 --balanced_max 100 --min_count 0 \
    --k 10 --workers 32

# Ablation balanceado
python xgboost_ablation.py --features ./plots_v3/graph_features.parquet
```

## Bugs conhecidos e armadilhas

- **O RAG produzido por `coarse2.cpp` é sempre um único componente conexo**, em qualquer ε. Causa: `add_edge()` conecta todo par de pixels/regiões espacialmente adjacentes sem filtrar por `color_dist` — essa distância é calculada e armazenada, mas nunca usada como critério de inclusão da aresta. Consequência prática: `n_components` sobre o RAG é trivialmente sempre 1 e não serve como feature discriminativa.
- **`remove_giant_component` — não remover o componente inteiro.** A primeira versão fazia isso e, dado o ponto acima, zerava 100% dos grafos (todas as imagens ficavam com `n_nodes=0`). A versão correta usa `mode='dominant_node'`, removendo só o nó de maior `area_frac` e preservando as arestas entre os demais.
- **dtype em arrays vazios:** `keep_idx` pode virar `float64` quando a lista de índices é vazia (caso `n_nodes<=1`), causando `IndexError` em indexação. Especificar `dtype=np.int64` explicitamente.
- **`eps_star_entropy` degenerado em ~7.7% das imagens** (303/3951 testadas): o algoritmo de joelho falha silenciosamente quando o ponto de colapso está muito próximo do início da curva, produzindo grafos com >100k nós (caso extremo: ~503k nós, quase 1 pixel = 1 nó). O fallback (`min_eps_floor` + substituição pela mediana) já está implementado, mas **ainda não foi aplicado retroativamente** ao dataset `coarsening_output_v3` já materializado.
- **Anti-pattern de performance em motores de coarsening/percolação C++:** nunca chamar uma função que reconstrua estatísticas globais varrendo todos os nós ativos a cada merge/update individual (isso dá O(N²) total). Atualizar só os vizinhos diretos afetados — O(grau local) por operação, O(N log N) total.
- **Resize de imagem:** usar bilinear, não nearest-neighbor — nearest-neighbor introduz artefatos de pixelização que distorcem as estatísticas de cor agregadas.
- **SSIM:** calcular nos 3 canais Lab, não só em luminância (L) — só-L subestima a perda de informação cromática.
- **Validação cruzada de heurísticas de outra IA:** uma sugestão do Gemini (cap de 100k combinações de features) introduziu viés lexicográfico para nós de alto grau. Revisar criticamente qualquer heurística vinda de fora antes de aceitar, especialmente em código de seleção/amostragem.

## Convenções de colaboração neste código

- Discutir a abordagem/metodologia antes de escrever código — pausar para validar a ideia conceitualmente primeiro, não implementar de imediato.
- Preferir a configuração mais simples e válida antes de expandir complexidade (ex.: 4-conectividade, distância Lab padrão, threshold único) — evita confundir complexidade com rigor.
- Respostas diretas e concisas no código e nos comentários, sem enchimento.
- Apontar quando uma sugestão estiver excessivamente complexa, circular, ou não enraizada na contribuição real do projeto — não seguir por inércia.
