# Experimentos

Esta pasta contém scripts de avaliação, comparação e geração de figuras. Os
artefatos que documentam seus resultados ficam em `../results/`.

## Convenção

- Cada experimento deve manter o script que o executa e produzir uma tabela
  agregada por fold e uma tabela-resumo em `results/`;
- `*_results.csv`, `*_folds.csv`, `*_comparisons.csv`, `*_summary.csv` e figuras
  selecionadas são versionados;
- predições por imagem, features, coortes de trabalho e caches são
  reconstituíveis e ficam ignorados em `results/`;
- novos scripts devem usar caminhos relativos à raiz do repositório e nunca
  depender do diretório de trabalho atual.

## Pontos de entrada

| Tema | Scripts principais | Resultado de referência |
|---|---|---|
| Método visual e ablações | `final_tfidf_visual_mlp.py`, `ablation_wikiart.py` | `../results/metodos_final.csv` |
| Comparação CNN/transfer | `../src/baselines/train_resnet_wikiart.py` | `../results/metodos_final.csv` |
| Controles clássicos | `evaluate_visual_rf_svm.py`, `evaluate_visual_knn.py` | `../results/metodos_final.csv` |
| Grafos de segunda ordem | `extract_line_graph_features.py`, `evaluate_line_graph_3k.py` | `../results/line_graph_full_results.csv` |
| Segmentação | `compare_segmentations_pilot.py`, `compare_segmentation_classifiers_12k.py` | `../results/segmentation_winners_12k_results.csv` |
| Texto/captions | `evaluate_catr_full_3clf.py`, `compare_text_controls_mlp.py` | `../results/catr_full_3clf_results.csv` |
| Pareto | `plot_pareto_time_bacc.py`, `plot_pareto_pretreino_parametros.py` | `../results/pareto_*.csv` |

Os scripts ainda aceitam os caminhos históricos `paper/...`, pois `paper` é um
atalho para `results`. Código novo deve usar `results/...` explicitamente.
O estado narrativo e as decisões de pesquisa ficam em `../conversa.md`.
