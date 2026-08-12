# Artefatos do artigo

Esta pasta reúne scripts de avaliação, tabelas compactas e figuras geradas para
o artigo. Cada experimento deve manter no mínimo: o script que o executa, uma
tabela agregada por fold e uma tabela-resumo; gráficos são versionados quando
forem candidatos a figura do texto.

## Convenção de artefatos

- `*_results.csv`, `*_folds.csv`, `*_comparisons.csv`, `*_summary.csv`: saídas
  compactas que documentam o resultado e são versionadas;
- `*_predictions.csv`, `*_features.parquet`, `cache_*/`, `*.npy`, `*.npz`:
  predições por imagem, features e caches reconstituíveis; ficam fora do Git;
- `*_sample.csv`: coortes/materializações de trabalho; ficam locais, salvo
  exceção explicitamente curada;
- `*.png`: figuras e diagnósticos selecionados.

## Pontos de entrada

| Tema | Scripts principais | Resultado de referência |
|---|---|---|
| Método visual e ablações | `final_tfidf_visual_mlp.py`, `ablation_wikiart.py` | `metodos_final.csv` |
| Comparação CNN/transfer | `src/baselines/train_resnet_wikiart.py` | `metodos_final.csv` |
| Controles clássicos | `evaluate_visual_rf_svm.py`, `evaluate_visual_knn.py` | `metodos_final.csv` |
| Grafos de segunda ordem | `extract_line_graph_features.py`, `evaluate_line_graph_3k.py` | `line_graph_full_results.csv` |
| Segmentação | `compare_segmentations_pilot.py`, `compare_segmentation_classifiers_12k.py` | `segmentation_winners_12k_results.csv` |
| Texto/captions | `evaluate_catr_full_3clf.py`, `compare_text_controls_mlp.py` | `catr_full_3clf_results.csv` |
| Pareto | `plot_pareto_time_bacc.py`, `plot_pareto_pretreino_parametros.py` | `pareto_*.csv` |

O estado narrativo e decisões de pesquisa em curso ficam em `../conversa.md`;
este README é apenas o contrato de organização dos artefatos.
