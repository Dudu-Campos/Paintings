# attic — execuções exploratórias arquivadas

Isto **não é lixo**, e por isso não foi apagado: são as ablações, sondas e
variantes que produziram os resultados registrados no `CLAUDE.md`, em especial
o **§6 (negativos medidos, não repetir)**. Um negativo só é útil enquanto for
verificável; apagar o script que o mediu transformaria "HOG −1,21pp" numa
alegação sem procedência.

O que saiu de `experiments/` e `src/` em 06/set: **238 `.py` e 33 `.sh`**.
Ficaram 167 `.py` e 26 `.sh`.

## Critério de permanência (o que NÃO veio para cá)

Um script ficou em `experiments/` se satisfaz qualquer um destes:

1. é citado pelo nome no `CLAUDE.md`;
2. foi modificado nos últimos 12 dias (trabalho ativo — a regra automática
   perdia as duas últimas sessões inteiras);
3. produz um artefato `results/`ou `outputs/` citado no `CLAUDE.md`, **inclusive
   em forma de diretório** (`outputs/places_lt_similarity_graph/` etc. — cinco
   scripts foram resgatados por essa cláusula depois de a primeira versão da
   regra só casar nomes com extensão);
4. é um `extract_*` cujo artefato aparece em `src/pipeline/registry.py`;
5. é `sweep_theta_*` / `evaluate_theta_*` (§5.17, resultado corrente);
6. está em `experiments/image-to-text/` (o captioner estudante do §5.18);
7. é importado, transitivamente, por qualquer um dos acima.

Verificado depois da mudança: **nenhum script vivo importa nada daqui**, os 187
arquivos restantes compilam, e 17 módulos-chave importam.

## Duplicatas que isto revelou

`rich_similarity_features` -- a função que produz a variante rica de 82d do grafo
de similaridade, o número de produção do §3 -- estava definida em **três**
arquivos. As implementações de `evaluate_production_visual_rich_similarity_graph`
e `evaluate_production_visual_extended_similarity_graph` são **byte-idênticas**
(mesmo md5, 28 linhas), então arquivar a primeira é seguro; a citada no §3 é a
segunda e continua viva.

## Recuperar

O estado pré-limpeza é o commit `dc273766e`. Para trazer um arquivo de volta:

```bash
git mv attic/experiments/<nome>.py experiments/<nome>.py
```

Note que um script recuperado pode importar símbolos que se mudaram para
`src/pipeline/{models,protocol,corpora}.py` em `c469357a9` — os nomes antigos
seguem re-exportados dos arquivos de origem, então o import continua
funcionando.

## Resgates posteriores

**06/set, após a auditoria das figuras de scaling law.** A regra casava
artefatos **citados no CLAUDE.md**, e o CLAUDE.md cita o `.png` da figura -- não
o `.csv` intermediário que o plotter lê e que outro script escreve. A metade que
*calcula* de um par calcula/desenha ficava sem cláusula.

Varredura precisa (para cada `plot_*` vivo, o insumo tem produtor vivo?) achou
exatamente **dois** casos, os dois resgatados:

| insumo | plotter vivo | produtor resgatado |
|---|---|---|
| `results/pca_curve_by_subset_multitask.csv` | `plot_pca_curve_by_subset_multitask.py` | `evaluate_pca_curve_by_subset_multitask.py` (§5.3) |
| `results/nested_mlp_multitask.csv` | `plot_cost_vs_accuracy_multitask.py` | `evaluate_nested_mlp_multitask.py` |

Falso alarme conferido no caminho: a grade do §5.1 **não** estava órfã --
`evaluate_representation_data_grid_multitask.py` está vivo. O que ficou aqui é
`evaluate_representation_data_grid.py`, a variante antiga do WikiArt.
