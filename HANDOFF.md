# Handoff — Paintings / GIST

> **Nota de organização (12 de agosto de 2026).** Este arquivo preserva as
> decisões e a migração histórica para SLIC. Para o estado experimental atual,
> resultados consolidados e coordenação entre máquinas, use `conversa.md` e
> `results/metodos_final.csv`. Dados, caches e outputs pesados são locais e ficam
> ignorados pelo Git.

Atualizado originalmente em 10 de agosto de 2026, durante a migração do
gerador de grafos para SLIC.

## Decisão atual

O pipeline principal não usa mais a segmentação por percolação cromática.
O substituto é SLIC com orçamento fixo e independente de 2.048 segmentos,
`compactness=10`, imagem redimensionada para lado máximo de 1.024 pixels.

Na avaliação pareada completa, com 75.500 imagens e os mesmos cinco folds, a
balanced accuracy foi `0.2517 ± 0.0101` para SLIC-2048 e `0.2167 ± 0.0036`
para o RAG legado. O ganho absoluto do SLIC foi de 3,50 pontos percentuais;
portanto, não houve evidência empírica que justificasse manter a segmentação
específica.

Concatenando RAG, kNN, Delaunay e MST, a balanced accuracy sobe para
`0.3106 ± 0.0086`. A concatenação bruta de 117 colunas e a versão de 71
colunas, que mantém os atributos compartilhados dos nós apenas uma vez,
produziram exatamente os mesmos scores nos cinco folds. Adotar a versão de 71
features como representação multigrafo principal.

Na curva de fusão pareada em 74.954 imagens, o melhor ponto foi o multigrafo
SLIC-71 + descritores-130 + Hilbert-57 + sentimento-16: 274 features e
`0.3815 ± 0.0100` de balanced accuracy. A mesma fusão com o multigrafo legado
obteve `0.3672 ± 0.0109`. Random walk e hybrid walk reduziram o resultado nas
duas curvas pareadas; não incluí-los no pipeline final.

Na mesma coorte pareada e na configuração final de 274 features:

- LDA: legado `0.3672 ± 0.0109`, SLIC `0.3815 ± 0.0100`;
- XGBoost: legado `0.3824 ± 0.0094`, SLIC `0.3855 ± 0.0109`;
- MLP: legado `0.4258 ± 0.0109`, SLIC `0.4427 ± 0.0083`.

A MLP usa validação interna dentro do treino de cada fold para early stopping.

## Fusão SLIC-only

A partir destes experimentos, as avaliações de fusão usam somente o multigrafo
SLIC e a maior coorte comum às representações ativas: 79.207 imagens.

- O stacking de probabilidades OOF com quatro LDAs e meta-classificador
  logístico obteve `0.3812 ± 0.0043`, essencialmente igual à concatenação
  antecipada por LDA, `0.3814 ± 0.0029`.
- A MLP com encoders separados de 128 dimensões para multigrafo SLIC,
  descritores, Hilbert e sentimento obteve `0.4585 ± 0.0033`, contra
  `0.4578 ± 0.0100` da MLP plana nos mesmos folds.
- A diferença média da MLP ramificada foi apenas +0,07 ponto percentual. Ela
  usa 468.763 parâmetros contra 573.979 da MLP plana e apresentou menor
  dispersão, mas não há evidência de ganho relevante de acurácia.

Todos os early stoppings neurais usam validação interna retirada apenas do
treino externo; o fold externo é avaliado uma única vez.

## Ablação de segmentadores

Os parâmetros de SLICO, watershed, Felzenszwalb e Quickshift foram calibrados
em 27 imagens usando apenas proximidade ao número efetivo de nós do SLIC e
tempo; nenhuma classe ou acurácia participou da escolha. Os parâmetros
selecionados foram SLICO com 1.800 segmentos, watershed com 1.800 marcadores,
Felzenszwalb com `scale=50, min_size=50` e Quickshift com
`kernel_size=3, max_dist=10`.

No piloto pareado de 2.693 imagens com LDA e 71 features multigrafo:

- SLIC: `0.2651 ± 0.0156`;
- SLICO: `0.2671 ± 0.0153`;
- watershed: `0.2528 ± 0.0172`;
- Felzenszwalb: `0.2858 ± 0.0166`;
- Quickshift: `0.2849 ± 0.0131`.

Felzenszwalb e Quickshift foram expandidos para 12.213 imagens comuns. Nessa
coorte, SLIC / Felzenszwalb / Quickshift obtiveram, respectivamente:

- LDA: `0.2969 / 0.3127 / 0.3104`;
- XGBoost: `0.3306 / 0.3302 / 0.3274`;
- MLP: `0.3212 / 0.3374 / 0.3321`.

Assim, Felzenszwalb ganhou cerca de 1,6 ponto percentual com LDA e MLP e
empatou com XGBoost. Quickshift também ganhou com LDA e MLP, mas perdeu com
XGBoost e gerou mais nós. Felzenszwalb é o candidato mais forte para uma
extração completa.

Felzenszwalb é mais adaptativo: p5–p95 de 699–3.670 nós, contra 1.272–2.022 do
SLIC. Porém, removendo os indicadores diretos de orçamento de regiões, a LDA
ainda obteve `0.3064` com Felzenszwalb contra `0.2932` com SLIC. Portanto,
a maior parte do ganho vem das regiões/estruturas geradas, não apenas da
contagem adaptativa de nós.

A máquina tem 64 núcleos físicos e 128 threads lógicas. Os extratores escalam
bem até 64 processos; Quickshift atingiu aproximadamente 12 imagens/s quando
executado sozinho com 64 workers.

## Geração SLIC concluída

- Unidade encerrada: `paintings-slic-production.service` (exit status 0)
- Saída: `outputs/wikiart_slic_features/`
- Resultado: 79.902 stems únicos, 27 estilos, zero falhas
- Tempo: aproximadamente 58 minutos de parede e 31,1 CPU-hours agregadas
- Checkpoint: a cada 500 imagens
- O processo gera quatro parquets a partir de uma única segmentação por imagem:
  - `graph_features.parquet`: RAG espacial, 32 features;
  - `knn_features.parquet`: kNN em Lab+XY, 21 features;
  - `delaunay_features.parquet`: Delaunay dos centroides, 32 features;
  - `mst_features.parquet`: MST ponderada por cor, 32 features.

Comandos de acompanhamento:

```bash
systemctl --user show paintings-slic-production.service \
  -p ActiveState -p SubState -p MainPID -p MemoryCurrent -p ExecMainStatus
journalctl --user -u paintings-slic-production.service -n 20 --no-pager
```

Os quatro parquets foram validados quanto a schema, finitude, unicidade e
alinhamento de stems. O extrator suporta retomada a partir de checkpoints.

## Execução padrão

```bash
source .venv-rtx3090-py312/bin/activate
make all
```

`make all` e o alias `make features` executam apenas
`src/pipeline/slic_multigraph_features.py`. O pipeline de produção não lê
arquivos `.coarse`, epsilon estrela nem contagens do método anterior.

O ambiente desta máquina é `.venv-rtx3090-py312`; não usar nem recriar `.venv`,
que pertence ao ambiente usado na outra máquina.

## Dataset oficial

```text
wikiart/                         imagens brutas, em 27 diretórios de estilo
data/artemis/wikiart_meta.csv    metadados oficiais
outputs/wikiart_slic_features/   novas features dos grafos SLIC
outputs/wikiart_features/        baselines e resultados legados
```

- 81.444 imagens físicas; 79.936 registros classificados.
- Depois de resolver 695 diferenças de codificação nos nomes e remover stems
  duplicados conflitantes, o dataset de features contém 79.902 imagens.
- `wikiart/` e ambientes `.venv-*` estão no `.gitignore`.

## Código principal

- `src/pipeline/slic_multigraph_features.py`: orquestra a extração completa;
- `src/pipeline/slic_graph_features.py`: segmentação SLIC e piloto pareado;
- `src/pipeline/graph_feature_utils.py`: grafos e agregados independentes da
  segmentação;
- `experiments/compare_slic_rag.py`: comparação pareada usada na decisão;
- `Makefile`: SLIC é o alvo padrão.

Os scripts históricos da percolação permanecem apenas como legado para
reprodutibilidade dos resultados anteriores. Alguns têm alterações locais e
dependências entre si; não são chamados pelo pipeline ativo e não devem ser
apagados isoladamente.

## Treino CNN

O treino CNN citado no handoff anterior está rodando em outra máquina. Não há
serviço CNN a administrar neste host.

## Atenção ao worktree

O repositório já estava sujo, com várias mudanças e arquivos não rastreados do
paper. Preservar essas alterações. O Git também reporta `D CLAUDE.md`; não
restaurar nem descartar sem confirmação do usuário.
