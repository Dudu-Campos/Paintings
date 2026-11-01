# Mapa de evidências e congelamento dos resultados

Data do congelamento inicial: 12 de agosto de 2026.

Este documento responde a quatro perguntas: qual é a contribuição central,
quais números podem sustentá-la, quais resultados pertencem apenas ao
apêndice e o que ainda precisa ser fechado antes da submissão.

## 1. Tese do artigo

Representações estruturais explícitas de uma pintura podem ser competitivas
sem pré-treinamento visual e, ao mesmo tempo, revelar de onde vem o sinal de
estilo. Três fontes de diversidade são úteis:

1. **escala espacial**, por meio da imagem global e de quatro quadrantes fixos;
2. **ordem estrutural**, por meio do grafo de linha do RAG;
3. **particionamento da imagem**, combinando SLIC e Felzenszwalb.

O artigo não precisa afirmar que uma concatenação de todos os módulos é um novo
estado da arte. A contribuição é mostrar, com controles pareados, que cada
visão acrescenta informação e que existe um ponto de operação competitivo e
barato sem ImageNet ou WikiArt supervisionado.

## 2. Protocolo canônico

- Problema: classificação de 27 estilos do WikiArt.
- Coorte visual principal: 79.207 imagens e stems únicos.
- Identificador da coorte: `results/line_graph_full_sample.csv`.
- Métrica primária: balanced accuracy.
- Avaliação: cinco folds estratificados com `random_state=42`.
- MLP: 10% do treino externo reservado para validação e early stopping;
  o fold de teste externo permanece intocado.
- Incerteza no manuscrito: desvio-padrão amostral entre folds.
- Testes de módulos: teste t pareado sobre os cinco folds, acompanhado da
  diferença em cada fold e de um placebo com features permutadas entre imagens.

Os resultados de 79.936 imagens das CNNs e os de 79.902 stems da GNN devem
aparecer com a coorte visível. Eles são controles importantes, mas não testes
pareados com a coorte de 79.207.

## 3. Representações

### 3.1 Vetor visual de 258 dimensões

O vetor visual canônico não contém texto:

- SLIC multigrafo: 71 dimensões;
- descritores clássicos: 130 dimensões;
- série espacial de Hilbert: 57 dimensões.

O SLIC usa até 2.048 segmentos, compactness 10, sigma 0 e lado máximo 1.024.
Cada região contém Lab médio, centroide normalizado e área. Uma única
segmentação alimenta quatro grafos: RAG, kNN em Lab+XY, Delaunay dos centroides
e MST do RAG ponderado por distância de cor. As 71 dimensões preservam 32
estatísticas completas do RAG e 13 estatísticas de arestas para cada um dos
outros três grafos.

Os 130 descritores reúnem histogramas Lab (90), estatísticas Lab (9), LBP
uniforme (18) e Haralick/GLCM (13). A representação Hilbert percorre a imagem
Lab de 128 x 128 em ordem de Hilbert e resume oito bandas, estatísticas globais
e correlações entre canais.

### 3.2 Escala espacial

A extensão multiescala reaplica os mesmos três blocos à imagem global e a
quatro quadrantes com posição fixa. O resultado tem 1.290 dimensões e mantém as
15 modalidades como ramos separados da MLP.

### 3.3 Grafo de linha

Cada aresta do RAG SLIC torna-se um nó de `L(G)`. Duas arestas de `L(G)` são
adjacentes quando as fronteiras originais compartilham uma região. As 63
features resumem topologia, contraste, distribuição espacial, ângulos em
junções, continuidade de bordas fortes, distância Lab entre regiões externas,
reversões de luminância e desigualdade de áreas/graus.

### 3.4 Felzenszwalb

A segunda segmentação usa `scale=50`, `sigma=0.8`, `min_size=50` e lado máximo
1.024. As regiões resultantes passam exatamente pelos mesmos construtores RAG,
kNN, Delaunay e MST, gerando outras 71 dimensões.

## 4. Evidência central

### 4.1 Ponto de operação visual sem pré-treinamento

| Método | BAcc | Coorte | Papel |
|---|---:|---:|---|
| Haralick + HOG + MLP | 22,68% ± 0,70 | 79.207 | baseline clássico de imagem |
| Random Forest sobre Visual-258 | 28,67% ± 0,56 | 79.207 | controle de classificador |
| kNN sobre Visual-258 | 28,09% ± 0,54 | 79.207 | controle de classificador |
| ResNet-50 do zero | 42,34% ± 2,48 | 79.936 | deep baseline sem pré-treino |
| SVM-RBF sobre Visual-258 | 43,18% ± 0,50 | 79.207 | melhor controle clássico |
| Visual-258 + MLP por ramos | **45,22% ± 1,10** | 79.207 | método visual global |
| Visual multiescala + MLP por ramos | **48,33% ± 1,28** | 79.207 | melhor visual confirmado |

Os desvios acima são amostrais. O ganho sobre a ResNet do zero é evidência de
eficiência estatística, mas não é um teste pareado porque a coorte da CNN tem
729 imagens a mais.

### 4.2 Relações de segunda ordem

No experimento pareado da MLP plana:

- SLIC-71: `38,49% -> 42,03%` com grafo de linha, ganho de `+3,54 pp`,
  `p=0,000220`, positivo nos cinco folds;
- Visual-258: `45,36% -> 46,60%`, ganho de `+1,24 pp`;
- neste segundo contraste, `p=0,0975` contra o baseline real, mas `p=0,00329`
  e `+2,83 pp` contra as 63 colunas permutadas.

O mesmo módulo melhora o vetor visual em LDA (`+1,58 pp`, `p=0,00236`) e
XGBoost (`+0,70 pp`, `p=0,04795`). Quarenta e seis das 63 features mantêm
`r²<0,90` contra SLIC-71, e a versão não redundante preserva o ganho.

Fontes: `results/line_graph_full_results.csv`,
`results/line_graph_full_results_comparisons.csv` e
`results/line_graph_full_redundancy.csv`.

### 4.3 Complementaridade de segmentação

Felzenszwalb isolado é praticamente equivalente ao SLIC no XGBoost
(`34,84%` contra `34,79%`), mas sua combinação é melhor:

- LDA, Visual-258 + line: `38,92% -> 40,94%`, ganho residual de `+2,02 pp`,
  `p=0,000258`;
- XGBoost, Visual-258 + line: `40,35% -> 41,26%`, ganho residual de
  `+0,91 pp`, `p=0,02522`;
- XGBoost contra o placebo permutado: `+1,29 pp`, `p=0,00428`.

Sessenta e duas das 71 features Felzenszwalb mantêm `r²<0,90` mesmo contra
Visual-258 + line. Isso sustenta a interpretação de complementaridade entre
particionamentos, e não apenas aumento de dimensionalidade.

O resultado MLP desta seção permanece pendente. Não extrapolar a partir de LDA
ou XGBoost.

### 4.4 Auditabilidade

Entre as 258 features visuais, a mediana de `r²` par a par é `0,00477`; apenas
24 pares atingem `r²>=0,99`. Há duas constantes e cópias quase determinísticas
de algumas estatísticas Lab, mas a maior parte do vetor não é duplicada.

Massa por representação:

| Representação | Estrutura LDA | Ganho XGBoost |
|---|---:|---:|
| Descritores | 41,21% | 43,36% |
| SLIC multigrafo | 36,20% | 36,56% |
| Hilbert | 22,59% | 20,08% |

Os sinais dominantes são textura Haralick/LBP, topologia MST/kNN e dispersão
de cor. Isso deve virar uma figura integrada, não duas listas extensas.

## 5. Baselines pré-treinados e limites da alegação

| Método | BAcc | Pré-treino | Coorte |
|---|---:|---|---:|
| VGG-19 conv5 GAP + MLP | 45,83% ± 1,35 | ImageNet | 79.207 |
| VGG-19 final GAP + MLP | 49,18% ± 0,88 | ImageNet | 79.207 |
| Gram compacto VGG-19 + MLP | 51,48% ± 0,88 | ImageNet | 79.207 |
| ResNet-50 frozen + cabeça linear | 55,46% ± 1,53 | ImageNet | 79.936 |
| ResNet-50 fine-tuned | **62,72% ± 1,29** | ImageNet | 79.936 |

Portanto, a alegação correta não é “superamos redes pré-treinadas”. A alegação
é “oferecemos uma alternativa interpretável e competitiva quando o
pré-treinamento não é usado ou quando custo e auditabilidade importam”.

## 6. Texto e risco de leakage

### 6.1 CATR/COCO

O gerador CATR público foi treinado em legendas humanas do MS COCO, sem Qwen e
sem fine-tuning no WikiArt. Em 79.207 imagens, a legenda isolada obtém
`19,72%` OOF. Ao ser acrescentada ao Numeric-274:

- LDA: `38,02% -> 39,65%` OOF (`+1,63 pp`);
- XGBoost: `39,70% -> 41,28%` OOF (`+1,58 pp`);
- BranchFusionMLP: `45,81% -> 46,60%` OOF (`+0,79 pp`).

A auditoria em 12.210 legendas encontrou zero menções a estilos ou artistas e
três coincidências exatas de título (`0,0246%`). Contudo, Numeric-274 inclui 16
features de sentimento provenientes do captioner antigo. Assim, o **ganho
incremental CATR** é defensável, mas o valor absoluto não representa um
pipeline totalmente livre da linhagem textual antiga.

Decisão editorial: CATR é uma extensão secundária. Uma futura comparação
Visual-258 versus Visual-258 + CATR resolveria a ressalva, mas não bloqueia o
artigo visual.

### 6.2 ArtEmis/Qwen

Os resultados `48,84%` (visual + TF-IDF) e `50,90%` (multiescala + TF-IDF)
usam um captioner com linhagem Qwen e exposição ao ArtEmis/WikiArt. A ausência
de menções literais de estilo reduz, mas não elimina, o risco de contaminação.
Esses valores ficam no apêndice como estudo exploratório.

Qwen direto é ainda menos adequado como headline: na amostra de 12.210, a
descrição neutra menciona algum estilo em `9,10%` e o estilo correto em
`3,90%`. O ramo emocional apresenta `7,65%` e `3,07%`, respectivamente.

## 7. Resultados negativos e por que são úteis

| Linha | Resultado resumido | Decisão |
|---|---|---|
| GNN sobre RAG + SLIC-71 | 42,15% ± 2,27; 11,43 h | controle profundo estruturado; não supera Visual-258 |
| Grafo quadrado `G²` | satura após o line graph; XGB cai 35,08% -> 34,53% no piloto | apêndice; ordem adicional não compensa |
| Real-ESRGAN | melhor ganho sobre original: +0,86 pp no XGB; `p=0,116`; ~30 h projetadas | apêndice; caro e inconclusivo |
| Big-LaMa | ganho adicional sobre canvas-placebo abaixo de 1 pp | apêndice; o canvas explica grande parte do efeito |
| Branch fusion | 45,85% contra 45,78% da MLP plana | arquitetura de fusão não é a fonte principal do ganho |
| OOF stacking | 38,12% contra 38,14% da concatenação LDA | concatenação simples é suficiente |
| Captioner emocional CATR + ArtEmis | EOS corrigido; linguagem ainda degradada/alucinada | resultado negativo de geração; não usar no pipeline |

Esses experimentos demonstram seleção de hipótese e controles, mas reuni-los
no corpo principal diluiria a tese. Uma tabela compacta no apêndice é
suficiente.

## 8. Resultados legados excluídos

Não usar para números finais:

- `results/results_slic_multigraph_full.csv`: 75.500 imagens;
- `results/forward_selection_3clf.csv`: 74.954 imagens;
- `results/forward_selection_results.csv`: protocolo anterior;
- `results/ablation_*`: dimensões/protocolos anteriores;
- `results/results_lstm_hilbert.csv`: linha histórica;
- pilotos de segmentação com 2.693 imagens quando o resultado Felzenszwalb
  completo estiver disponível;
- pilotos de 400 imagens de super-resolução e Big-LaMa, supersedidos pelas
  validações maiores.

`results/metodos_final.csv` permanece útil como registro histórico, mas não é
mais a tabela canônica do artigo.

## 9. Pendências antes da submissão

### Obrigatórias

1. Incorporar a MLP Felzenszwalb completa e decidir se ela muda o método
   visual recomendado.
2. Fazer split desjunto por artista. Há 1.029 artistas cruzando partições e
   muitos artistas associados a um único estilo; um revisor provavelmente
   questionará aprendizado de identidade do artista.
3. Auditar duplicatas ou quase duplicatas visuais entre folds.
4. Repetir a medição de custo em condições fixas e registrar CPU, GPU, threads,
   armazenamento e o que é contado como extração versus treino.
5. Congelar dependências e proveniência/licença do WikiArt.

### Editoriais

1. Escolher venue, limite de páginas e formato de referências.
2. Escolher nome definitivo. “Visual GIST” deve ser abandonado porque pode ser
   confundido com o descritor GIST clássico.
3. Definir se a multiescala é apresentada como método principal de maior
   acurácia ou como ponto de operação mais caro.
4. Produzir figuras vetoriais do pipeline, dos ganhos de módulos e do Pareto.

### Não obrigatórias

- concatenar multiescala + line graph + Felzenszwalb;
- ampliar Real-ESRGAN, Big-LaMa, `G²` ou Qwen;
- desenvolver uma nova arquitetura de fusão;
- treinar outro captioner emocional.

Essas linhas só devem ser reabertas se o resultado da MLP pendente ou uma
exigência de revisão mudar a pergunta científica.

