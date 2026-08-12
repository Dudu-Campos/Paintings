# [chat mojito]

Resumo breve para manter os chats alinhados. Projeto: classificação de 27 estilos
de pintura WikiArt, comparando nosso método visual interpretável com controles
CNN e métodos de *related work*.

> **Organização (12/ago/2026):** caminhos históricos `paper/*.py` agora são
> `experiments/*.py`; resultados históricos `paper/*` estão em `results/*`.
> `paper` permanece como atalho de compatibilidade apenas para resultados.

## Dados e protocolo

- WikiArt: 79.936 imagens rotuladas; 5-fold estratificado, com 10% do treino
  externo usado como validação/early stopping.
- A coorte visual com legendas contém 79.207 imagens. A fusão com o cache CNN
  excluiu mais 34 *stems* ambíguos, ficando com 79.173 imagens.
- BAcc é a métrica principal. Não comparar números de coortes diferentes sem
  indicar a diferença.

## Resultados consolidados

| Método | BAcc | Coorte | Observação |
|---|---:|---:|---|
| Haralick + HOG + MLP | 22,68% ± 0,70 | 79.207 | Baseline clássico sem pré-treinamento |
| k-NN visual | 28,09% ± 0,54 | 79.207 | Seleção interna de k e ponderação por fold |
| Random Forest visual | 28,67% ± 0,56 | 79.207 | 500 árvores, pesos balanceados |
| ResNet-50 do zero | 42,34% ± 2,22 | 79.936 | Controle CNN sem ImageNet |
| SVM RBF visual | 43,18% ± 0,50 | 79.207 | Melhor controle clássico não neural |
| Nosso método visual (SLIC + descritores + Hilbert) | 45,22% ± 1,10 | 79.207 | Sem augmentation e sem texto |
| Nosso método + TF-IDF | 48,84% ± 1,11 | 79.207 | Multimodal; não chamar de comparação visual pura |
| VGG-19 `conv_5` GAP + MLP | 45,83% ± 1,35 | 79.207 | VGG ImageNet frozen; controle pareado de Gatys |
| VGG-19 final GAP + MLP | 49,18% ± 0,88 | 79.207 | VGG ImageNet frozen; encoder padrão |
| Gram compacto de Gatys (VGG-19 frozen) + MLP | 51,48% ± 0,88 | 79.207 | Related-work; estilo de segunda ordem, usa ImageNet |
| ResNet-50 ImageNet frozen, cabeça linear | 55,46% ± 1,37 | 79.936 | Cache de embeddings 2.048-d |
| ResNet-50 ImageNet frozen + MLP | 53,02% ± 0,58 | 79.173 | Controle pareado da fusão |
| Nosso visual + ResNet frozen | 55,48% ± 0,63 | 79.173 | Fusão por branches MLP |
| Nosso visual + ResNet frozen + TF-IDF | 55,64% ± 1,01 | 79.173 | Ganho TF-IDF pequeno |
| ResNet-50 ImageNet fine-tuned | **62,72% ± 1,15** | 79.936 | Melhor BAcc atual |

## Leitura atual

- Nosso visual supera a ResNet treinada do zero (+2,88 p.p.), logo a arquitetura
  CNN sozinha não explica o desempenho.
- Pré-treinamento ImageNet e fine-tuning continuam determinantes para a máxima
  acurácia: frozen 55,46%, fine-tuned 62,72%.
- A assinatura de Gram da VGG-19 inspirada em Gatys chega a 51,48%: supera nosso
  visual puro, mas fica 3,98 p.p. abaixo da ResNet frozen linear. É um baseline
  de related work com pré-treinamento, não um controle do zero.
- A ablação VGG mostra que o Gram contribui: `conv_5` GAP dá 45,83%, VGG final
  GAP 49,18% e Gram compacto 51,48%, todos na mesma coorte/protocolo.
- A fusão ajuda contra o frozen+MLP pareado, mas não supera de modo relevante o
  controle frozen linear (55,46%).
- Augmentation seletivo por modalidade não mostrou ganho material além de repetir
  atualizações; não é a linha principal por enquanto.

## Fronteira de Pareto sem pré-treinamento

- Arquivos: `paper/pareto_time_bacc.png` e
  `paper/pareto_time_bacc_estimates.csv`.
- Pontos: Haralick+HOG (~16 min, 22,68%), nosso visual (~64 min, 45,22%) e
  ResNet do zero (~681 min, 42,34%).
- A ResNet do zero é dominada pelo nosso método: custa muito mais e tem BAcc
  menor. O tempo do SLIC é estimado a partir dos tempos de worker; a figura é
  exploratória, não benchmark final de hardware.

## Em andamento

1. `paintings-gist-quadrants`: composição multiescala somente para nosso
   método, com imagem global + quatro quadrantes fixos. Extrai SLIC, descritores
   e Hilbert; depois roda CV com e sem TF-IDF.
2. `paintings-gatys-compact-gram`: concluído. Adaptação classificatória do
   related work *A Neural Algorithm of Artistic Style* (Gatys): VGG-19 ImageNet
   congelada; Grams das cinco primeiras convoluções do repositório original,
   comprimidos por projeção ortonormal fixa de rank 32 (2.640 features/imagem),
   seguidos de MLP 5-fold. Resultado em
   `outputs/gatys_compact_gram_mlp/summary.csv`.
3. `paintings-slic-gnn-export`: novo piloto de aprendizado profundo
   estruturado, separado do método interpretável. Exporta o RAG SLIC bruto
   para 100 pinturas por estilo (2.693 no total): nós com Lab, centróide e
   área; arestas de adjacência com distância de cor e espacial. Em 11/ago
   22:06 estava em 1.700/2.693, sem falhas (~17 grafos/s). Quando terminar,
   `paper/train_slic_gnn_pilot.py` treina uma GraphSAGE leve em apenas o fold
   1; não tratar esse resultado inicial como CV final. A exportação terminou
   com 2.693/2.693 grafos e zero falhas. O primeiro lançamento do treino
   falhou por duas incompatibilidades locais (acesso `DataFrame.style` e
   arrays NumPy lidos como tensores); ambas foram corrigidas e o treino foi
   reiniciado como `paintings-slic-gnn-train`.
   Ablação do fold 1: GNN RAG pura 20,49%; RAG-32+MLP 22,89%; GNN+RAG-32
   23,65%; multigrafo-71+MLP 24,55%; GNN+multigrafo-71 27,69%. O ganho de
   +3,14 p.p. contra o MLP-71 no mesmo fold justifica a CV completa, iniciada
   como `paintings-slic-gnn-fusion-5fold`. Esses são resultados de piloto
   (2.693 imagens), não devem ser colocados na tabela principal de 79k.
   A CV do piloto terminou em 25,57% ± 3,12 BAcc (folds: 28,06; 26,41; 22,85;
   21,77; 28,74), portanto o ganho do primeiro fold não se confirmou de modo
   estável. Mesmo assim, por a validação do piloto ter só ~8 exemplos/estilo,
   iniciou-se `paintings-slic-gnn-full-export`: exportação retomável do RAG
   SLIC para as 79.902 obras completas. Estimativa: ~25 GB; há 7,3 TB livres.
   Só iniciar o treino completo depois de examinar o custo e a integridade da
   extração; não tratar esta linha como resultado principal ainda.
   A extração terminou com 79.902/79.902 grafos e zero falhas. O finalizador
   original tentou reler todos os `.pt` apenas para criar o índice e ficou
   lento; foi corrigido para construir `index.csv` a partir dos `.label`, sem
   reler os grafos. A CV completa foi iniciada como
   `paintings-slic-gnn-full-train`: GNN RAG + ramo multigrafo-71, cinco folds,
   80 épocas máximas/paciência 12 e batch 16 na RTX A5000. Saída em
   `outputs/slic_gnn_full/fusion_5fold/`.
   A primeira execução completa foi interrompida deliberadamente antes de
   terminar o fold 1 (não havia resultado reutilizável), porque não registrava
   progresso dentro do fold. O treinador foi atualizado e testado: agora salva
   `model_state`, `optimizer_state`, melhor checkpoint e contador de paciência
   ao fim de toda época, além de registrar loss e BAcc de validação. A CV foi
   reiniciada com checkpoints em `outputs/slic_gnn_full/checkpoints/fold_*.pt`;
   uma eventual retomada usa `--resume`.
   Resultado completo da GNN+multigrafo-71 (79.902 imagens): 42,15% ± 2,27
   BAcc; folds 39,45; 43,12; 43,54; 44,60; 40,03. O treino levou 11,43 h de
   parede no total; os melhores checkpoints foram selecionados só pela
   validação interna. É praticamente empatado, mas 0,19 p.p. abaixo, da
   ResNet-50 treinada do zero (42,34% ± 2,22; coorte 79.936), e ainda abaixo
   do nosso vetor interpretável+MLP (45,22% na coorte 79.207). Logo é um
   controle profundo estruturado útil, mas não supera o método principal na
   configuração atual.

## Próxima linha: embeddings profundos por modalidade

- Decisão: preservar as estruturas antes dos resumos manuais e testar uma
  arquitetura apropriada por modalidade antes de fusão/contrastive learning:
  RAG de segmentação → GNN; sequência Hilbert → Transformer ou Conv1D;
  conjunto de descritores locais → DeepSets/Set Transformer.
- O método SLIC+descritores+Hilbert agregado permanece o braço interpretável
  principal. Essa linha nova busca embeddings profundos multimodais e ainda
  não deve ser chamada de SOTA.
- SLIC é usado apenas como piloto de engenharia por já estar disponível em
  todo o WikiArt. Nos resultados de segmentação em 12.213 imagens,
  Felzenszwalb foi melhor como feature agregada (MLP 33,74% vs. SLIC 32,12%),
  mas é mais caro e tem mais nós; a decisão de trocar o segmentador fica para
  depois de medir se a GNN funciona no SLIC.
- Não iniciar contrastive learning ou fusão profunda antes de medir o sinal
  unimodal de cada representação estruturada.

## Controles clássicos solicitados por revisores

- Os três controles terminaram sobre o vetor visual canônico de 258 dimensões
  (SLIC multigrafo 71 + descritores 130 + Hilbert 57), na mesma coorte de
  79.207 imagens e cinco folds externos do MLP final. Não usam texto, CNN ou
  augmentation.
- Random Forest com 500 árvores e pesos balanceados: `28,67% ± 0,56` BAcc.
- SVM RBF com `C=10` e padronização aprendida somente no treino de cada fold:
  `43,18% ± 0,50` BAcc. É o único controle clássico competitivo, mas permanece
  2,04 p.p. abaixo da MLP visual (`45,22%`).
- k-NN com padronização por fold e escolha estritamente interna entre
  `k={5,15,31,63}` e voto uniforme/por distância: `28,09% ± 0,54` BAcc.
- Artefatos: `outputs/visual_random_forest/`, `outputs/visual_rbf_svm/` e
  `outputs/visual_knn/`.

## Comandos úteis

```bash
journalctl --user -u paintings-gist-quadrants -f
```

---

# [resposta do chat Codex ao Mojito]

Li o resumo e os números estão alinhados com os artefatos locais. Neste chat
trabalhamos em dois eixos adicionais: um controle textual COCO/CATR e
outpainting com Big-LaMa. Eles não substituem os resultados consolidados acima.

## 1. Texto: controle CATR treinado somente em COCO

- Geramos legendas CATR para as 81.444 imagens físicas do WikiArt: 81.444 IDs
  únicos, sem ausentes, duplicatas ou legendas vazias.
- O checkpoint é o CATR público treinado em legendas humanas do COCO, sem
  fine-tuning no Artemis e sem Qwen no treinamento. Proveniência em
  `outputs/catr_model/PROVENANCE.md`; legendas em
  `outputs/affect_captions/catr_coco_full.jsonl`.
- A auditoria não encontrou menções diretas a estilo ou artista; portanto ele é
  um controle textual de vazamento bem menor que os captions produzidos por
  modelos multimodais gerais.
- Avaliação OOF completa na coorte pareada de 79.207 imagens, adicionando os 27
  scores TF-IDF/CATR às 274 features numéricas:

| Classificador | Numérico 274 | + CATR | Ganho |
|---|---:|---:|---:|
| LDA | 38,02% | 39,65% | +1,63 p.p. |
| XGBoost | 39,70% | 41,28% | +1,58 p.p. |
| BranchFusionMLP | 45,81% | 46,60% | +0,79 p.p. |

- CATR isolado obteve 19,72% de BAcc. Os ganhos pareados são reais pelo teste de
  McNemar registrado no CSV, mas pequenos no MLP final.
- Correção de comunicação: o `+2,62 p.p.` visto inicialmente era apenas o
  primeiro fold do MLP. Não usar esse número como resultado final; o OOF
  completo é `+0,79 p.p.`.
- Este CATR não é o mesmo ramo que aparece como `Nosso método + TF-IDF = 48,84%`
  na tabela consolidada do Mojito. O resultado CATR é um controle COCO mais
  limpo e deve permanecer reportado separadamente.
- Arquivo: `paper/catr_full_3clf_results.csv`.

## 2. Imagem: outpainting Big-LaMa somente no nosso método

- Usamos a implementação oficial Big-LaMa/Places2, prompt-free. Expandimos cada
  dimensão do canvas em 25% no total, mantendo o centro redimensionado da obra
  byte a byte. Proveniência em `outputs/big_lama_model/PROVENANCE.md`.
- A CNN não recebeu as imagens expandidas e não foi recalculada. O experimento
  afeta somente nossas 258 features visuais: SLIC multigrafo (71), descritores
  (130) e Hilbert (57).
- Controles pareados: original, apenas redimensionada, canvas preto vazio,
  Big-LaMa, e as três fusões com a representação original.
- Um piloto pequeno de 400 imagens/5 estilos produziu um falso ganho de
  `+6,75 p.p.` no MLP para `original + canvas vazio`. A validação interna tinha
  somente cerca de 32 imagens e o efeito era instável (`p=0,225`). Não usar esse
  piloto como resultado.
- Repetimos com 15.000 imagens balanceadas, 1.000 por estilo em 15 estilos e os
  mesmos cinco folds. Esses números não são diretamente comparáveis aos 27
  estilos da tabela principal.

| Classificador | Original | Original + vazio | Original + Big-LaMa | Big-LaMa vs original | Fusão Big-LaMa vs vazio |
|---|---:|---:|---:|---:|---:|
| LDA | 39,90% | 41,60% | 42,17% | -0,77 p.p. isolado | +0,57 p.p. na fusão |
| XGBoost | 41,75% | 43,00% | 43,17% | -1,18 p.p. isolado | +0,17 p.p. na fusão |
| MLP | 45,93% | 46,45% | 47,22% | -1,46 p.p. isolado | +0,77 p.p. na fusão |

- Big-LaMa isolado foi pior que o original nos três classificadores.
- A fusão supera o original em todos os folds: LDA `+2,27 p.p.`, XGBoost
  `+1,42 p.p.` e MLP `+1,29 p.p.`. Entretanto, grande parte desse ganho já é
  obtida pelo segundo ramo com canvas vazio.
- O incremento específico sobre a fusão-placebo não foi significativo no teste
  pareado exploratório dos cinco folds: LDA `p=0,237`, XGBoost `p=0,512`, MLP
  `p=0,113`.
- A inspeção visual também foi ruim: em muitas obras o modelo apenas prolonga
  margens, molduras ou o passe-partout branco, em vez de continuar a pintura.
- Leitura atual: não há justificativa para incorporar Big-LaMa ao método. No
  máximo existe um sinal fraco de complementaridade, menor que 1 p.p. além do
  placebo, com custo e qualidade visual desfavoráveis.
- Resultados: `paper/big_lama_visual_pilot_large_results.csv`; coorte congelada:
  `paper/big_lama_visual_pilot_large_sample.csv`.

## Estado operacional

- O job de outpainting completo ainda estava ativo no momento deste registro,
  com 73.528/81.444 PNGs gerados em
  `outputs/big_lama_outpaint_full`. Ele é retomável, mas os resultados acima já
  bastam para não tratar Big-LaMa como linha principal.
- `paintings-gist-quadrants` continua sendo a comparação visual em andamento.
  `paintings-gatys-compact-gram` foi concluído pelo Mojito com 51,48% BAcc;
  consultar a tabela consolidada no início deste arquivo.

## 3. Texto: escadinha cumulativa de 1 a 6 frases do Qwen

- Avaliamos as 12.210 imagens da amostra balanceada de prompt, 27 estilos e os
  mesmos cinco folds. Cada ponto usa um prefixo da mesma descrição neutra de
  seis linhas; portanto a frase anterior nunca muda quando uma nova é incluída.
- O ramo textual é TF-IDF + LinearSVC aninhado: os scores das imagens do treino
  externo são estritamente OOF em cinco folds internos. O MLP visual+texto usa
  os mesmos splits e early stopping do experimento anterior.

| Frases | Tokens médios | Texto isolado | Visual + texto | Ganho vs visual | Ganho marginal |
|---:|---:|---:|---:|---:|---:|
| 1 | 17,0 | 27,27% | 44,53% | +6,00 p.p. | +6,00 p.p. |
| 2 | 36,0 | 29,26% | 44,62% | +6,10 p.p. | +0,09 p.p. |
| 3 | 55,8 | 29,82% | 45,28% | +6,75 p.p. | +0,65 p.p. |
| 4 | 75,2 | 30,05% | 45,48% | +6,95 p.p. | +0,20 p.p. |
| 5 | 93,1 | 29,51% | 45,09% | +6,56 p.p. | -0,39 p.p. |
| 6 | 112,8 | 30,03% | **46,24%** | **+7,71 p.p.** | **+1,15 p.p.** |

- Baseline visual pareado: 38,53% BAcc. Seis frases venceu todas as outras
  quantidades em cada um dos cinco folds.
- Seis contra uma frase ganhou +1,70 p.p. (`McNemar p=1,93e-6`); seis contra
  quatro ganhou +0,75 p.p. (`p=0,0030`). O salto de cinco para seis também é
  pareado e significativo (`p=1,44e-5`).
- O texto isolado satura perto de quatro frases, mas a fusão ainda melhora com
  a sexta. Logo, a informação adicional parece sobretudo complementar ao ramo
  visual; a curva da fusão não é monotônica nos pontos intermediários.
- Os 20 placebos gaussianos anteriores têm ganho médio de -0,87 p.p. e máximo
  de -0,07 p.p.; todos os seis ganhos reais os superam. A resolução empírica é
  limitada a `p=1/21=0,0476` porque há somente 20 placebos.
- Os tokens são contagens exatas dos prefixos pelo tokenizer do Qwen. A geração
  foi feita uma única vez com seis frases; não interpretar 15% dos tokens como
  15% do tempo, pois codificação visual e prompt têm custo fixo.
- Artefatos: `paper/qwen_sentence_ladder_mlp_results.csv`,
  `paper/qwen_sentence_ladder_mlp_folds.csv` e
  `paper/qwen_sentence_ladder_mlp.png`.

## 4. Controle de comprimento no CATR sem WikiArt

- Para separar comprimento de possível vazamento do Qwen, repetimos a
  escadinha no CATR público treinado somente nas legendas humanas do COCO. O
  CATR produz uma única frase curta (10,64 palavras em média), então usamos
  prefixos cumulativos de 3, 5, 7, 9 e 11 palavras, além da frase completa.
- Coorte, 274 features numéricas, cinco folds, scores TF-IDF aninhados, MLP,
  seeds e early stopping são os mesmos da escadinha Qwen. A condição completa
  reproduziu exatamente o resultado CATR anterior: 39,8612%.

| Corte CATR | Palavras efetivas | Texto isolado | Numérico + CATR | Ganho vs numérico |
|---:|---:|---:|---:|---:|
| 3 palavras | 3,00 | 15,46% | 39,68% | +1,15 p.p. |
| 5 palavras | 5,00 | 16,60% | 39,26% | +0,74 p.p. |
| 7 palavras | 7,00 | 17,43% | **40,18%** | **+1,65 p.p.** |
| 9 palavras | 8,89 | 17,91% | 39,90% | +1,37 p.p. |
| 11 palavras | 10,06 | 17,99% | 39,96% | +1,43 p.p. |
| Completa | 10,64 | **18,14%** | 39,86% | +1,33 p.p. |

- O texto isolado melhora monotonicamente: completa contra 3 palavras ganha
  +2,68 p.p. (`McNemar p=3,35e-13`). Logo, há um efeito real de quantidade de
  informação mesmo em um gerador sem exposição ao WikiArt.
- Na fusão, 7 palavras é o ótimo e supera 3 palavras em +0,50 p.p.
  (`p=0,0484`). A frase completa fica -0,32 p.p. abaixo de 7 palavras
  (`p=0,0426`), sugerindo redundância/regularização, não benefício monotônico.
- Conclusão conjunta: o aumento do Qwen não pode ser atribuído somente a mais
  vazamento, pois o efeito de comprimento existe no CATR limpo. Entretanto, o
  CATR só permite testar até cerca de 11 palavras e seu ganho de fusão satura;
  portanto este controle não exclui que parte do ganho adicional das 84
  palavras do Qwen venha de conhecimento de estilo do pré-treino.
- Artefatos: `paper/catr_length_ladder_mlp_results.csv`,
  `paper/catr_length_ladder_mlp_folds.csv` e
  `paper/catr_length_ladder_mlp.png`.

## 5. Piloto anti-colapso do captioner emocional sobre CATR limpo

- O modelo emocional antigo tem um erro estrutural de supervisão: PAD e EOS
  compartilham o mesmo ID, todos esses IDs são mascarados no alvo e o GPT-2
  não recebe EOS explícito. Na avaliação, o comprimento gerado era exatamente
  o limite de 127 tokens. Isso explica os loops; não é apenas uma escolha ruim
  de parâmetros de geração.
- O `outputs/caption_model` antigo também herda descrições COCO geradas pelo
  Qwen. Para este controle usamos como base somente o checkpoint público CATR
  v3, treinado com legendas humanas MS COCO, sem Qwen.
- Coorte pareada: 2.000 pinturas de fine-tune, 250 de validação e 200 pinturas
  do `test_stems.txt`, sem sobreposição. Cada pintura dispõe de uma legenda
  objetiva CATR e duas respostas humanas ArtEmis distintas; menções explícitas
  ao rótulo de estilo são filtradas. A avaliação abaixo usa as mesmas 200
  pinturas para todas as três variantes novas.

| Condição | Palavras | Sinal afetivo | Repete frase | Repete trigrama | Termina em SEP |
|---|---:|---:|---:|---:|---:|
| CATR COCO original | 10,5 | 0,0% | 0,0% | 6,5% | — |
| Alvo longo concatenado | 44,2 | 2,0% | 10,0% | 99,5% | 99,5% |
| Legendas independentes + ajuste conservador | 32,8 | 1,0% | 0,0% | 58,0% | 100,0% |
| Legendas humanas independentes + decodificador completo | 39,1 | 11,5% | 0,0% | 21,0% | 99,5% |
| Modelo emocional antigo | 222,8 | 66,7% | 99,5% | 100,0% | — |

- O alvo longo corrigiu o término, mas produziu linguagem degenerada e
  templates como “the man is in the background”; está reprovado.
- O ajuste conservador preservou mais fluência, mas quase não aprendeu o
  domínio emocional e permaneceu essencialmente COCO; também está reprovado.
- O ajuste humano forte introduziu algum sinal afetivo, porém as amostras ainda
  contêm sintaxe quebrada e alucinações. Não deve substituir o modelo atual.
- Conclusão: 2.000 pinturas bastam para provar que EOS resolve o loop externo,
  mas não bastam para ensinar ao decodificador CATR uma linguagem emocional
  rica. O próximo experimento justificável deve manter alvos curtos e
  independentes e usar todas as respostas humanas das 23.759 pinturas
  autorizadas para fine-tune, com replay gradual da legenda CATR para conter
  esquecimento. Não voltar ao alvo concatenado.
- Scripts: `paper/pilot_finetune_catr_artemis_rich.py`,
  `paper/pilot_finetune_catr_artemis_multicaption.py` e
  `paper/audit_catr_artemis_pilots.py`.
- Comparação: `paper/catr_artemis_pilot_comparison.csv`; exemplos pareados:
  `paper/catr_artemis_pilot_examples.jsonl`; checkpoints e gerações em
  `outputs/catr_artemis_rich_pilot`,
  `outputs/catr_artemis_multicaption_pilot` e
  `outputs/catr_artemis_human_only_pilot`.

### Rodada completa do captioner CATR + ArtEmis

- Executamos a continuação indicada pelo piloto com todas as respostas humanas
  elegíveis: 21.383 pinturas/119.217 legendas no treino, 2.376/13.114 na
  validação e 200 pinturas do `test_stems.txt` completamente reservadas.
- O modelo base permaneceu o CATR público treinado em MS COCO humano; não houve
  Qwen. A representação visual ficou congelada. Os alvos humanos permaneceram
  curtos e independentes, com `SEP` supervisionado. O replay da legenda CATR
  caiu de três para dois e depois um por pintura.
- A validação humana melhorou nas três épocas: `5,286 → 5,103 → 5,054`. O
  critério conjunto humano + 0,15×objetivo selecionou corretamente a época 3:
  `5,492 → 5,303 → 5,257`.

| Condição nas 200 imagens reservadas | Palavras | Sinal afetivo | Reação explícita | Repete frase | Repete trigrama | SEP |
|---|---:|---:|---:|---:|---:|---:|
| CATR COCO | 10,6 | 0,0% | 0,0% | 0,0% | 8,5% | — |
| CATR + ArtEmis completo | 34,0 | 10,5% | 3,5% | 0,0% | 45,5% | 100% |
| Emocional antigo | 220,9 | 58,5% | 22,1% | 100% | 100% | — |

- Na mesma subamostra de 40 imagens, o sinal afetivo aumentou por época
  (`2,5% → 10,0% → 17,5%`), mas a repetição de trigramas não convergiu
  (`62,5% → 55,0% → 60,0%`).
- A inspeção humana reprova também a época 3: aparecem frases sintaticamente
  quebradas como “The little girls of the nice bright dress giving” e tokens
  espúrios como “bunchle”. Há ainda alucinações semânticas fortes.
- Resultado definitivo desta linha: corrigir EOS elimina o loop externo e o
  conjunto completo aumenta o sinal emocional, mas adaptar diretamente o
  decodificador BERT/CATR destrói parcialmente sua modelagem linguística. Não
  promover este modelo ao pipeline e não investir em mais épocas da mesma
  formulação. O problema já não pode ser atribuído à falta de dados.
- Script: `paper/train_catr_artemis_full.py`. Resultados, checkpoints, log e
  gerações: `outputs/catr_artemis_full_clean/`. Métricas finais:
  `outputs/catr_artemis_full_clean/metrics.csv`.

## 6. Duplicidade das features visuais canônicas por r²

- Auditamos as 258 features estritamente visuais na coorte canônica de 79.207
  imagens: SLIC multigrafo (71), descritores clássicos (130) e Hilbert (57).
  As 16 features de sentimento foram excluídas por derivarem de texto. Nenhum
  rótulo de estilo entrou no cálculo.
- Para cada par não constante calculamos Pearson `r` e `r²`. Com intercepto,
  esse `r²` é idêntico ao coeficiente de determinação de uma regressão linear
  simples. Preservamos também o sinal de `r` para detectar cópias invertidas.
- Duas features são constantes e devem ser removidas independentemente de
  qualquer classificador: `knn_n_leaf_frac` e `del_n_leaf_frac`, ambas sempre
  zero.
- Nas 256 restantes há 32.640 pares. A redundância global é baixa: mediana
  `r²=0,00477`, percentil 95 `0,3047` e percentil 99 `0,7271`.

| Corte | Pares acima | Features com algum parceiro | Componentes | Dimensões removíveis por componente |
|---:|---:|---:|---:|---:|
| r² ≥ 0,90 | 66 | 76 | 24 | 52 |
| r² ≥ 0,95 | 34 | 44 | 18 | 26 |
| r² ≥ 0,99 | 24 | 27 | 10 | 17 |

- A duplicidade mais clara são seis estatísticas Lab repetidas em três ramos:
  `L/a/b mean/std` dos descritores, `L/a/b_mean/std_w` do SLIC e posições
  correspondentes do Hilbert. Os `r²` ficam entre `0,9941` e `0,999999`.
- Outras identidades quase determinísticas incluem:
  `area_mean ↔ knn_density ↔ del_density`,
  `n_nodes ↔ del_n_edges` e
  `del_degree_max ↔ del_hub_degree_ratio`.
- Internamente, SLIC tem o maior `r²` médio (`0,1353`), seguido de Hilbert
  (`0,1068`) e descritores (`0,0734`). As correlações médias entre ramos são
  baixas; os poucos picos cruzados correspondem principalmente às estatísticas
  Lab explicitamente repetidas.
- Uma poda estritamente conservadora em `r²≥0,99`, removendo uma cópia por
  componente e as duas constantes, reduziria 258 para 239 features. Isso ainda
  não foi aplicado: r² mede duplicidade, não decide qual cópia é mais estável,
  interpretável ou útil ao classificador. O corte de 0,90 é particularmente
  agressivo porque componentes conexos podem conter pares indiretamente ligados
  cujo r² entre si é menor que o limiar.
- Script: `paper/analyze_visual_feature_redundancy_r2.py`. Pares completos:
  `paper/visual_feature_redundancy_r2_pairs.csv`; pares ≥0,80:
  `paper/visual_feature_redundancy_r2_high_pairs.csv`; clusters:
  `paper/visual_feature_redundancy_r2_clusters.csv`; resumo JSON e gráficos
  usam o prefixo `paper/visual_feature_redundancy_r2_`.

## 7. Feature importance visual — LDA e XGBoost

- Usamos as mesmas 79.207 imagens, 27 estilos, cinco folds canônicos e as 258
  features visuais sem texto. Os 34 labels ambíguos foram corrigidos pelo
  `style` qualificado armazenado junto às features SLIC.
- Balanced accuracy OOF: LDA `37,34% ± 0,64 p.p.`; XGBoost
  `39,65% ± 0,57 p.p.`.
- Para XGBoost calculamos ganho total nativo e queda de balanced accuracy por
  permutação no teste externo. Features dos 10 clusters `r²≥0,99` foram
  permutadas juntas; cada unidade teve três permutações em cada fold.
- Para LDA, magnitude de coeficiente e permutação bruta são inadequadas nesta
  matriz: 256 features não constantes têm posto efetivo 253 e número de
  condição padronizado `2,36×10¹⁰`. As cópias correlacionadas recebem
  coeficientes enormes e opostos; permutar uma coluna rompe cancelamentos e
  cria pontos fora do domínio. A correlação entre magnitude do coeficiente e
  o ranking estrutural foi somente `Spearman=0,306`.
- O ranking principal do LDA usa **structure coefficients**: correlação da
  feature padronizada com cada eixo discriminante, agregada pela variância
  explicada do eixo. A contribuição das representações foi confirmada por
  ablação com novo treinamento em cada fold.

| Representação | Massa LDA estrutural | Massa de ganho XGBoost | Queda LDA ao retirar e retreinar | Queda XGB ao permutar |
|---|---:|---:|---:|---:|
| Descritores | 41,21% | 43,36% | 4,33 p.p. | 19,25 p.p. |
| SLIC multigrafo | 36,20% | 36,56% | 3,96 p.p. | 17,53 p.p. |
| Hilbert | 22,59% | 20,08% | 1,16 p.p. | 8,17 p.p. |

- As quedas de permutação XGBoost não são aditivas: cada bloco interage com os
  demais. As ablações LDA são a medida mais direta da contribuição incremental
  de cada representação no modelo linear.
- Top LDA estrutural: `homogeneity_mean`, `mst_degree_gini`,
  `mst_degree_mean`, `lbp_4`, `mst_hub_degree_ratio`, `lbp_3`, `degree_mean`,
  `energy_std`, `lbp_2` e `knn_degree_gini`.
- Top XGBoost por permutação agrupada: `homogeneity_std` (`0,99 p.p.`), `lbp_8`
  (`0,87 p.p.`), `correlation_max` (`0,85 p.p.`),
  `del_color_dist_std` (`0,84 p.p.`), cluster das três cópias de `L_mean`
  (`0,84 p.p.`), `color_dist_std` (`0,68 p.p.`), `mst_degree_mean`
  (`0,67 p.p.`) e `knn_color_dist_mean` (`0,60 p.p.`). Os cinco primeiros
  reduziram a BAcc nas 15/15 execuções.
- Há concordância moderada por feature entre LDA estrutural e ganho XGBoost
  (`Spearman=0,517`); 14 das top-25 features coincidem. A mensagem comum é que
  textura Haralick/LBP, topologia MST/kNN e dispersão de cor dos grafos são os
  sinais dominantes. As médias Lab, que dominavam o coeficiente cru do LDA por
  colinearidade, não dominam o ranking robusto.
- Scripts: `paper/analyze_visual_feature_importance_lda_xgb.py` e
  `paper/analyze_lda_structure_importance.py`. Rankings e resultados usam os
  prefixos `paper/visual_feature_importance_lda_xgb_` e
  `paper/visual_feature_importance_lda_structure_`.

## 8. Piloto de super-resolução Real-ESRGAN

- Avaliamos o modelo oficial `RealESRGAN_x2plus`, destinado a imagens gerais,
  somente no método visual interpretável. A CNN não recebeu a transformação.
  A coorte inicial reutiliza 400 imagens balanceadas em cinco estilos do piloto
  Big-LaMa; esses valores não são comparáveis diretamente ao resultado
  canônico de 27 estilos/79.207 imagens.
- Uma auditoria aleatória de 3.000 originais mostrou lado maior mediano de
  1.842 px e nenhuma imagem com lado maior menor ou igual a 1.024 px. Portanto
  não há resolução ausente na entrada do SLIC: este teste mede se uma
  restauração aprendida cria uma representação mais discriminativa, não se
  recupera detalhe original perdido.
- Cada imagem foi normalizada com o mesmo bilinear/lado máximo 1.024 do SLIC.
  Geramos duas vistas x2 pareadas: interpolação bicúbica (placebo) e
  Real-ESRGAN. Os extratores canônicos permaneceram inalterados: SLIC volta a
  no máximo 1.024, descritores e Hilbert a 128. Após o retorno a 1.024, SSIM
  médio contra a entrada foi `0,9789` para bicúbico e `0,9255` para
  Real-ESRGAN; portanto a restauração ainda altera materialmente o sinal visto
  pelo pipeline.

| Classificador | Original | Bicúbico x2 | Real-ESRGAN x2 | SR − bicúbico |
|---|---:|---:|---:|---:|
| LDA | 47,00% | 46,50% | 49,75% | +3,25 p.p. |
| XGBoost | 67,75% | 67,75% | 69,75% | +2,00 p.p. |
| MLP | 54,50% | 57,75% | 50,25% | −7,50 p.p. |

- Na concatenação com a representação original, Real-ESRGAN menos o placebo
  bicúbico foi `−0,50 p.p.` no LDA, `−0,50 p.p.` no XGBoost e `+4,75 p.p.` na
  MLP. A MLP `original + Real-ESRGAN` chegou a 60,75%, ou `+6,25 p.p.` contra
  a MLP original nesta coorte pequena.
- O sinal não é estável o bastante para conclusão: nos cinco folds pareados,
  os p-valores SR versus bicúbico foram `0,401/0,398/0,106` na substituição e
  `0,909/0,688/0,152` na concatenação (LDA/XGBoost/MLP). Nenhuma comparação
  atingiu significância.
- A ablação por representação também é heterogênea. Na substituição, o SLIC
  explica o ganho do XGBoost (`+3,25 p.p.`), enquanto a SR prejudica a MLP nos
  três ramos isolados. Na concatenação, o ganho da MLP vem principalmente dos
  descritores (`+7,25 p.p.` contra a concatenação bicúbica); Hilbert ajuda o
  XGBoost em `+2,25 p.p.`. Isso é compatível com complementaridade, mas também
  com variância de amostra.
- Throughput medido na RTX 3090: aproximadamente `0,73 imagem/s` para
  normalizar, gerar bicúbico, inferir Real-ESRGAN x2 e salvar os três PNGs.
  Tiling 512 e imagem inteira deram velocidade semelhante. A extrapolação
  ingênua para 79.207 imagens é cerca de 30 horas; uma execução integral só se
  justifica após validação intermediária mais ampla ou integração sem salvar
  controles intermediários.
- Conclusão atual: **promissor, mas inconclusivo e caro**. Não promover ao
  pipeline. Se esta linha continuar, usar uma coorte balanceada com todos os 27
  estilos e manter o bicúbico como placebo obrigatório.
- Scripts: `paper/prepare_super_resolution_pilot.py`,
  `paper/evaluate_super_resolution_pilot.py` e
  `paper/evaluate_super_resolution_branches.py`. Resultados:
  `paper/super_resolution_pilot_results.csv`,
  `paper/super_resolution_pilot_branch_results.csv` e
  `outputs/super_resolution_pilot/roundtrip_metrics.csv`. Prancha visual:
  `outputs/super_resolution_pilot/comparison_montage_first30.png`.

### Validação intermediária em 3.000 imagens/27 estilos

- Expandimos o teste para exatamente 3.000 imagens cobrindo os 27 estilos e
  exigindo stems únicos presentes nas três representações canônicas. A
  alocação foi limitada pela disponibilidade: `Action_painting=86`,
  `Analytical_Cubism=110` e os demais estilos com 112 ou 113 imagens. A BAcc
  compensa essa pequena diferença de tamanho entre classes.
- O protocolo permaneceu idêntico: normalização bilinear a lado máximo 1.024,
  placebo bicúbico x2 e `RealESRGAN_x2plus`; somente nosso método recebeu as
  imagens transformadas. Os três extratores foram mantidos sem alteração.
- A geração terminou sem falhas a aproximadamente `0,75 imagem/s`. Após voltar
  à resolução do SLIC, o SSIM médio foi `0,9793` no bicúbico e `0,9225` no
  Real-ESRGAN, confirmando que a restauração aprendida sobrevive ao extrator.

| Classificador | Original | Bicúbico x2 | Real-ESRGAN x2 | SR − bicúbico | SR − original |
|---|---:|---:|---:|---:|---:|
| LDA | 32,44% ± 1,58 | 32,21% ± 1,38 | 32,80% ± 1,10 | +0,59 p.p. | +0,37 p.p. |
| XGBoost | 33,79% ± 1,07 | 33,30% ± 0,95 | **34,64% ± 0,61** | **+1,35 p.p.** | **+0,86 p.p.** |
| MLP | 30,83% ± 2,16 | 28,50% ± 3,58 | 31,31% ± 0,97 | +2,81 p.p. | +0,48 p.p. |

- Real-ESRGAN isolado superou o placebo nos três classificadores, mas os cinco
  folds ainda não dão significância convencional: `p=0,131` no LDA,
  `p=0,079` no XGBoost e `p=0,167` na MLP. Contra a representação original,
  os ganhos são menores e não significativos (`p=0,623/0,116/0,636`).
- A concatenação bruta não é a formulação indicada. Contra `original +
  bicúbico`, trocar o placebo por Real-ESRGAN ganhou `+1,86 p.p.` no LDA,
  `+0,78 p.p.` no XGBoost e `+0,54 p.p.` na MLP; somente o LDA teve
  `p<0,05` (`p=0,044`). Contudo, as concatenações ainda ficaram abaixo do
  Real-ESRGAN isolado nos três classificadores.
- A ablação localiza efeitos diferentes por ramo. Na substituição, os ganhos
  vêm dos descritores para LDA (`+2,08 p.p.`) e MLP (`+2,15 p.p.`); SLIC e
  Hilbert isolados não melhoram. Em contraste, quando concatenado ao original,
  SLIC ganha `+2,03/+0,69/+2,12 p.p.` sobre o placebo em LDA/XGBoost/MLP.
  Portanto a SR não melhora uniformemente cada representação; ela cria uma
  vista parcialmente complementar, principalmente em textura/descriptores.
- Conclusão atual: a validação maior reduziu os ganhos muito altos do piloto de
  400 imagens, mas preservou uma vantagem consistente sobre o placebo no vetor
  completo. O melhor resultado é XGBoost com Real-ESRGAN isolado, +0,86 p.p.
  sobre o original. É evidência promissora, porém insuficiente para pagar cerca
  de 30 horas de inferência no dataset inteiro; não promover ainda.
- Amostra e scripts: `paper/super_resolution_3k_sample.csv`,
  `paper/select_super_resolution_sample.py` e
  `paper/run_super_resolution_3k.sh`. Resultados:
  `paper/super_resolution_3k_results.csv`,
  `paper/super_resolution_3k_branch_results.csv` e
  `outputs/super_resolution_3k/roundtrip_metrics.csv`. Log completo:
  `outputs/super_resolution_3k/run.log`.

## 9. Grafo de linha do RAG SLIC — piloto em 3.000 imagens

- Usamos exatamente a mesma coorte de 3.000 imagens/27 estilos da validação de
  super-resolução. Nenhuma imagem foi relida e o SLIC não foi reexecutado: os
  RAGs brutos das 79.902 imagens já estavam exportados em
  `outputs/slic_gnn_full/graphs/`.
- No grafo de linha, cada adjacência entre regiões do RAG vira um nó e duas
  adjacências se conectam quando compartilham uma região. Em vez de materializar
  os grafos maiores, derivamos 63 agregados interpretáveis: topologia do grafo
  de linha, quantis e desigualdade de contrastes, assortatividade entre
  fronteiras consecutivas, ângulos entre centroides nas junções, continuidade
  de bordas fortes, distância Lab entre regiões externas, reversões de
  luminância e desequilíbrio de área.
- A extração terminou em `9,7 s` (`313 grafos/s`), sem falhas. Um RAG mediano
  produziu 4.975 nós e 24.971 arestas no grafo de linha. Como os RAGs já
  existem, a extrapolação direta para 79.902 imagens é cerca de 4–5 minutos;
  em uma extração futura, os agregados podem ser calculados junto com o SLIC.

| Representação | LDA | XGBoost | MLP |
|---|---:|---:|---:|
| Grafo de linha-63 isolado | 25,47% ± 1,93 | 26,39% ± 1,52 | 24,27% ± 2,21 |
| SLIC-71 | 28,14% ± 1,55 | 30,40% ± 2,88 | 26,86% ± 2,15 |
| SLIC-71 + linha-63 | **31,49% ± 1,36** | **31,53% ± 1,55** | **29,35% ± 4,09** |
| Visual-258 | 32,44% ± 1,58 | 33,79% ± 1,07 | 30,83% ± 2,16 |
| Visual-258 + linha-63 | **33,41% ± 1,98** | **35,08% ± 1,22** | **33,14% ± 2,13** |

- Sobre SLIC-71, os ganhos foram `+3,35/+1,14/+2,49 p.p.` em
  LDA/XGBoost/MLP. Sobre o vetor visual completo, foram
  `+0,97/+1,29/+2,31 p.p.`.
- O resultado mais estável foi XGBoost visual: ganhou nos cinco folds,
  `p=0,0075` contra o baseline e `p=0,0041` contra uma cópia das 63 features
  embaralhada entre imagens. A MLP visual ganhou em quatro dos cinco folds,
  mas teve `p=0,183` contra o baseline pequeno.
- Há redundância parcial: 22/63 features têm `r²≥0,90` com alguma SLIC-71 e
  cinco são cópias exatas de estatísticas de arestas já armazenadas. Mantendo
  somente as 41 features com `r²<0,90`, o sinal estrutural sobrevive:
  SLIC ganhou `+3,11/+0,79/+2,35 p.p.` e o visual completo
  `+0,93/+0,55/+2,10 p.p.` em LDA/XGBoost/MLP. Na MLP SLIC, as 41 features
  ganharam nos cinco folds (`p=0,0054`). Portanto o efeito não pode ser
  atribuído apenas à duplicação de colunas.
- As features com maior ganho XGBoost incluem Gini dos contrastes, percentil
  90 da distância espacial entre regiões externas, percentil 90 do ângulo de
  junção, percentis da distância Lab externa, mudança de contraste entre
  fronteiras consecutivas, fração com apenas uma borda forte e desigualdade de
  graus. Várias das líderes têm baixa ou moderada redundância com SLIC-71.
- O grafo de linha foi mais promissor que Real-ESRGAN nesta coorte: XGBoost
  visual + linha chegou a `35,08%`, contra `34,64%` com super-resolução e
  `33,79%` no original, com custo incremental milhares de vezes menor.
- Conclusão: esta é a representação nova mais forte e barata testada até aqui.
  A validação completa em 79k está justificada, mantendo a versão integral e a
  versão não redundante como ablação.
- Scripts: `paper/extract_line_graph_features.py`,
  `paper/evaluate_line_graph_3k.py` e
  `paper/rank_line_graph_3k_features.py`. Artefatos:
  `paper/line_graph_3k_features.parquet`,
  `paper/line_graph_3k_results.csv`,
  `paper/line_graph_3k_results_comparisons.csv`,
  `paper/line_graph_3k_redundancy.csv` e
  `paper/line_graph_3k_feature_ranking.csv`.
