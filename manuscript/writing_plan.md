# Plano de escrita e fechamento

## Mensagem em uma frase

Uma representação explícita de cor, textura e estrutura regional alcança um
ponto de operação competitivo sem pré-treinamento; escala, relações de segunda
ordem e segmentações distintas acrescentam sinal mensurável e auditável.

## Tabelas do corpo principal

1. **Comparação principal**: chance, Haralick+HOG, RF, kNN, SVM-RBF, ResNet do
   zero, Visual-258 e visual multiescala. Colunas obrigatórias: BAcc, coorte,
   pré-treino e custo.
2. **Ablation de estrutura**: SLIC, SLIC+line, Visual-258, Visual+line e
   Visual+line+Felzenszwalb para LDA, XGBoost e MLP. A célula MLP final fica
   marcada como pendente até importar a execução externa.
3. **Baselines pré-treinados**: VGG GAP, Gram de Gatys, ResNet frozen e
   fine-tuned, separados visualmente dos métodos sem pré-treino.
4. **Importância por representação**: massa LDA, ganho XGBoost e ablação.

## Figuras do corpo principal

1. Diagrama do pipeline: imagem -> segmentação -> regiões -> quatro grafos ->
   estatísticas; ramo de descritores e ramo Hilbert; módulos multiescala,
   line graph e Felzenszwalb.
2. Ganho pareado do line graph e Felzenszwalb por fold e classificador. Mostrar
   baseline, representação real e placebo permutado.
3. Fronteira custo x balanced accuracy, separando sem pré-treino, encoder
   frozen e fine-tuning.
4. Figura compacta de importância: contribuição dos três blocos e top features
   interpretáveis.

Matriz de confusão e redundância completa vão para o apêndice.

## Ordem de trabalho

1. Escrever introdução, método e protocolo a partir do estado congelado.
2. Importar o MLP Felzenszwalb quando a outra máquina terminar.
3. Gerar a tabela estrutural final e a figura de ganhos pareados.
4. Executar controles por artista e duplicidade.
5. Fechar medição de custo.
6. Fazer revisão bibliográfica e substituir todos os marcadores `[CITE: ...]`.
7. Escolher venue e converter `main.md` para o template LaTeX correspondente.
8. Revisar linguagem, limitações e checklist de reprodutibilidade.

## Critério de parada experimental

Depois das quatro pendências obrigatórias — MLP Felzenszwalb, split por
artista, duplicatas e custo — novos experimentos só entram se corrigirem uma
ameaça concreta à validade. A busca por mais décimos de balanced accuracy não
é, por si só, motivo para atrasar a submissão.

