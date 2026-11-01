# Manuscript workspace

Esta pasta separa o artigo dos artefatos experimentais. O atalho histórico
`paper/` aponta para `results/` e não deve receber o manuscrito.

## Arquivos

- `main.md`: primeira versão científica do artigo, escrita em inglês;
- `results_master.csv`: registro mestre dos números candidatos ao artigo;
- `evidence_map.md`: decisões editoriais, proveniência, limitações e pendências;
- `writing_plan.md`: figuras, tabelas e sequência de fechamento.

## Regra de fonte

Os CSVs e resumos indicados em `results_master.csv` são as fontes numéricas.
`CLAUDE.md` é o registro de pesquisa, não uma fonte de valores para o texto.
Quando duas execuções têm nomes semelhantes, seus números não devem ser
combinados fora do escopo pareado indicado no registro.

Em particular, estes três valores representam configurações diferentes:

- `45.2248%`: representação visual de 258 dimensões com MLP por ramos;
- `45.3601%`: baseline visual de 258 dimensões reavaliado com a MLP plana no
  experimento pareado do grafo de linha;
- `45.8106%` OOF: Numeric-274, que adiciona 16 features de sentimento.

## Convenção estatística do manuscrito

- Métrica primária: balanced accuracy;
- protocolo principal: validação cruzada estratificada em cinco folds;
- seed dos folds: 42;
- incerteza: desvio-padrão **amostral** entre os cinco folds (`ddof=1`);
- comparações de módulos: diferenças fold a fold no mesmo experimento;
- valores OOF são mantidos em coluna própria e não substituem silenciosamente
  a média dos folds;
- diferenças de coorte, pré-treinamento e linhagem textual devem aparecer na
  mesma tabela que a acurácia.

Os arquivos antigos alternam entre `ddof=0` e `ddof=1`. A coluna
`bacc_sd_sample` do registro mestre foi recalculada a partir dos cinco valores
por fold quando necessário.

## Estado do congelamento

O artigo pode ser escrito agora. Não há justificativa para abrir novas linhas
exploratórias antes de fechar as ameaças à validade já identificadas.

Pendências que alteram diretamente as conclusões:

1. importar o resultado da MLP com `Visual 258 + line + Felzenszwalb`, em
   execução em outra máquina;
2. executar uma avaliação desjunta por artista;
3. auditar duplicatas visuais entre folds;
4. medir custo fim a fim sob hardware, threads e política de CPU/GPU fixos;
5. escolher o nome definitivo do método e o venue do artigo.

O teste multiescala já está concluído (`48.3285%` sem texto). Sua combinação
com line graph e Felzenszwalb não é necessária para iniciar o manuscrito: os
módulos respondem a perguntas diferentes e podem ser apresentados como pontos
de operação, não como uma busca combinatória pelo maior número.

