# Paintings

Experimentos para classificação de 27 estilos de pintura do WikiArt. O projeto
compara representações visuais estruturadas e interpretáveis (segmentação,
descritores e série de Hilbert) com baselines clássicos, CNNs e representações
pré-treinadas.

## Estrutura

- `src/`: extração de dados/features, baselines e visualizações;
- `experiments/`: scripts de avaliação, comparação e geração de figuras;
- `results/`: tabelas agregadas e figuras versionadas;
- `paper/`: atalho de compatibilidade para `results/`; não adicionar código
  novo ali;
- `meta/`: metadados pequenos mantidos no repositório;
- `data/`, `wikiart/`, `raw/` e `outputs/`: dados, caches, modelos e artefatos
  de execução locais — não são versionados;
- `conversa.md`: estado compartilhado entre as conversas de pesquisa;
- `HANDOFF.md`: decisões e detalhes históricos do pipeline SLIC.

## Ambiente e dados

Python 3.11+ é o alvo. Crie/ative um ambiente virtual apropriado à máquina e
instale as dependências:

```bash
python -m pip install -r requirements.txt
```

As imagens devem estar em `wikiart/<estilo>/`. Os metadados oficiais usados
pelos fluxos atuais ficam em `data/artemis/wikiart_meta.csv`, baixados
localmente pelo fluxo de dados. Datasets, embeddings, imagens transformadas e
checkpoints não entram no Git.

## Pipeline principal

Com o ambiente ativo e os dados locais disponíveis:

```bash
make slic_features
```

Isso extrai as quatro famílias de features a partir de uma única segmentação
SLIC. Outros alvos úteis são `make cnn_scratch`, `make cnn_frozen`,
`make cnn_transfer`, `make gatys_gram` e `make vgg_pool`. Veja `make -n <alvo>`
antes de executar uma extração longa.

## Resultados

`results/metodos_final.csv` é a tabela compacta de métodos consolidados.
Resultados e figuras adicionais ficam em `results/`; os scripts que os geram e
o índice de reprodução ficam em `experiments/README.md`.

O protocolo principal usa validação cruzada estratificada de cinco folds, com
balanced accuracy como métrica primária. Não compare resultados de coortes
distintas sem registrar a diferença de dados e de pré-treinamento.
