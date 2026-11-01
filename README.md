# PRISM — Decomposing images in a PRISM

Code and supplementary material for:

> **Decomposing images in a PRISM: Single-Source Multi-View Framework for
> Image Representation**
> Eduardo Campos, Willam Cardoso, Diego Furtado, Solange Oliveira, Alneu Lopes
> Institute of Mathematical and Computer Sciences, University of São Paulo,
> São Carlos, Brazil

## Appendix

The appendix referenced in the paper is in this repository:

**[`appendix/prism_appendix.pdf`](appendix/prism_appendix.pdf)** — *Appendix A,
Encoding Maps detail* (18 pages): the formal definition of each encoding map,
the per-view feature lists, and the supplementary tables.

## What PRISM is

A single image is decomposed into eight hand-designed views — global colour and
texture descriptors, two region-adjacency multigraphs (Felzenszwalb and SLIC),
their two line graphs, a space-filling-curve descriptor, a Fourier descriptor
and a Gabor descriptor — plus a parameter-free similarity-graph operator over
the resulting space. No labels from outside the target corpus are used by the
visual bank; the optional text channel is reported as a separate, clearly
marked condition because it does carry external supervision.

Seven corpora are evaluated: MultitaskPainting100k (25 and 125 styles),
OilPainting, Rijksmuseum drawings, Food-101, Places-LT, MINC-2500 and
NCT-CRC-HE-100K.

## Layout

| path | contents |
|---|---|
| `appendix/` | the paper's appendix (PDF) |
| `src/pipeline/` | the library: feature registry, models, protocol, corpora |
| `src/extractors/` | the 25 image-to-feature modules (8 production + dormant) |
| `src/baselines/` | backbones (ResNet, ViT, VGG, Gatys) and the fine-tuning trainer |
| `experiments/` | evaluation, ablation and figure scripts |
| `results/` | aggregated tables and the figures used in the paper |
| `attic/` | retired scripts, kept so that reported negative results stay verifiable |
| `configs/`, `Makefile` | benchmark declarations and extraction targets |

`src/pipeline/registry.py` is the single source of truth for which
representations exist, how the cohort is defined as an explicit intersection,
and which columns are dropped as degenerate.

Datasets, image caches, model checkpoints and intermediate graph exports are
**not** versioned (see `.gitignore`). All seven corpora are public and are
obtained from their original distributors; the official train/test splits are
used wherever the dataset publishes one.

## Environment

Python 3.11+.

```bash
python -m pip install -r requirements.txt
```

## Protocol

Unless a script states otherwise:

- `SEED = 42`, five-fold `StratifiedKFold(shuffle=True)` for corpora without a
  published split; the published split is used for those that have one.
- **Balanced accuracy** is the primary metric. Accuracy and macro-F1 are
  emitted alongside it in every CSV, because training uses
  `class_weight="balanced"` and therefore trades simple accuracy for macro
  recall — tables state which of those quantities they report.
- Results from different cohorts or different classifier heads are not mixed in
  a single table.

`experiments/README.md` indexes which script produces which table and figure.

## Reproducing a table or figure

Each script carries its command and its guards in its own docstring. Scripts
that aggregate previously computed posteriors re-check them against the values
recorded in the paper and abort on divergence, so running them is also a
regression test.
