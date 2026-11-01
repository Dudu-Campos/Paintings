# Ladders of measurement within a representation

These extractors expose an explicit `--feature-profile` instead of silently
replacing a feature table:

| representation | compact | rich | construction of the added measurements |
|---|---:|---:|---|
| Descriptors | 130d | 425d | RGB/HSV/Lab distributions, extra LBP and GLCM scales, gradient orientations |
| Hilbert curve | 57d | 417d | 4/16/32-band pyramid, increments and autocorrelations of the same Lab sequence |
| Fourier | 152d | 608d | historical luminance spectrum plus the identical polar statistic on Lab channels |
| Region graph | 32d RAG | 71d multigraph | RAG plus kNN, Delaunay and MST edge summaries (`src/extractors/slic_multigraph_features.py`) |

The expanded profile of the first three rows always begins with the compact
vector, in exactly the same order.  This makes curves such as `BAcc(k)`
interpretable as *more measurements of the same representation*, rather than
a switch of representation or model.

For WikiArt, run all image-based ladders with:

```bash
make rich_features
```

This writes separate outputs and never overwrites the compact feature tables
that underlie the reported results.  Evaluate a compact-versus-rich ladder
within each fold, with fitting/ranking performed on the training split only.
