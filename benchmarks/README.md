# Benchmark interface

The benchmark matrix is configured, not hard-coded.  Dataset declarations are
in `configs/benchmarks/datasets.toml`; comparison-model declarations are in
`configs/benchmarks/models.toml`.

```bash
# Inspect what is currently available.
python experiments/benchmark.py list

# Validate paths and show all commands without starting a long run.
python experiments/benchmark.py run \
  --datasets wikiart pandora rijksmuseum \
  --models engineered_mlp resnet50_scratch resnet50_frozen \
  --dry-run

# Run one data/model pair.  Local runs are resumable in outputs/benchmarks/.
python experiments/benchmark.py run --datasets pandora --models engineered_mlp

# Normalize completed runs into one comparison table.
python experiments/benchmark.py summarize
```

`engineered_*` evaluates registered explicit representations.  `resnet50_*`
uses the same metadata, target, folds, early-stopping split and metrics, but
trains an end-to-end CNN or a frozen ImageNet head.  `resnet50_frozen` is the
probe over an already extracted ImageNet embedding; run its extractor before
selecting it on a new dataset.  `benchmark.py summarize` promotes only the
compact comparison table to `results/benchmarks/summary.csv`; checkpoints,
predictions and per-fold working files remain ignored under `outputs/`.

To add a texture dataset, add its images and label table, copy the section in
`configs/benchmarks/dataset.template.toml`, then implement/register only the
feature extractor that is absent.  The evaluator and matrix launcher need no
dataset-specific edits.

For another comparison model, either add a `runner = "blocks"` entry (for any
combination of representations in `registry.py`) or copy the custom-command
template in `configs/benchmarks/models.template.toml`.  Its placeholders cover
the dataset name, image root, metadata, target, fold count, worker count and
output directory.  A custom evaluator only needs to emit its standard summary
in that output directory to participate in the combined table.
