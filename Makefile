SHELL := /bin/bash

# WikiArt and feature paths
WIKIART_DIR       ?= data/wikiart
WIKIART_META      ?= data/artemis/wikiart_meta.csv
OUT_SLIC_FEATURES ?= outputs/wikiart_slic_features
OUT_FEATURES      ?= outputs/wikiart_features
OUT_RICH_FEATURES ?= outputs/wikiart_rich_features
OUT_RICH_FOURIER  ?= results/fourier_rich_features.parquet
OUT_QUADRANT_FEATURES ?= outputs/wikiart_quadrant_features
ROBUSTNESS_DIR    ?= data/wikiart_robustness
ROBUSTNESS_PER_STYLE ?= 100
ROBUSTNESS_WORKERS ?= 8
ROBUSTNESS_FEATURES_DIR ?= outputs/wikiart_robustness_features

# SLIC configuration selected on a balanced validation pilot.
SLIC_SEGMENTS    ?= 2048
SLIC_COMPACTNESS ?= 10
SLIC_WORKERS     ?= 32
QUADRANT_WORKERS ?= 16

# Activate the desired virtual environment before invoking make, or override
# with e.g. make PYTHON=.venv-rtx3090-py312/bin/python.
PYTHON ?= python3

.PHONY: all setup slic_features features rich_features quadrant_features gatys_gram vgg_pool classify baselines robustness_dataset robustness_graph_features slm_artemis affect_captions affect_finetune affect_features affect cnn_scratch cnn_frozen cnn_frozen_cached cnn_transfer benchmark_list benchmark_dry_run benchmark_run benchmark_summary

all: slic_features

## 0. Setup the complete project environment.
setup:
	$(PYTHON) -m pip install -r requirements.txt

## 1. Primary graph pipeline: one SLIC segmentation, four graph families.
slic_features:
	$(PYTHON) src/pipeline/slic_multigraph_features.py \
		--image-dir $(WIKIART_DIR) \
		--meta $(WIKIART_META) \
		--output-dir $(OUT_SLIC_FEATURES) \
		--segments $(SLIC_SEGMENTS) \
		--compactness $(SLIC_COMPACTNESS) \
		--workers $(SLIC_WORKERS)

# Alias for the primary feature target.
features: slic_features

## Extended but nested measurement ladders for the within-representation
## saturation experiment.  The graph family is already expanded by
## `slic_features`: RAG + kNN + Delaunay + MST = 71d in one SLIC pass.
rich_features:
	mkdir -p $(OUT_RICH_FEATURES)/descriptors $(OUT_RICH_FEATURES)/hilbert
	$(PYTHON) src/baselines/extract_descriptors_full.py \
		--image_dir $(WIKIART_DIR) --meta $(WIKIART_META) \
		--output $(OUT_RICH_FEATURES)/descriptors --feature-profile rich \
		--workers $(SLIC_WORKERS)
	$(PYTHON) src/baselines/extract_hilbert_features.py \
		--image_dir $(WIKIART_DIR) --meta $(WIKIART_META) \
		--output $(OUT_RICH_FEATURES)/hilbert --feature-profile rich \
		--workers $(SLIC_WORKERS)
	$(PYTHON) experiments/PRISM/extract_fourier_full_features.py \
		--image-dir $(WIKIART_DIR) --meta $(WIKIART_META) \
		--output $(OUT_RICH_FOURIER) --feature-profile rich \
		--workers $(SLIC_WORKERS)

## Global image remains the existing feature table; extract only its four fixed quadrants.
quadrant_features:
	for REGION in top_left top_right bottom_left bottom_right; do \
		$(PYTHON) src/pipeline/slic_multigraph_features.py \
			--image-dir $(WIKIART_DIR) --meta $(WIKIART_META) \
			--output-dir $(OUT_QUADRANT_FEATURES)/$$REGION/slic \
			--segments $(SLIC_SEGMENTS) --compactness $(SLIC_COMPACTNESS) \
			--region $$REGION --workers $(QUADRANT_WORKERS); \
		$(PYTHON) src/baselines/extract_descriptors_full.py \
			--image_dir $(WIKIART_DIR) --meta $(WIKIART_META) \
			--output $(OUT_QUADRANT_FEATURES)/$$REGION \
			--region $$REGION --workers $(QUADRANT_WORKERS); \
		$(PYTHON) src/baselines/extract_hilbert_features.py \
			--image_dir $(WIKIART_DIR) --meta $(WIKIART_META) \
			--output $(OUT_QUADRANT_FEATURES)/$$REGION \
			--region $$REGION --workers $(QUADRANT_WORKERS); \
	done

## Gatys-style compact Gram baseline (frozen VGG-19; related-work adaptation).
gatys_gram:
	$(PYTHON) src/baselines/extract_gatys_compact_gram.py
	$(PYTHON) experiments/evaluate_gatys_compact_gram_mlp.py

## Paired frozen VGG controls for the compact Gatys Gram representation.
vgg_pool:
	$(PYTHON) src/baselines/extract_vgg_pool_features.py
	$(PYTHON) experiments/evaluate_vgg_pool_mlp.py

## Controlled image perturbations for modality-invariance diagnostics.
robustness_dataset:
	$(PYTHON) src/data/build_wikiart_robustness.py \
		--image-dir $(WIKIART_DIR) --meta $(WIKIART_META) \
		--output-dir $(ROBUSTNESS_DIR) --per-style $(ROBUSTNESS_PER_STYLE) \
		--workers $(ROBUSTNESS_WORKERS)

robustness_graph_features:
	$(PYTHON) src/pipeline/slic_multigraph_features.py \
		--image-dir $(ROBUSTNESS_DIR) --manifest $(ROBUSTNESS_DIR)/manifest.csv \
		--output-dir $(ROBUSTNESS_FEATURES_DIR)/slic \
		--segments $(SLIC_SEGMENTS) --compactness $(SLIC_COMPACTNESS) \
		--workers $(SLIC_WORKERS)

## 5. Classification ablation
classify:
	$(PYTHON) src/pipeline/xgboost_ablation.py \
		--graph_parquet $(OUT_SLIC_FEATURES)/graph_features.parquet \
		--knn_parquet $(OUT_SLIC_FEATURES)/knn_features.parquet \
		--output $(OUT_SLIC_FEATURES)/classification

## 6. Extract CNN and descriptor baselines
baselines:
	mkdir -p $(OUT_FEATURES)
	$(PYTHON) src/baselines/extract_cnn_full.py \
		--image_dir $(WIKIART_DIR) --meta $(WIKIART_META) --output $(OUT_FEATURES)
	$(PYTHON) src/baselines/extract_descriptors_full.py \
		--image_dir $(WIKIART_DIR) --meta $(WIKIART_META) --output $(OUT_FEATURES)

## 7. Affect pipeline (pinturas ArtGAN+BestArtworks → ViT-S destilado → 768-dim)
# Coloque ArtGAN em raw/artgan/ e Best Artworks em raw/best_artworks/ antes de rodar.
ARTGAN_DIR      := raw/artgan
BESTARTWORKS_DIR := raw/best_artworks

affect_captions:
	$(PYTHON) src/affect/generate_captions.py \
		--image_dirs $(ARTGAN_DIR) $(BESTARTWORKS_DIR)

affect_finetune:
	$(PYTHON) src/affect/finetune_vits.py

affect_features:
	$(PYTHON) src/affect/extract_affect_features.py

affect: affect_captions affect_finetune affect_features

## 8. Imported Qwen descriptions of ArtEmis/WikiArt paintings.
ARTEMIS_MANIFEST := data/artemis/manifest.csv
SLM_ARTEMIS_OUT  := outputs/slm_descriptions/artemis_descriptions.jsonl

slm_artemis:
	$(PYTHON) src/slm/generate_descriptions.py \
		--manifest $(ARTEMIS_MANIFEST) \
		--output $(SLM_ARTEMIS_OUT) \
		--batch_size 8

## 9. End-to-end WikiArt ResNet-50 controls for transfer learning.
## Both targets use the same folds and hyperparameters; only initialization changes.
CNN_TRAIN_ARGS   ?= --batch-size 128 --workers 16 --epochs 75 --patience 12 --lr 0.03

# Configuration-driven matrix.  Changing DATASETS/MODELS is enough to extend
# the comparison to a new art or texture benchmark.
DATASETS ?= wikiart pandora rijksmuseum
MODELS ?= engineered_mlp resnet50_scratch resnet50_frozen
BENCHMARK_WORKERS ?= 16

benchmark_list:
	$(PYTHON) experiments/benchmark.py list

benchmark_dry_run:
	$(PYTHON) experiments/benchmark.py run --datasets $(DATASETS) --models $(MODELS) --workers $(BENCHMARK_WORKERS) --dry-run

benchmark_run:
	$(PYTHON) experiments/benchmark.py run --datasets $(DATASETS) --models $(MODELS) --workers $(BENCHMARK_WORKERS)

benchmark_summary:
	$(PYTHON) experiments/benchmark.py summarize

cnn_scratch:
	$(PYTHON) src/baselines/train_resnet_wikiart.py \
		--image-dir $(WIKIART_DIR) --meta $(WIKIART_META) --init scratch \
		--output outputs/cnn_scratch $(CNN_TRAIN_ARGS)

cnn_frozen:
	$(PYTHON) src/baselines/train_resnet_wikiart.py \
		--image-dir $(WIKIART_DIR) --meta $(WIKIART_META) --init frozen \
		--output outputs/cnn_frozen $(CNN_TRAIN_ARGS)

cnn_frozen_cached:
	$(PYTHON) src/baselines/train_frozen_resnet_cached.py \
		--image-dir $(WIKIART_DIR) --meta $(WIKIART_META) \
		--output outputs/cnn_frozen_cached $(CNN_TRAIN_ARGS)

cnn_transfer:
	$(PYTHON) src/baselines/train_resnet_wikiart.py \
		--image-dir $(WIKIART_DIR) --meta $(WIKIART_META) --init imagenet \
		--output outputs/cnn_transfer $(CNN_TRAIN_ARGS)
