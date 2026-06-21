SHELL := /bin/bash

# Paths
META      := meta/meta.csv
RAW_DIR   := raw/images
COARSE_BIN := ./coarse2

OUT_COARSENING := outputs/coarsening
OUT_EPS_STAR   := outputs/eps_star
OUT_FEATURES   := outputs/features

EPS_LOOKUP := $(OUT_EPS_STAR)/eps_star_entropy_index.parquet

# Compilation flags
CXX      := g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17

# Python via uv (resolve o ambiente automaticamente, sem ativar o venv)
PYTHON := uv run python

.PHONY: all setup build eps_star coarsen features baselines clean coco_download slm_descriptions slm caption_finetune

all: build eps_star coarsen features

## 0. Setup do ambiente Python (core). Extras: uv sync --extra baselines|affect|slm
setup:
	uv sync

## 1. Build C++ binaries
build:
	$(CXX) $(CXXFLAGS) src/cpp/coarse2.cpp -o coarse2
	$(CXX) $(CXXFLAGS) src/cpp/eps_star_ssim.cpp -o eps_star_ssim

## 2. Compute ε* per image (entropy criterion)
eps_star:
	mkdir -p $(OUT_EPS_STAR)
	$(PYTHON) src/pipeline/compute_eps_star_entropy.py \
		--coarse_bin $(COARSE_BIN) \
		--image_dir $(RAW_DIR) \
		--meta $(META) \
		--output_dir $(OUT_EPS_STAR)

## 3. Materialize graphs at ε*
coarsen:
	mkdir -p $(OUT_COARSENING)
	$(PYTHON) src/pipeline/run_coarsening2.py \
		--coarse_bin $(COARSE_BIN) \
		--image_dir $(RAW_DIR) \
		--eps_lookup $(EPS_LOOKUP) \
		--coarse_dir $(OUT_COARSENING) \
		--workers 32

## 4. Extract graph features (balanced, min=31 max=100)
features:
	mkdir -p $(OUT_FEATURES)
	$(PYTHON) src/pipeline/lda_graph_features.py \
		--coarse_dir $(OUT_COARSENING)/coarse \
		--meta $(META) \
		--output $(OUT_FEATURES) \
		--balanced_min 31 --balanced_max 100 --min_count 0 \
		--workers 32
	$(PYTHON) src/pipeline/lda_knn_features.py \
		--coarse_dir $(OUT_COARSENING)/coarse \
		--meta $(META) \
		--output $(OUT_FEATURES) \
		--balanced_min 31 --balanced_max 100 --min_count 0 \
		--k 10 --workers 32
	$(PYTHON) src/pipeline/lda_combined_features.py \
		--coarse_dir $(OUT_COARSENING)/coarse \
		--meta $(META) \
		--output $(OUT_FEATURES) \
		--balanced_min 31 --balanced_max 100 --min_count 0 \
		--k 10 --workers 32

## 5. Classification ablation
classify:
	$(PYTHON) src/pipeline/xgboost_ablation.py \
		--features $(OUT_FEATURES)/graph_features.parquet
	$(PYTHON) src/pipeline/compare_baselines_filtered.py \
		--features_dir $(OUT_FEATURES)

## 6. Extract CNN and descriptor baselines
baselines:
	mkdir -p $(OUT_FEATURES)
	$(PYTHON) src/baselines/extract_cnn_full.py \
		--image_dir $(RAW_DIR) --meta $(META) --output $(OUT_FEATURES)
	$(PYTHON) src/baselines/extract_descriptors_full.py \
		--image_dir $(RAW_DIR) --meta $(META) --output $(OUT_FEATURES)

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

## 8. SLM image description (COCO 2017 → descrições via vLLM)
COCO_DIR        := data/coco
COCO_SPLITS     := val2017
SLM_MODEL       := Qwen/Qwen2.5-VL-3B-Instruct
SLM_OUT         := outputs/slm_descriptions/coco_descriptions.jsonl

coco_download:
	$(PYTHON) src/slm/download_coco.py --data_dir $(COCO_DIR) --splits $(COCO_SPLITS)

slm_descriptions:
	$(PYTHON) src/slm/generate_descriptions.py \
		--manifest $(COCO_DIR)/manifest.csv \
		--split $(COCO_SPLITS) \
		--model $(SLM_MODEL) \
		--output $(SLM_OUT) \
		--batch_size 8

slm: coco_download slm_descriptions

## 9. Image captioning fine-tune (ViT+GPT2) usando as descrições do SLM como alvo
# Requer imagens COCO baixadas (make coco_download) e o extra: uv sync --extra caption
CAPTION_OUT := outputs/caption_model

caption_finetune:
	$(PYTHON) src/slm/finetune_caption.py \
		--descriptions $(SLM_OUT) \
		--coco_dir $(COCO_DIR) \
		--output_dir $(CAPTION_OUT) \
		--train_split train2017 \
		--val_split val2017

clean:
	rm -f coarse2 eps_star_ssim
