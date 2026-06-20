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

.PHONY: all build eps_star coarsen features baselines clean coco_download slm_descriptions slm

all: build eps_star coarsen features

## 1. Build C++ binaries
build:
	$(CXX) $(CXXFLAGS) src/cpp/coarse2.cpp -o coarse2
	$(CXX) $(CXXFLAGS) src/cpp/eps_star_ssim.cpp -o eps_star_ssim

## 2. Compute ε* per image (entropy criterion)
eps_star:
	mkdir -p $(OUT_EPS_STAR)
	python src/pipeline/compute_eps_star_entropy.py \
		--coarse_bin $(COARSE_BIN) \
		--image_dir $(RAW_DIR) \
		--meta $(META) \
		--output_dir $(OUT_EPS_STAR)

## 3. Materialize graphs at ε*
coarsen:
	mkdir -p $(OUT_COARSENING)
	python src/pipeline/run_coarsening2.py \
		--coarse_bin $(COARSE_BIN) \
		--image_dir $(RAW_DIR) \
		--eps_lookup $(EPS_LOOKUP) \
		--coarse_dir $(OUT_COARSENING) \
		--workers 32

## 4. Extract graph features (balanced, min=31 max=100)
features:
	mkdir -p $(OUT_FEATURES)
	python src/pipeline/lda_graph_features.py \
		--coarse_dir $(OUT_COARSENING)/coarse \
		--meta $(META) \
		--output $(OUT_FEATURES) \
		--balanced_min 31 --balanced_max 100 --min_count 0 \
		--workers 32
	python src/pipeline/lda_knn_features.py \
		--coarse_dir $(OUT_COARSENING)/coarse \
		--meta $(META) \
		--output $(OUT_FEATURES) \
		--balanced_min 31 --balanced_max 100 --min_count 0 \
		--k 10 --workers 32
	python src/pipeline/lda_combined_features.py \
		--coarse_dir $(OUT_COARSENING)/coarse \
		--meta $(META) \
		--output $(OUT_FEATURES) \
		--balanced_min 31 --balanced_max 100 --min_count 0 \
		--k 10 --workers 32

## 5. Classification ablation
classify:
	python src/pipeline/xgboost_ablation.py \
		--features $(OUT_FEATURES)/graph_features.parquet
	python src/pipeline/compare_baselines_filtered.py \
		--features_dir $(OUT_FEATURES)

## 6. Extract CNN and descriptor baselines
baselines:
	mkdir -p $(OUT_FEATURES)
	python src/baselines/extract_cnn_full.py \
		--image_dir $(RAW_DIR) --meta $(META) --output $(OUT_FEATURES)
	python src/baselines/extract_descriptors_full.py \
		--image_dir $(RAW_DIR) --meta $(META) --output $(OUT_FEATURES)

## 7. Affect pipeline (pinturas ArtGAN+BestArtworks → ViT-S destilado → 768-dim)
# Coloque ArtGAN em raw/artgan/ e Best Artworks em raw/best_artworks/ antes de rodar.
ARTGAN_DIR      := raw/artgan
BESTARTWORKS_DIR := raw/best_artworks

affect_captions:
	python src/affect/generate_captions.py \
		--image_dirs $(ARTGAN_DIR) $(BESTARTWORKS_DIR)

affect_finetune:
	python src/affect/finetune_vits.py

affect_features:
	python src/affect/extract_affect_features.py

affect: affect_captions affect_finetune affect_features

## 8. SLM image description (COCO 2017 → descrições via vLLM)
COCO_DIR        := data/coco
COCO_SPLITS     := val2017
SLM_MODEL       := Qwen/Qwen2.5-VL-3B-Instruct
SLM_OUT         := outputs/slm_descriptions/coco_descriptions.jsonl

coco_download:
	python src/slm/download_coco.py --data_dir $(COCO_DIR) --splits $(COCO_SPLITS)

slm_descriptions:
	python src/slm/generate_descriptions.py \
		--manifest $(COCO_DIR)/manifest.csv \
		--split $(COCO_SPLITS) \
		--model $(SLM_MODEL) \
		--output $(SLM_OUT) \
		--batch_size 8

slm: coco_download slm_descriptions

clean:
	rm -f coarse2 eps_star_ssim
