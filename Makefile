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

.PHONY: all build eps_star coarsen features baselines clean

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

clean:
	rm -f coarse2 eps_star_ssim
