"""Constantes compartilhadas da pipeline image-to-text.

Um único ponto de verdade para caminhos, dimensões e tokens especiais, para
que todas as etapas concordem (o estudante precisa dos mesmos tokens que o
tokenizer emitiu; o avaliador precisa do mesmo comprimento máximo).

Esboço inicial: valores são propostas — ajustar conforme as decisões em aberto
no README.
"""

from __future__ import annotations

import time
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # raiz do repo Paintings
STAGE_DIR = Path(__file__).resolve().parent          # experiments/image-to-text
OUT_DIR = STAGE_DIR / "artifacts"                    # saídas de todas as etapas

# COCO (professor treina/gera aqui) — cada servidor tem sua cópia.
COCO_ROOT = ROOT / "data" / "coco"
# WikiArt (só usado no evaluate, para comparar com o CATR) — TODO confirmar path.
WIKIART_ROOT = ROOT / "data" / "wikiart"

# Artefatos nomeados que cada etapa produz/consome.
IMAGE_INDEX_CSV = OUT_DIR / "image_index.csv"            # 01_prepare
TEACHER_CAPTIONS_CSV = OUT_DIR / "teacher_captions.csv"  # 02_teacher_text_llm
TEACHER_CAPTIONS_JSONL = OUT_DIR / "teacher_captions.jsonl"
TOKENIZER_DIR = OUT_DIR / "tokenizer"                    # 03_tokenizer
STUDENT_CKPT_DIR = OUT_DIR / "student_ckpt"              # 04_student
TRAIN_LOG_CSV = OUT_DIR / "train_log.csv"                # 04_student (loss/ROUGE)
EVAL_REPORT_CSV = OUT_DIR / "eval_report.csv"            # 05_evaluate

# --------------------------------------------------------------------------
# Professor (VLM)
# --------------------------------------------------------------------------
TEACHER_MODEL = "Qwen/Qwen2-VL-7B-Instruct"   # mesmo VLM fixado no projeto
TEACHER_MAX_NEW_TOKENS = 96
TEACHER_TEMPERATURE = 0.0                      # greedy p/ alvos limpos

# --------------------------------------------------------------------------
# Tokens especiais (compartilhados tokenizer <-> student)
# --------------------------------------------------------------------------
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
IMG = "<img>"                                  # marcador do prefixo visual
SPECIAL_TOKENS = [PAD, BOS, EOS, IMG]
VOCAB_SIZE = 8000                              # BPE sobre legendas do professor
# Piloto COCO val (~5k legendas, 5093 palavras únicas): o BPE satura em ~6.3k
# tokens antes do teto. Para o COCO train completo (118k) subir p/ ~12000.

# --------------------------------------------------------------------------
# Estudante (decoder-only, prefixo visual)
# --------------------------------------------------------------------------
D_MODEL = 512
MAX_SEQ_LEN = 512
N_LAYERS = 8
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
PREFIX_LEN = 10                                # tokens de prefixo (ClipCap)
IMAGE_ENCODER = "ViT-B/32"                     # CLIP congelado; TODO alternativa

# Comprimento textual máximo do alvo (trunca o professor p/ "caber").
# Prompt do professor pede 2 frases / <=40 palavras -> ~45 tokens de folga.
MAX_CAPTION_TOKENS = 48

# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------
SEED = 42
LR = 2e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 256
EPOCHS = 15
LABEL_SMOOTHING = 0.1
WARMUP_RATIO = 0.05


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
