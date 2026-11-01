"""Embedding denso das legendas, como modalidade em vez de bloco tipo-posterior.

O bloco `Text` atual é um **posterior de classe**: um classificador sobre TF-IDF
das legendas devolvendo p(estilo|legenda), com log-lift e nested OOF. Isso o
obriga a carregar toda a maquinaria anti-vazamento e o deixa exposto à crítica
de auto-referência. Um embedding é uma modalidade honesta -- **nenhum rótulo do
corpus entra nele**, ele só descreve.

Modelo: `sentence-transformers/all-MiniLM-L6-v2` (384d, mean pooling sobre os
tokens com máscara de atenção, depois normalização L2). Pequeno e específico
para similaridade de sentença; roda em CPU em minutos.

Duas coisas a registrar sobre o que este bloco é:

1. **As legendas são do CATR/COCO e são delirantes neste domínio.** Exemplos
   reais do Multitask: "A close up of a kite with a sky background" para um
   `Abstract_Art`, "A large group of candies are arranged in a display case"
   para outro. O captioner não tem vocabulário para pintura. Semanticamente é
   ruído; a hipótese de trabalho é que funcione como **impressão digital de
   estatística visual**, não como descrição. Se o bloco pagar, é provavelmente
   por isso, e o texto do artigo tem de dizer isso em vez de sugerir compreensão
   semântica.

2. **Text é supervisão externa**, igual a ResNet e ViT: o captioner foi treinado
   em ~600 mil legendas humanas do COCO, e o encoder de sentença em corpora
   maiores ainda. Não entra no número visual-only do §3.

Prior não animador: na forma de posterior, Text sobre base com grafo deu +1,01pp
no LDA e **+0,16pp n.s. no MLP** no OilPainting (§7.2).

Uso (o filesystem é compartilhado; escolher o venv da máquina):

    .venv/bin/python3 experiments/embed_captions.py                  # mojito
    .venv-rtx3090-py312/bin/python3 experiments/embed_captions.py    # margarita
    ... --captions outputs/affect_captions/catr_coco_oilpainting.jsonl \\
        --output outputs/oilpainting_caption_embeddings
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path,
                        default=ROOT / "outputs/affect_captions/catr_coco_multitask.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu",
                        help="cpu basta: 100k legendas curtas saem em poucos minutos")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/multitask_caption_embeddings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stems, texts = [], []
    with args.captions.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stems.append(str(row["stem"]))
            texts.append(row["caption"])
    log(f"{len(texts):,} legendas | exemplo: {texts[0]!r}")
    lengths = np.array([len(t.split()) for t in texts])
    log(f"palavras por legenda: mediana {np.median(lengths):.0f}, "
        f"p90 {np.percentile(lengths, 90):.0f} | "
        f"{len(set(texts)):,} legendas distintas "
        f"({100 * len(set(texts)) / len(texts):.1f}%)")

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    dim = model.config.hidden_size
    log(f"{args.model}: {dim}d em {args.device}")

    out = np.zeros((len(texts), dim), dtype=np.float32)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start:start + args.batch_size]
            batch = tokenizer(chunk, padding=True, truncation=True,
                              max_length=64, return_tensors="pt").to(args.device)
            hidden = model(**batch).last_hidden_state
            # Mean pooling COM máscara: tokens de padding não podem entrar na
            # média, senão legendas curtas ficam diluídas por zeros.
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            out[start:start + len(chunk)] = pooled.cpu().numpy()
            if start and start % (args.batch_size * 50) == 0:
                rate = start / (time.perf_counter() - started)
                log(f"  {start:,}/{len(texts):,}  {rate:.0f} legendas/s  "
                    f"ETA {(len(texts) - start) / rate / 60:.1f} min")
    np.save(args.output / "caption_embed_features.npy", out)
    np.save(args.output / "caption_embed_stems.npy", np.array(stems, dtype=object))
    log(f"salvo: {out.shape} em {args.output} "
        f"({(time.perf_counter() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
