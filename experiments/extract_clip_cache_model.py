"""Cache de features de imagem CLIP para um modelo arbitrario, na ORDEM da coorte.

Existe porque a ablacao de captioner precisa de uma segunda escala de CLIP
(ViT-L/14) e o cache de producao so tem o ViT-B/32. O `04_student.py` constroi
o cache dele embutido no pipeline do estudante; aqui o mesmo procedimento fica
exposto como script, com duas invariantes que o §8 obriga:

* **A ordem das linhas e a do `<ds>_student_captions.csv`**, que e a fonte
  canonica de stem+style para todos os consumidores de CLIP deste projeto. O
  `evaluate_clip_text_arms.py` supoe linha-a-linha e so checa o TAMANHO -- um
  cache na ordem errada passaria a checagem e daria resultado embaralhado em
  silencio, que e o -2,92pp do §5.18b.
* **Cada modelo usa o processador CANONICO dele** (`CLIPImageProcessor.
  from_pretrained(<modelo>)`), nao um resize escrito a mao. Comparar duas
  escalas de CLIP sob pre-processamento diferente confundiria modelo com
  resolucao, e o §8 mede esse par valendo 4,91pp.

As imagens saem da copia local reduzida (lado menor 256), a MESMA que o cache
do B/32 usou -- §9.0 registra que ela e derivada, nao copia, e para CLIP a 224
px isso e o ponto de operacao canonico.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

# Varredura de museu estoura o limite anti-bomba do PIL (§8); aqui a imagem e
# so redimensionada para 224, entao liberar e seguro.
Image.MAX_IMAGE_PIXELS = None

torch.set_num_threads(16)
ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--clip-model", required=True)
    ap.add_argument("--image-dir", type=Path,
                    default=Path("/tmp/prism_data/multitask_painting100k/images"))
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--captions", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--decode-workers", type=int, default=16,
                    help="decode de JPEG grande domina o custo; o forward do "
                         "CLIP em lote de 256 e rapido perto disso")
    args = ap.parse_args()
    args.meta = args.meta or ROOT / f"data/{args.dataset}_meta.csv"
    args.captions = args.captions or ROOT / f"results/{args.dataset}_student_captions.csv"

    frame = pd.read_csv(args.captions, dtype={"stem": str, "style": str})
    meta = pd.read_csv(args.meta, dtype={"stem": str, "filename": str, "style": str})
    nome_de = dict(zip(meta["stem"], meta["filename"]))

    files = [f for f in args.image_dir.rglob("*") if f.is_file()]
    by_pair = {(f.parent.name, f.name): f for f in files}
    by_name: dict[str, Path] = {}
    ambiguos = set()
    for f in files:
        if f.name in by_name:
            ambiguos.add(f.name)
        by_name[f.name] = f
    log(f"{len(files):,} arquivos no disco, {len(ambiguos)} nomes ambiguos")

    paths, faltando = [], 0
    for stem, style in zip(frame["stem"], frame["style"]):
        fn = nome_de.get(stem)
        hit = by_pair.get((str(style), str(fn))) if fn else None
        if hit is None and fn and fn not in ambiguos:
            hit = by_name.get(str(fn))
        if hit is None:
            faltando += 1
        paths.append(hit)
    if faltando:
        raise SystemExit(f"{faltando:,} de {len(frame):,} imagens nao resolvidas -- "
                         "abortando em vez de gravar cache com buraco")
    log(f"{len(paths):,} caminhos resolvidos, na ordem do CSV de legendas")

    processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    clip = CLIPModel.from_pretrained(args.clip_model).to(DEVICE).eval()
    dim = int(clip.config.projection_dim)
    log(f"{args.clip_model}: projecao {dim}d")

    feats = np.zeros((len(paths), dim), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(paths), args.batch):
            lote = paths[i:i + args.batch]
            with ThreadPoolExecutor(args.decode_workers) as pool:
                imgs = list(pool.map(lambda q: Image.open(q).convert("RGB"), lote))
            px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
            emb = clip.get_image_features(pixel_values=px)
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output
            feats[i:i + len(lote)] = emb.float().cpu().numpy()
            if (i // args.batch) % 20 == 0:
                feito = min(i + args.batch, len(paths))
                taxa = feito / max(time.time() - t0, 1e-9)
                log(f"  {feito:,}/{len(paths):,}  {taxa:.0f} img/s  "
                    f"faltam {(len(paths)-feito)/max(taxa,1e-9)/60:.1f} min")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, feats)
    log(f"salvo: {args.output} {feats.shape}")


if __name__ == "__main__":
    main()
