"""Canal textual do estudante destilado, avaliado no Multitask.

O estudante de `experiments/image-to-text/` foi medido no WikiArt (0,242 BAcc
contra 0,181 do CATR standalone). Aqui ele roda no Multitask, que e o corpus
vivo -- as imagens do WikiArt nao existem mais nesta maquina.

Protocolo identico ao do `05_evaluate.py` para o numero ser comparavel: CLIP
ViT-B/32 congelado -> estudante -> legenda, e depois TF-IDF binario ngram(1,2)
min_df=10 + LinearSVC L1 C=0.1 `class_weight=balanced`, 5 folds, BAcc OOF, com
os tokens do nome do estilo removidos antes do vetorizador.

O canal NAO entra no metodo final (§3, supervisao externa: o professor e o CATR
sobre COCO e o encoder e o CLIP). Isto mede quanto ele vale, nao se ele entra.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
I2T = ROOT / "experiments" / "image-to-text"
sys.path.insert(0, str(I2T))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--image-dir", type=Path,
                        default=Path("/tmp/prism_data/multitask_painting100k/images"))
    parser.add_argument("--ckpt", type=Path, default=I2T / "artifacts/student_ckpt/best.pt")
    parser.add_argument("--clip-cache", type=Path, default=None)
    parser.add_argument("--captions", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = coorte inteira")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.clip_cache = args.clip_cache or ROOT / f"outputs/{args.dataset}_clip_vitb32.npy"
    args.captions = args.captions or ROOT / f"results/{args.dataset}_student_captions.csv"
    args.output = args.output or ROOT / f"results/{args.dataset}_student_text_channel.json"

    import importlib
    ev = importlib.import_module("05_evaluate")
    st = importlib.import_module("04_student")

    meta = pd.read_csv(args.meta, dtype={"stem": str, "filename": str, "style": str})
    if args.limit:
        meta = meta.groupby("style", group_keys=False).head(
            max(1, args.limit // meta["style"].nunique()))
    # Casar por (estilo, arquivo) quando as imagens estao em pastas por estilo.
    # Indexar so por `p.name` e o gotcha do §8: o OilPainting tem **258 nomes
    # duplicados** (636 arquivos) em estilos diferentes, e o dicionario fica com
    # o ULTIMO caminho de cada nome -- legenda errada em silencio. O Multitask
    # tem diretorio plano, entao ali o fallback por nome e o caminho certo.
    files = [f for f in args.image_dir.rglob("*") if f.is_file()]
    by_pair = {(f.parent.name, f.name): f for f in files}
    by_name: dict[str, Path] = {}
    ambiguous = set()
    for f in files:
        if f.name in by_name:
            ambiguous.add(f.name)
        by_name[f.name] = f

    def resolve(style: str, filename: str):
        hit = by_pair.get((str(style), str(filename)))
        if hit is not None:
            return hit
        if str(filename) in ambiguous:
            return None          # nome ambiguo e estilo nao bateu: recusar
        return by_name.get(str(filename))

    resolved = [resolve(r["style"], r["filename"]) for _, r in meta.iterrows()]
    missing = sum(v is None for v in resolved)
    if missing:
        print(f"[aviso] {missing:,} de {len(meta):,} imagens nao resolvidas "
              f"({len(ambiguous)} nomes ambiguos no disco)", flush=True)
    # Legendar EXATAMENTE a coorte do registry, nao todo arquivo do diretorio.
    # Duas coisas de graca: (a) o rijksdraw tem uma imagem corrompida
    # (`RP-T-1940-571.jpg`) que o `build_clip_cache` nao tolera -- ele abre o lote
    # inteiro numa list comprehension e um arquivo ruim derruba a execucao -- e a
    # coorte ja a exclui; (b) as legendas ficam alinhadas por construcao com tudo
    # o que consome features. Checar legibilidade arquivo a arquivo custaria ~14
    # min de latencia de NFS e daria o mesmo resultado.
    try:
        from evaluate_minc2500_official import HAND8
        from registry import load_blocks
        cohort_stems = set(map(str, load_blocks(args.dataset, HAND8,
                                                allow_external=True).stems))
    except Exception as exc:                       # dataset fora do registry
        print(f"[aviso] coorte indisponivel ({exc}); usando o metadado inteiro",
              flush=True)
        cohort_stems = None

    keep = [v is not None and (cohort_stems is None or str(st) in cohort_stems)
            for v, st in zip(resolved, meta["stem"])]
    dropped = len(resolved) - sum(keep)
    if dropped:
        print(f"[aviso] {dropped:,} de {len(resolved):,} linhas fora da coorte "
              f"ou sem arquivo", flush=True)
    meta = meta[keep].reset_index(drop=True)
    paths = [str(v) for v, k in zip(resolved, keep) if k]
    print(f"[{time.strftime('%H:%M:%S')}] {len(paths):,} imagens, "
          f"{meta['style'].nunique()} estilos", flush=True)

    if args.captions.exists():
        frame = pd.read_csv(args.captions, dtype=str)
        # O cache tem de COBRIR a coorte, senao um arquivo parcial (de um
        # `--limit` anterior) e reaproveitado em silencio e a execucao "termina"
        # em segundos com uma fracao das imagens. Aconteceu em 08/set: um teste
        # de fumaca de 200 imagens deixou 184 linhas aqui e a rodada cheia do
        # minc2500 as reusou em vez de gerar 57.500.
        if len(frame) < len(meta):
            raise SystemExit(
                f"cache PARCIAL em {args.captions}: {len(frame):,} legendas para "
                f"{len(meta):,} linhas da coorte. Apagar o arquivo e regerar, ou "
                f"passar --captions apontando para outro caminho.")
        print(f"legendas reaproveitadas de {args.captions} ({len(frame):,})", flush=True)
    else:
        feats = st.build_clip_cache(paths, args.clip_cache)
        tok = ev.load_tokenizer() if hasattr(ev, "load_tokenizer") else None
        if tok is None:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(I2T / "artifacts/tokenizer/tokenizer.json"))
        bos, eos, pad = (tok.token_to_id(t) for t in ("<bos>", "<eos>", "<pad>"))
        model = ev.load_student(args.ckpt, tok.get_vocab_size(), pad)
        captions = ev.gen_captions(model, feats, tok, bos, eos, ev.MAX_CAPTION_TOKENS)
        frame = pd.DataFrame({"stem": meta["stem"], "style": meta["style"], "caption": captions})
        frame.to_csv(args.captions, index=False)
        print(f"legendas salvas: {args.captions}", flush=True)

    result = ev.tfidf_classify(frame)
    result["dataset"] = args.dataset
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
