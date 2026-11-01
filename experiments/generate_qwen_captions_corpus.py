"""Legenda a coorte com o PROFESSOR (Qwen2-VL-7B), para o teto do canal de texto.

O §5.18 registra que a diferenca entre o embedding do CLIP e os ~15pp do canal de
texto NAO e identificavel -- "a linguagem descarta a supervisao de estilo" e "o
nosso decoder e ruim" preveem a mesma coisa. Este braco separa: se o professor
tambem ficar em ~15-20, o limite e o GARGALO DE LINGUAGEM; se der 45, o limite e
o nosso DECODER.

Tres decisoes de protocolo, todas para nao medir outra coisa sem querer:

1. **O prompt e byte-identico ao da destilacao** (`02_teacher_text_llm.PROMPT`).
   O estudante foi treinado para imitar ESTE professor sob ESTE prompt; usar um
   prompt melhor aqui mediria "Qwen com prompt afinado", nao o teto do estudante.
   Ele inclui "Do not invent a story or name any style, artist or place", entao a
   restricao de vazamento e a mesma dos dois lados.
2. **Orcamento visual igualado ao da destilacao.** O professor viu COCO nativo
   (~640x480 = 307k pixels); as pinturas tem mediana de 1.000 px de lado maior
   (§7.1), ate 16.905. Sem cap o Qwen veria 10x mais tokens visuais aqui do que
   quando ensinou, e o "teto" mediria um professor com mais orcamento que o
   original. `--max-pixels` default 307.200 iguala os dois.
3. **A coorte e a ORDEM vem do CSV do estudante**, nao do metadado. O §5.18b
   mede que stems reconstruidos saem em ordem diferente e custam -2,92pp em
   silencio, porque o casamento por stem nao reclama.

Retomavel: grava JSONL append-only e pula o que ja tem legenda.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# O FlashInfer compila kernels em JIT e exige `ninja`, que nao existe nesta
# maquina -- sem isso o EngineCore do vLLM morre em "FileNotFoundError: ninja"
# DEPOIS de 145 s carregando o modelo. Instalar ninja mexeria no venv do NFS, que
# tres maquinas compartilham (§1). Como este braco gera com `temperature=0`
# (greedy), o sampler de top-p/top-k do FlashInfer nao e usado para nada.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None                 # §8: varreduras de museu passam de 600 MP

ROOT = Path(__file__).resolve().parent.parent
I2T = ROOT / "experiments/image-to-text"
sys.path.insert(0, str(I2T))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--captions", type=Path, default=None,
                        help="CSV do estudante: fonte da coorte e da ordem")
    parser.add_argument("--image-root", type=Path,
                        default=ROOT / "data/multitask_painting100k/original/images")
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=640 * 480)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--loaders", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-mem-frac", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_image(path: Path, max_pixels: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize((max(28, int(image.width * scale)),
                              max(28, int(image.height * scale))), Image.BICUBIC)
    return image


def _safe_load(path: Path, max_pixels: int):
    try:
        return load_image(path, max_pixels)
    except Exception:                                   # noqa: BLE001
        return None


def main() -> None:
    args = parse_args()
    ds = args.dataset
    args.captions = args.captions or ROOT / f"results/{ds}_student_captions.csv"
    args.meta = args.meta or ROOT / f"data/{ds}_meta.csv"
    args.out_jsonl = args.out_jsonl or ROOT / f"results/{ds}_qwen_captions.jsonl"
    args.out_csv = args.out_csv or ROOT / f"results/{ds}_qwen_captions.csv"

    cohort = pd.read_csv(args.captions, dtype={"stem": str, "style": str})
    if args.limit:
        cohort = cohort.groupby("style", group_keys=False).head(
            max(1, args.limit // cohort["style"].nunique()))
    log(f"{ds}: coorte de {len(cohort):,} imagens, {cohort['style'].nunique()} classes")

    # Resolver caminho por (estilo, arquivo): o §8 registra QUATRO extratores que
    # indexaram so por nome de arquivo e gravaram menos imagens EM SILENCIO.
    meta = pd.read_csv(args.meta, dtype={"stem": str, "filename": str, "style": str})
    by_stem = dict(zip(meta["stem"], meta["filename"]))
    paths, missing = {}, 0
    for stem, style in zip(cohort["stem"], cohort["style"]):
        name = by_stem.get(stem)
        candidate = args.image_root / style / name if name else None
        if candidate is None or not candidate.is_file():
            candidate = next(args.image_root.glob(f"*/{name}"), None) if name else None
        if candidate is None:
            missing += 1
        else:
            paths[stem] = candidate
    if missing:
        log(f"  AVISO: {missing:,} imagens nao resolvidas no disco")
    log(f"  resolvidas: {len(paths):,}")

    done = set()
    if args.out_jsonl.is_file():
        with args.out_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["stem"])
                except Exception:                      # linha truncada por kill
                    continue
        log(f"  retomando: {len(done):,} ja legendadas")

    todo = [(s, t) for s, t in zip(cohort["stem"], cohort["style"])
            if s in paths and s not in done]
    log(f"  a fazer: {len(todo):,}")

    if todo:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        import importlib
        teacher = importlib.import_module("02_teacher_text_llm")
        prompt_body = teacher.PROMPT                    # byte-identico a destilacao

        processor = AutoProcessor.from_pretrained(args.model)
        chat = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_body}]}]
        prompt_text = processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_frac,
                  limit_mm_per_prompt={"image": 1}, dtype="bfloat16",
                  max_model_len=4096)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

        started = time.time()
        with args.out_jsonl.open("a", encoding="utf-8") as sink:
            for start in range(0, len(todo), args.chunk):
                block = todo[start:start + args.chunk]
                # Leitura EM PARALELO: as 48 GB de originais estao no NFS, que o
                # §1 mede 12x mais lento que o NVMe local, e decodificar pintura
                # de ~1.000 px num laco de uma thread deixa a GPU ociosa. Com o
                # pool a vazao subiu de 4,3 para o valor registrado no log.
                with ThreadPoolExecutor(max_workers=args.loaders) as pool:
                    loaded = list(pool.map(
                        lambda pair: (pair, _safe_load(paths[pair[0]], args.max_pixels)),
                        block))
                requests, rows = [], []
                for (stem, style), image in loaded:
                    if image is None:
                        log(f"  ilegivel {stem}")
                        continue
                    requests.append({"prompt": prompt_text,
                                     "multi_modal_data": {"image": image}})
                    rows.append((stem, style))
                if not requests:
                    continue
                for (stem, style), out in zip(rows, llm.generate(requests, sampling)):
                    sink.write(json.dumps(
                        {"stem": stem, "style": style,
                         "caption": out.outputs[0].text.strip()},
                        ensure_ascii=False) + "\n")
                sink.flush()
                seen = min(start + args.chunk, len(todo))
                rate = seen / max(1e-9, time.time() - started)
                log(f"  {seen:,}/{len(todo):,} — {rate:.1f} img/s — "
                    f"restam {(len(todo)-seen)/max(rate,1e-9)/60:.0f} min")

    # --- consolidar no MESMO schema do estudante, na MESMA ordem -----------
    rows = [json.loads(l) for l in args.out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    frame = pd.DataFrame(rows).drop_duplicates("stem", keep="last").set_index("stem")
    kept = cohort[cohort["stem"].isin(frame.index)].copy()
    kept["caption"] = frame.loc[kept["stem"], "caption"].to_numpy()
    kept[["stem", "style", "caption"]].to_csv(args.out_csv, index=False)
    log(f"salvo: {args.out_csv} ({len(kept):,} legendas, "
        f"{100*len(kept)/len(cohort):.1f}% da coorte)")


if __name__ == "__main__":
    main()
