"""Etapa 02 — geração de texto com o professor (Qwen2-VL via vLLM).

Para cada imagem do image_index, o VLM gera uma descrição com ênfase no tom
emocional do que é *visível* (cor, luz, composição, expressões) — sem inventar
narrativa. Essas legendas são os alvos de destilação do estudante.

Pontos de atenção (início/fim da descrição):
  - prompt pede N frases e "termine com ponto";
  - `max_tokens` como teto duro;
  - pós-processamento corta na última sentença completa e ANEXA <eos> explícito
    (o estudante só aprende a parar se todo alvo terminar em <eos>);
  - filtrar recusas, repetições e legendas fora da faixa de comprimento.

Entrada:
    artifacts/image_index.csv      (etapa 01)

Saída:
    artifacts/teacher_captions.csv   image_id, path, split, caption, n_tokens
    artifacts/teacher_captions.jsonl mesmo conteúdo + prompt + raw (auditoria)

Uso:
    python 02_teacher_text_llm.py --limit 30000 --tp 1 --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EOS,
    IMAGE_INDEX_CSV,
    OUT_DIR,
    TEACHER_CAPTIONS_CSV,
    TEACHER_CAPTIONS_JSONL,
    TEACHER_MODEL,
    TEACHER_MAX_NEW_TOKENS,
    TEACHER_TEMPERATURE,
    log,
)

PROMPT = (
    "Describe this image in exactly two short sentences, at most 40 words "
    "total. State the main visible subjects, then the emotional tone evoked by "
    "the colors, light and composition — only from what is visible. Do not "
    "invent a story or name any style, artist or place. End with a period."
)

REFUSAL_RE = re.compile(r"\b(i cannot|i can't|as an ai|i'm sorry|i am unable)\b", re.I)
SENT_END_RE = re.compile(r"[.!?]")
# Calibrado ao prompt de "2 frases / <=40 palavras": aceita até ~55 tokens
# (folga para pontuação e artigos) e trunca na última sentença completa.
MIN_TOKENS, MAX_TOKENS_KEEP = 8, 55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=IMAGE_INDEX_CSV)
    parser.add_argument("--model", default=TEACHER_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="0 = todas")
    parser.add_argument("--splits", default="train,val,test",
                        help="splits do image_index a processar")
    parser.add_argument("--max-tokens", type=int, default=TEACHER_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEACHER_TEMPERATURE)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--gpu-mem-frac", type=float, default=0.90)
    parser.add_argument("--batch", type=int, default=0,
                        help="0 = enviar TODOS os prompts numa só generate() "
                             "(recomendado); N>0 fatia só p/ checkpoint")
    parser.add_argument("--resume", action="store_true", help="pular já feitos")
    return parser.parse_args()


def postprocess(text: str) -> tuple[str, bool]:
    """Corta na última sentença completa e anexa <eos>.

    Retorna (caption_com_eos, aceita?). Rejeita recusas, vazios e legendas
    fora da faixa de comprimento. O <eos> só é anexado a legendas aceitas —
    é o sinal de parada que o estudante aprenderá.
    """
    text = " ".join(str(text).split())
    if not text or REFUSAL_RE.search(text):
        return "", False
    ends = [m.end() for m in SENT_END_RE.finditer(text)]
    if ends:
        text = text[: ends[-1]]                # corta na última sentença completa
    n = len(text.split())
    if n < MIN_TOKENS or n > MAX_TOKENS_KEEP:
        return "", False
    return f"{text} {EOS}", True


def load_done(path: Path) -> set[int]:
    done: set[int] = set()
    if path.is_file():
        for line in path.open(encoding="utf-8"):
            try:
                done.add(int(json.loads(line)["image_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def load_image(path: str, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        image = image.resize((max(1, int(w * s)), max(1, int(h * s))))
    return image


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    wanted = set(args.splits.split(","))
    index = index[index["split"].isin(wanted)].reset_index(drop=True)
    if args.limit:
        index = index.head(args.limit)

    done = load_done(TEACHER_CAPTIONS_JSONL) if args.resume else set()
    if done:
        index = index[~index["image_id"].isin(done)].reset_index(drop=True)
    log(f"Gerando legendas do professor para {len(index)} imagens "
        f"(model={args.model}, vLLM tp={args.tp}, já feitas={len(done)})")

    # Prompt de chat com um placeholder de imagem — construído uma vez; o vLLM
    # recebe a imagem separadamente via multi_modal_data.
    processor = AutoProcessor.from_pretrained(args.model)
    chat = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_frac,
        limit_mm_per_prompt={"image": 1},
        dtype="bfloat16",
        max_model_len=4096,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=[EOS],
    )

    kept, dropped, total = [], 0, len(index)
    # --batch 0 (padrão) = uma única generate() sobre TODOS os prompts; o
    # continuous batching do vLLM agenda internamente com throughput máximo.
    # --batch N > 0 fatia só para poder fazer checkpoint em runs muito grandes.
    chunk_size = args.batch if args.batch > 0 else total
    mode = "a" if (args.resume and TEACHER_CAPTIONS_JSONL.is_file()) else "w"
    with TEACHER_CAPTIONS_JSONL.open(mode, encoding="utf-8") as jl:
        for start in range(0, total, chunk_size):
            chunk = index.iloc[start:start + chunk_size]
            # Carrega imagens e monta a lista de requests do bloco de uma vez.
            requests = []
            valid_rows = []
            for _, row in chunk.iterrows():
                try:
                    image = load_image(row["path"], args.max_side)
                except Exception as exc:  # noqa: BLE001
                    log(f"  imagem ilegível {row['file_name']}: {exc}")
                    continue
                requests.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": image},
                })
                valid_rows.append(row)

            # Um único envio em lote: vLLM recebe todos os prompts juntos.
            outputs = llm.generate(requests, sampling)

            for row, out in zip(valid_rows, outputs):
                raw = out.outputs[0].text.strip()
                caption, ok = postprocess(raw)
                rec = {
                    "image_id": int(row["image_id"]),
                    "file_name": row["file_name"], "path": row["path"],
                    "split": row["split"], "prompt": PROMPT, "raw": raw,
                    "caption": caption, "n_tokens": len(caption.split()),
                    "accepted": ok,
                }
                jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if ok:
                    kept.append(rec)
                else:
                    dropped += 1
            jl.flush()
            log(f"  {min(start + chunk_size, total)}/{total} — "
                f"aceitas={len(kept)} descartadas={dropped}")

    kept_df = pd.DataFrame(kept, columns=[
        "image_id", "file_name", "path", "split", "caption", "n_tokens"])
    kept_df.to_csv(TEACHER_CAPTIONS_CSV, index=False)
    rate = dropped / max(1, total)
    log(f"Escrito {TEACHER_CAPTIONS_CSV} ({len(kept_df)}) e "
        f"{TEACHER_CAPTIONS_JSONL}; descarte={rate:.1%}")
    if len(kept_df):
        log(f"n_tokens: min={kept_df.n_tokens.min()} "
            f"mediana={int(kept_df.n_tokens.median())} max={kept_df.n_tokens.max()}")


if __name__ == "__main__":
    main()
