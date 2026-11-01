"""Generate one cumulative six-sentence neutral description per WikiArt image.

The output is generated only once.  Downstream experiments truncate it after
sentences 1..6, so every ladder step has an identical prefix and its marginal
token and inference cost can be measured without six separate generations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


ROOT = Path(__file__).parent.parent.parent
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
SAMPLE_CSV = ROOT / "paper/qwen_prompt_ablation_sample.csv"
DEFAULT_OUTPUT = (
    ROOT / "outputs/affect_captions/qwen_neutral_6sentence_ladder.jsonl"
)
PROMPT = (
    "Write exactly six numbered lines about only what is visibly present in "
    "this image. Each line must be one complete concise sentence of at most "
    "18 words. Line 1: main subjects or objects. Line 2: dominant colors and "
    "lighting. Line 3: spatial arrangement and composition. Line 4: visible "
    "shapes and textures. Line 5: depth and background. Line 6: additional "
    "visible details. Do not infer emotions, mood, symbolism, artistic style, "
    "artist, historical period, or intent. Do not add an introduction or "
    "conclusion."
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(str(json.loads(line)["stem"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def generated_length(ids: torch.Tensor, eos_ids: set[int]) -> int:
    for index, token in enumerate(ids.tolist(), 1):
        if int(token) in eos_ids:
            return index
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_csv", type=Path, default=SAMPLE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sample = pd.read_csv(args.sample_csv, dtype=str)
    if sample["stem"].duplicated().any():
        raise ValueError("Sample contains duplicate stems")
    done = load_done(args.output)
    tasks = [row for row in sample.itertuples(index=False) if row.stem not in done]
    if args.limit:
        tasks = tasks[:args.limit]
    log(
        f"Sample={len(sample):,}; already done={len(done):,}; "
        f"tasks this run={len(tasks):,}"
    )
    if not tasks:
        return

    visual_pixels = 256 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=visual_pixels,
        max_pixels=visual_pixels,
        local_files_only=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    ).eval()
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else set(map(int, eos))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    total_generated_tokens = 0
    with args.output.open("a", encoding="utf-8") as output_handle:
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start:start + args.batch_size]
            images = [Image.open(row.path).convert("RGB") for row in batch]
            messages = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }] for image in images]
            texts = [
                processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            torch.cuda.synchronize()
            batch_started = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            batch_seconds = time.perf_counter() - batch_started
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            captions = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            lengths = [generated_length(ids, eos_ids) for ids in generated_ids]
            for row, caption, token_count in zip(batch, captions, lengths):
                output_handle.write(json.dumps({
                    "stem": str(row.stem),
                    "file_name": str(row.filename),
                    "style": str(row.style),
                    "dataset_split": str(row.split),
                    "condition": "neutral_structured_6_sentences",
                    "prompt": PROMPT,
                    "caption": caption.strip(),
                    "model": MODEL_ID,
                    "visual_pixels": visual_pixels,
                    "max_new_tokens": args.max_new_tokens,
                    "generated_tokens": token_count,
                    "batch_generation_seconds_per_image": (
                        batch_seconds / len(batch)
                    ),
                }) + "\n")
            output_handle.flush()
            for image in images:
                image.close()
            processed += len(batch)
            total_generated_tokens += sum(lengths)
            if processed % 100 == 0 or processed == len(tasks):
                elapsed = time.perf_counter() - started
                rate = processed / elapsed
                eta = (len(tasks) - processed) / rate
                log(
                    f"{processed:,}/{len(tasks):,}; {rate:.2f} images/s; "
                    f"mean output={total_generated_tokens / processed:.1f} tokens; "
                    f"elapsed={elapsed / 3600:.2f}h; ETA={eta / 3600:.2f}h"
                )
    log(f"Completed: {args.output}")


if __name__ == "__main__":
    main()
