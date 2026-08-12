"""Fast fixed-resolution runner for the paired Qwen WikiArt prompt ablation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from generate_prompt_ablation_wikiart import (
    EXTENSIONS,
    IMAGE_DIR,
    META_CSV,
    MODEL_ID,
    ROOT,
    SEED,
    load_done,
    log,
)


DEFAULT_OUTPUT = (
    ROOT / "outputs/affect_captions/qwen_prompt_ablation_500_per_style.jsonl"
)
DEFAULT_SAMPLE_CSV = ROOT / "paper/qwen_prompt_ablation_sample.csv"
PROMPTS = {
    "neutral": (
        "Use exactly two concise sentences and no more than 50 words total. "
        "Describe only what is visibly present in this image. Mention the main "
        "subjects or objects, colors, spatial arrangement, composition, and "
        "visible textures. Do not infer emotions, mood, symbolism, artistic "
        "style, artist, historical period, or intent."
    ),
    "emotional": (
        "Use exactly two concise sentences and no more than 50 words total. "
        "Describe the emotional atmosphere and aesthetic feeling of this image. "
        "Do not name specific objects or people. Focus on how the colors make "
        "you feel, the overall mood and emotional tone, and the visual qualities "
        "that create this impression. Do not name an artistic style, artist, "
        "or historical period."
    ),
}


def build_sample(max_per_style: int, sample_csv: Path) -> pd.DataFrame:
    paths = [
        path for path in IMAGE_DIR.rglob("*")
        if path.suffix.lower() in EXTENSIONS
    ]
    exact = {(path.parent.name, path.stem): path for path in paths}
    by_stem: dict[str, Path] = {}
    for path in paths:
        by_stem.setdefault(path.stem, path)

    meta = pd.read_csv(META_CSV)
    resolved = []
    for row in meta.itertuples(index=False):
        path = exact.get((str(row.style), str(row.stem)))
        if path is None:
            path = by_stem.get(str(row.stem))
        resolved.append(str(path) if path is not None else None)
    meta["path"] = resolved
    meta = meta.dropna(subset=["path"]).drop_duplicates("stem").copy()
    rng = np.random.default_rng(SEED)
    sampled = []
    for _, group in meta.groupby("style", sort=True):
        take = min(max_per_style, len(group))
        indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
        sampled.append(group.loc[indices])
    sample = (
        pd.concat(sampled, ignore_index=True)
        .sort_values(["style", "stem"])
        .reset_index(drop=True)
    )
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    sample[["stem", "filename", "artist", "style", "split", "path"]].to_csv(
        sample_csv, index=False
    )
    return sample


def make_tasks(sample: pd.DataFrame, done: set[tuple[str, str]]) -> list[dict]:
    tasks = []
    for row in sample.itertuples(index=False):
        for condition in ("neutral", "emotional"):
            if (str(row.stem), condition) in done:
                continue
            tasks.append({
                "stem": str(row.stem),
                "file_name": str(row.filename),
                "style": str(row.style),
                "dataset_split": str(row.split),
                "path": Path(row.path),
                "condition": condition,
                "prompt": PROMPTS[condition],
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample_csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--max_per_style", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit_tasks", type=int, default=0)
    args = parser.parse_args()

    sample = build_sample(args.max_per_style, args.sample_csv)
    done = load_done(args.output)
    tasks = make_tasks(sample, done)
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    log(
        f"Sample={len(sample):,} images across {sample['style'].nunique()} styles; "
        f"done={len(done):,}; remaining tasks={len(tasks):,}"
    )
    if not tasks:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    with args.output.open("a", encoding="utf-8") as output_handle:
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start:start + args.batch_size]
            images = [Image.open(task["path"]).convert("RGB") for task in batch]
            messages = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": task["prompt"]},
                ],
            }] for task, image in zip(batch, images)]
            texts = [
                processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            inputs = processor(
                text=texts, images=images, padding=True, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            captions = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            for task, caption in zip(batch, captions):
                output_handle.write(json.dumps({
                    "stem": task["stem"],
                    "file_name": task["file_name"],
                    "style": task["style"],
                    "dataset_split": task["dataset_split"],
                    "condition": task["condition"],
                    "prompt": task["prompt"],
                    "caption": caption.strip(),
                    "model": MODEL_ID,
                    "visual_pixels": visual_pixels,
                }) + "\n")
            output_handle.flush()
            for image in images:
                image.close()
            processed += len(batch)
            if processed % 100 == 0 or processed == len(tasks):
                elapsed = time.perf_counter() - started
                rate = processed / elapsed
                eta = (len(tasks) - processed) / rate
                log(
                    f"{processed:,}/{len(tasks):,}; {rate:.2f} tasks/s; "
                    f"elapsed={elapsed / 3600:.2f}h; ETA={eta / 3600:.2f}h"
                )
    log(f"Completed: {args.output}")


if __name__ == "__main__":
    main()
