"""Paired neutral-versus-emotional prompt ablation on WikiArt with Qwen2-VL.

The same deterministic, style-stratified image sample is described under both
prompts.  Tasks are interleaved by image, output is append-only JSONL, and an
existing output is resumed by the (stem, condition) key.
"""

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


ROOT = Path(__file__).parent.parent.parent
IMAGE_DIR = ROOT / "wikiart"
META_CSV = ROOT / "data/artemis/wikiart_meta.csv"
DEFAULT_OUTPUT = (
    ROOT / "outputs/affect_captions/qwen_prompt_ablation_500_per_style.jsonl"
)
DEFAULT_SAMPLE_CSV = ROOT / "paper/qwen_prompt_ablation_sample.csv"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PROMPTS = {
    "neutral": (
        "Describe only what is visibly present in this image in 2-3 concise "
        "sentences. Mention the main subjects or objects, colors, spatial "
        "arrangement, composition, and visible textures. Do not infer emotions, "
        "mood, symbolism, artistic style, artist, historical period, or intent."
    ),
    "emotional": (
        "Describe the emotional atmosphere and aesthetic feeling of this image "
        "in 2-3 concise sentences. Do not name specific objects or people. "
        "Focus on how the colors make you feel, the overall mood and emotional "
        "tone, and the visual qualities that create this impression. Do not "
        "name an artistic style, artist, or historical period."
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_path_index() -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {}
    for path in IMAGE_DIR.rglob("*"):
        if path.suffix.lower() in EXTENSIONS:
            candidates.setdefault(path.stem, []).append(path)
    duplicated = {stem: paths for stem, paths in candidates.items() if len(paths) > 1}
    if duplicated:
        log(
            f"Warning: {len(duplicated)} duplicated stems; "
            "using paths whose parent matches the metadata style"
        )
    return {stem: paths[0] for stem, paths in candidates.items()}


def build_sample(max_per_style: int, sample_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(META_CSV)
    path_index = build_path_index()
    meta["path"] = meta["stem"].astype(str).map(
        lambda stem: str(path_index[stem]) if stem in path_index else None
    )
    meta = meta.dropna(subset=["path"]).drop_duplicates("stem").copy()
    rng = np.random.default_rng(SEED)
    sampled = []
    for style, group in meta.groupby("style", sort=True):
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


def load_done(output: Path) -> set[tuple[str, str]]:
    done = set()
    if not output.exists():
        return done
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                done.add((str(row["stem"]), str(row["condition"])))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def make_tasks(sample: pd.DataFrame, done: set[tuple[str, str]]) -> list[dict]:
    tasks = []
    # Pair conditions next to each other so an interrupted run stays balanced.
    for row in sample.itertuples(index=False):
        for condition in ("neutral", "emotional"):
            key = (str(row.stem), condition)
            if key not in done:
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
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--limit_tasks", type=int, default=0)
    args = parser.parse_args()
    if args.max_per_style < 1 or args.batch_size < 1:
        raise ValueError("max_per_style and batch_size must be positive")

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
        log("Nothing to process")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    log(f"Loading {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
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
            images = []
            messages = []
            for task in batch:
                image = Image.open(task["path"]).convert("RGB")
                images.append(image)
                messages.append([{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": task["prompt"]},
                    ],
                }])
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
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_length:]
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
