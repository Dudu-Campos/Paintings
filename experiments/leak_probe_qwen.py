"""Data-leakage / memorization probe for the Qwen-VL captioner.

The evaluated cohort of WikiArt paintings is public and predates the VLM's
training crawl.  This probe asks the model, blind and with no context, to name
the ARTIST, the YEAR and the TITLE of each painting.  If the model can recite
those from the pixels alone, it has either memorized the specific instance or
carries strong style/artist priors from pre-training -- exactly the confound
that motivates the CATR/COCO "clean" control in this project.  We therefore
report a per-field recall rate as an auditable upper bound on that leakage.

Ground truth comes from WikiArt itself: the file name encodes
``artist_title-year.jpg`` and ``classes.csv`` carries the qualified ``artist``
and ``description`` (title slug).  No label from the classifier is used.

Two backends are supported so the probe runs whatever compute is free:

* ``--backend openai``  -- query an already-running vLLM / OpenAI-compatible
  server over HTTP.  No extra GPU is used; the model stays resident in the
  server.  Point ``--base-url`` at it and pass the key via ``--api-key-env``.
* ``--backend transformers`` -- load ``Qwen2-VL`` locally with transformers
  (needs a free GPU and the weights in the HF cache).  Mirrors the decoding
  used by ``src/affect/generate_qwen_sentence_ladder.py``.

Outputs (written under ``--out-dir``, default ``results/``):

* ``leak_probe_qwen_cohort.csv``   -- the exact 10 pinned images + ground truth
* ``leak_probe_qwen_raw.jsonl``    -- every question, prompt and raw answer
* ``leak_probe_qwen_results.csv``  -- per-image match flags
* ``leak_probe_qwen_summary.csv``  -- aggregate recall per field

Example (endpoint backend, 10 images)::

    python experiments/leak_probe_qwen.py \\
        --wikiart-root ~/.cache/kagglehub/datasets/steubk/wikiart/versions/1 \\
        --backend openai \\
        --base-url http://10.200.200.3:8000/v1 \\
        --api-key-env VLLM_API_KEY \\
        --model Qwen/Qwen2-VL-7B-Instruct \\
        --n 10 --seed 42
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------
# Questions.  Each is asked in its own fresh turn with no history, so the
# model cannot chain hints across questions.  The free-identification prompt
# catches artist/style volunteered without being asked.
# --------------------------------------------------------------------------
QUESTIONS: dict[str, str] = {
    "artist": (
        "Who is the artist that painted this artwork? "
        "Answer with only the artist's full name and nothing else. "
        "If you are not sure, answer exactly: UNKNOWN."
    ),
    "year": (
        "In which year was this artwork created? "
        "Answer with only a four-digit year and nothing else. "
        "If you are not sure, answer exactly: UNKNOWN."
    ),
    "title": (
        "What is the title of this artwork? "
        "Answer with only the title and nothing else. "
        "If you are not sure, answer exactly: UNKNOWN."
    ),
    "free_identify": (
        "Identify this specific painting. If you recognize it, give the "
        "artist, the title and the year. If you do not recognize it, answer "
        "exactly: UNKNOWN."
    ),
}

YEAR_RE = re.compile(r"(1[0-9]{3}|20[0-2][0-9])")
WORD_RE = re.compile(r"[a-z0-9]+")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip accents and collapse to alphanumeric words."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(WORD_RE.findall(text.lower()))


def parse_ground_truth(relative_path: str, artist_field: str) -> dict:
    """Derive artist / title / year from the WikiArt file name and metadata.

    File names look like ``Style/artist-name_title-with-year-1889.jpg``.
    """
    stem = Path(relative_path).stem            # artist-name_title-year
    artist_slug, _, title_slug = stem.partition("_")
    year_match = YEAR_RE.search(title_slug) or YEAR_RE.search(stem)
    year = int(year_match.group(0)) if year_match else None
    # Title without the trailing year token, hyphens -> spaces.
    title = title_slug
    if year is not None:
        title = title.replace(str(year), " ")
    title = title.replace("-", " ").strip()
    # Prefer the qualified artist column from classes.csv when present.
    artist = artist_field if artist_field and str(artist_field) != "nan" else artist_slug.replace("-", " ")
    return {
        "gt_artist": normalize(artist),
        "gt_artist_raw": str(artist),
        "gt_title": normalize(title),
        "gt_title_raw": title,
        "gt_year": year,
    }


def build_cohort(
    wikiart_root: Path, n: int, seed: int, subset: str | None
) -> pd.DataFrame:
    """Pin a reproducible, style-stratified sample of ``n`` images."""
    classes_csv = wikiart_root / "classes.csv"
    if not classes_csv.is_file():
        raise FileNotFoundError(f"classes.csv not found under {wikiart_root}")
    meta = pd.read_csv(classes_csv)
    meta = meta[meta["filename"].apply(lambda f: (wikiart_root / f).is_file())]
    if subset and "subset" in meta.columns:
        meta = meta[meta["subset"].astype(str) == subset]
    if meta.empty:
        raise ValueError("No resolvable images after filtering")
    meta = meta.copy()
    meta["style"] = meta["filename"].apply(lambda f: str(Path(f).parts[0]))
    # Deterministic round-robin over styles so 10 images span many styles.
    shuffled = meta.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    picked_idx: list[int] = []
    seen_styles: set[str] = set()
    for idx, style in shuffled["style"].items():
        if len(picked_idx) >= n:
            break
        if style in seen_styles:
            continue
        seen_styles.add(style)
        picked_idx.append(idx)
    # If fewer distinct styles than n, top up with any remaining rows.
    if len(picked_idx) < n:
        for idx in shuffled.index:
            if len(picked_idx) >= n:
                break
            if idx not in picked_idx:
                picked_idx.append(idx)
    rows = []
    for idx in picked_idx[:n]:
        row = shuffled.loc[idx]
        rel = str(row["filename"])
        gt = parse_ground_truth(rel, row.get("artist", ""))
        rows.append({
            "stem": Path(rel).stem,
            "style": row["style"],
            "relative_path": rel,
            "path": str(wikiart_root / rel),
            "phash": row.get("phash", ""),
            "subset": row.get("subset", ""),
            **gt,
        })
    return pd.DataFrame(rows).sort_values(["style", "stem"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def fuzzy(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def artist_hit(answer: str, gt_artist: str) -> bool:
    """Artist counts as recalled if the GT surname/full name is in the answer
    or the two normalized strings are highly similar."""
    ans = normalize(answer)
    if not gt_artist or not ans or "unknown" in ans.split():
        return False
    gt_tokens = [t for t in gt_artist.split() if len(t) > 2]
    # Surname (last meaningful token) present verbatim in the answer.
    if gt_tokens and gt_tokens[-1] in ans.split():
        return True
    if fuzzy(ans, gt_artist) >= 0.80:
        return True
    return False


def year_match(answer: str, gt_year: int | None) -> tuple[bool, bool, int | None]:
    """Return (exact, within_1, predicted_year)."""
    if gt_year is None:
        return False, False, None
    match = YEAR_RE.search(answer)
    if not match:
        return False, False, None
    pred = int(match.group(0))
    return pred == gt_year, abs(pred - gt_year) <= 1, pred


def title_overlap(answer: str, gt_title: str) -> float:
    ans = set(normalize(answer).split())
    gt = set(gt_title.split())
    if not gt or "unknown" in ans:
        return 0.0
    return len(ans & gt) / len(gt)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
def encode_image_data_url(path: str, max_side: int = 768) -> str:
    import io

    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class OpenAIBackend:
    """Query an OpenAI-compatible /chat/completions endpoint (e.g. vLLM)."""

    def __init__(self, base_url: str, api_key: str, model: str, max_side: int):
        import requests  # noqa: F401  (fail fast if missing)

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_side = max_side

    def ask(self, image_path: str, prompt: str) -> str:
        import requests

        data_url = encode_image_data_url(image_path, self.max_side)
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class TransformersBackend:
    """Load Qwen2-VL locally.  Needs a free GPU and cached weights."""

    def __init__(self, model_id: str, max_side: int):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.max_side = max_side
        visual_pixels = 768 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=visual_pixels
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

    def ask(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to("cuda")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikiart-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, help="Reuse a pinned cohort CSV instead of sampling")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", default="test", help="WikiArt subset to sample from (test/train); empty for any")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--backend", choices=["openai", "transformers"], default="openai")
    parser.add_argument("--base-url", default="http://10.200.200.3:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--dry-run", action="store_true", help="Build cohort + ground truth only; no model calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.cohort and args.cohort.is_file():
        cohort = pd.read_csv(args.cohort)
        log(f"Loaded pinned cohort: {len(cohort)} images from {args.cohort}")
    else:
        cohort = build_cohort(
            args.wikiart_root, args.n, args.seed,
            args.subset or None,
        )
        cohort_path = args.out_dir / "leak_probe_qwen_cohort.csv"
        cohort.to_csv(cohort_path, index=False)
        log(f"Pinned cohort of {len(cohort)} images -> {cohort_path}")

    for _, row in cohort.iterrows():
        log(f"  {row['style']:<28} {row['gt_artist_raw']:<24} "
            f"y={row['gt_year']}  {row['gt_title_raw']}")

    if args.dry_run:
        log("Dry run: ground truth built, no model queried.")
        return

    if args.backend == "openai":
        api_key = os.environ.get(args.api_key_env, "")
        backend = OpenAIBackend(args.base_url, api_key, args.model, args.max_side)
        log(f"Backend: OpenAI endpoint {args.base_url} model={args.model}")
    else:
        backend = TransformersBackend(args.model, args.max_side)
        log(f"Backend: transformers local model={args.model}")

    raw_path = args.out_dir / "leak_probe_qwen_raw.jsonl"
    result_rows = []
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for _, row in cohort.iterrows():
            answers: dict[str, str] = {}
            for field, prompt in QUESTIONS.items():
                try:
                    answer = backend.ask(row["path"], prompt)
                except Exception as exc:  # noqa: BLE001
                    answer = f"ERROR: {type(exc).__name__}: {exc}"
                answers[field] = answer
                raw_handle.write(json.dumps({
                    "stem": row["stem"], "style": row["style"],
                    "field": field, "prompt": prompt, "answer": answer,
                    "gt_artist": row["gt_artist_raw"],
                    "gt_title": row["gt_title_raw"], "gt_year": row["gt_year"],
                }, ensure_ascii=False) + "\n")
                raw_handle.flush()

            gt_year = None if pd.isna(row["gt_year"]) else int(row["gt_year"])
            a_hit = artist_hit(answers["artist"], row["gt_artist"]) \
                or artist_hit(answers["free_identify"], row["gt_artist"])
            y_exact, y_near, y_pred = year_match(answers["year"], gt_year)
            fy_exact, fy_near, _ = year_match(answers["free_identify"], gt_year)
            t_overlap = max(
                title_overlap(answers["title"], row["gt_title"]),
                title_overlap(answers["free_identify"], row["gt_title"]),
            )
            result_rows.append({
                "stem": row["stem"], "style": row["style"],
                "gt_artist": row["gt_artist_raw"], "gt_year": gt_year,
                "gt_title": row["gt_title_raw"],
                "ans_artist": answers["artist"], "ans_year": answers["year"],
                "ans_title": answers["title"], "ans_free": answers["free_identify"],
                "artist_hit": a_hit,
                "year_exact": y_exact or fy_exact,
                "year_within1": y_near or fy_near,
                "year_pred": y_pred,
                "title_overlap": round(t_overlap, 3),
                "title_hit": t_overlap >= 0.5,
            })
            log(f"{row['stem'][:40]:<40} artist={'Y' if a_hit else '.'} "
                f"year={'Y' if (y_exact or fy_exact) else '.'} "
                f"title={t_overlap:.2f}")

    results = pd.DataFrame(result_rows)
    results_path = args.out_dir / "leak_probe_qwen_results.csv"
    results.to_csv(results_path, index=False)

    summary = pd.DataFrame([
        {"field": "artist", "metric": "recall", "value": results["artist_hit"].mean()},
        {"field": "year", "metric": "exact", "value": results["year_exact"].mean()},
        {"field": "year", "metric": "within_1", "value": results["year_within1"].mean()},
        {"field": "title", "metric": "hit@0.5", "value": results["title_hit"].mean()},
        {"field": "title", "metric": "mean_overlap", "value": results["title_overlap"].mean()},
    ])
    summary_path = args.out_dir / "leak_probe_qwen_summary.csv"
    summary.to_csv(summary_path, index=False)

    log(f"Saved {results_path}, {summary_path}, {raw_path}")
    log("Aggregate leakage (recall of ground truth from pixels alone):")
    for _, srow in summary.iterrows():
        log(f"  {srow['field']:<7} {srow['metric']:<13} {srow['value']:.3f}")


if __name__ == "__main__":
    main()
