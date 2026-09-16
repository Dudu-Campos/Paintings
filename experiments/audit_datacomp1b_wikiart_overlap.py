"""Audit DataComp-1B captions for WikiArt overlap -- pretraining-corpus leak probe.

``clip_zeroshot_style.py`` showed that a CLIP checkpoint pretrained on
DataComp-1B (``datacomp_xl_s13b_b90k``) guesses WikiArt style *zero-shot*
even better than the original OpenAI/WIT checkpoint. That is consistent
with (but not proof of) WikiArt itself -- or captions naming its artists
and styles -- being present in DataComp-1B's training corpus. This script
tests that directly: it scans the ``text`` (caption) column of every
DataComp-1B parquet shard on the Hugging Face Hub
(``mlfoundations/datacomp_1b``, 2,664 shards, 1.39B rows) for any of the
1,119 WikiArt artist names (``raw/wikiart/classes.csv``) or the style
aliases from ``experiments/evaluate_catr_coco.py::STYLE_ALIASES`` (mirrored
here, same convention as ``leak_probe_qwen.py`` and ``clip_zeroshot_style.py``).

Only the ``text`` column is fetched per shard (DuckDB column-projection
pushdown over HTTP range requests against the remote parquet files) --
~75GB total across all shards, not the full 340GB of metadata (url,
sha256, face_bboxes, ...) and nowhere near the 8.5TB of precomputed CLIP
embedding ``.npz`` siblings, neither of which this script touches.

A 3-shard smoke test showed why artist/style keywords cannot be treated as
a single tier: at 1.39B rows, generic terms ("baroque", "minimalist",
"realism") and even short artist mononyms ("afro" -> "Afro Futurism")
produce ~1,400 matches per shard (~3.7M extrapolated over all shards) that
are mostly unrelated to WikiArt -- famous painters' names circulate all
over the open web independent of WikiArt as a site. Dumping full rows for
that tier would be neither storable nor reviewable, and it is not the
question this script asks anyway.

So two tiers, reported differently:
  * ``--out-literal``  -- exact "wikiart" / "wikiart.org" / "wiki art"
    mentions. Rare and highly diagnostic: a hit here means the crawler
    ingested a WikiArt page directly. Full uid/text/shard rows kept.
  * ``--out-keywords``  -- per-artist/per-style-alias match *counts* only
    (plus up to 3 example captions each, for spot-checking), accumulated
    across all shards. This is background signal about how much generic
    art content the pool contains, not evidence of WikiArt-specific leak.

Usage:
    python experiments/audit_datacomp1b_wikiart_overlap.py \\
        --wikiart-root raw/wikiart [--limit-shards N] [--start-shard N]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

HF_TREE_API = (
    "https://huggingface.co/api/datasets/mlfoundations/datacomp_1b/tree/main?limit=1000"
)
HF_RESOLVE_BASE = "https://huggingface.co/datasets/mlfoundations/datacomp_1b/resolve/main/"

# Mirrors experiments/evaluate_catr_coco.py::STYLE_ALIASES (see also
# leak_probe_qwen.py and clip_zeroshot_style.py, which use the same list).
STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "Abstract_Expressionism": ("abstract expressionism", "abstract expressionist"),
    "Action_painting": ("action painting",),
    "Analytical_Cubism": ("analytical cubism", "analytic cubism"),
    "Art_Nouveau_Modern": ("art nouveau",),
    "Baroque": ("baroque",),
    "Color_Field_Painting": ("color field painting", "colour field painting"),
    "Contemporary_Realism": ("contemporary realism",),
    "Cubism": ("cubism", "cubist"),
    "Early_Renaissance": ("early renaissance",),
    "Expressionism": ("expressionism", "expressionist"),
    "Fauvism": ("fauvism", "fauvist"),
    "High_Renaissance": ("high renaissance",),
    "Impressionism": ("impressionism", "impressionist"),
    "Mannerism_Late_Renaissance": ("mannerism", "mannerist", "late renaissance"),
    "Minimalism": ("minimalism", "minimalist"),
    "Naive_Art_Primitivism": ("naive art", "primitive art", "primitivism"),
    "New_Realism": ("new realism",),
    "Northern_Renaissance": ("northern renaissance",),
    "Pointillism": ("pointillism", "pointillist"),
    "Pop_Art": ("pop art",),
    "Post_Impressionism": ("post impressionism", "post impressionist"),
    "Realism": ("realism", "realist"),
    "Rococo": ("rococo",),
    "Romanticism": ("romanticism",),
    "Surrealism": ("surrealism", "surrealist"),
    "Symbolism": ("symbolism", "symbolist"),
    "Synthetic_Cubism": ("synthetic cubism",),
    "Ukiyo_e": ("ukiyo e", "ukiyo-e"),
}

LITERAL_KEYWORDS = ("wikiart", "wiki art", "wiki-art")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def list_shards() -> list[str]:
    """Paginate the HF dataset-tree API for every .parquet shard path."""
    paths: list[str] = []
    url = HF_TREE_API
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "paintings-leak-audit/1.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            link_header = resp.headers.get("Link", "")
        paths.extend(e["path"] for e in data if e["path"].endswith(".parquet"))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        url = m.group(1) if m else None
    return sorted(paths)


def wikiart_keywords(wikiart_root: Path) -> list[str]:
    classes = pd.read_csv(wikiart_root / "classes.csv")
    artists = classes["artist"].dropna().astype(str).str.lower().str.strip().unique().tolist()
    styles = {a.lower() for aliases in STYLE_ALIASES.values() for a in aliases}
    return sorted(set(artists) | styles)


def build_pattern(keywords: list[str]) -> str:
    return r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikiart-root", type=Path, required=True)
    ap.add_argument("--out-literal", type=Path,
                     default=ROOT / "results" / "datacomp1b_wikiart_literal_hits.csv")
    ap.add_argument("--out-keywords", type=Path,
                     default=ROOT / "results" / "datacomp1b_wikiart_keyword_counts.csv")
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all remaining shards")
    args = ap.parse_args()

    keywords = wikiart_keywords(args.wikiart_root)
    kw_pattern = build_pattern(keywords)
    literal_pattern = build_pattern(list(LITERAL_KEYWORDS))
    log(f"{len(keywords)} artist/style keywords (pattern {len(kw_pattern)} chars); "
        f"{len(LITERAL_KEYWORDS)} literal terms {LITERAL_KEYWORDS}")

    shards = list_shards()
    log(f"{len(shards)} DataComp-1B parquet shards discovered on the Hub")
    end = args.start_shard + args.limit_shards if args.limit_shards else len(shards)
    shards = shards[args.start_shard:end]
    log(f"scanning shards [{args.start_shard}:{end}) ({len(shards)} shards)")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false")

    args.out_literal.parent.mkdir(parents=True, exist_ok=True)
    write_literal_header = not args.out_literal.exists()

    kw_counts: dict[str, int] = {}
    kw_examples: dict[str, list[str]] = {}
    total_literal, total_kw = 0, 0
    t0 = time.time()

    for i, shard in enumerate(shards):
        url = HF_RESOLVE_BASE + shard
        try:
            lit_rows = con.execute(
                "SELECT uid, text FROM read_parquet($url) "
                "WHERE regexp_matches(lower(text), $pattern)",
                {"pattern": literal_pattern, "url": url},
            ).fetchall()
            count_rows = con.execute(
                "SELECT kw, COUNT(*), FIRST(text) FROM ("
                "  SELECT UNNEST(regexp_extract_all(lower(text), $pattern)) AS kw, text "
                "  FROM read_parquet($url) WHERE regexp_matches(lower(text), $pattern)"
                ") GROUP BY kw",
                {"pattern": kw_pattern, "url": url},
            ).fetchall()
        except Exception as exc:  # noqa: BLE001 -- keep scanning past transient HTTP errors
            log(f"  [{i + 1}/{len(shards)}] {shard} FAILED: {exc}")
            continue

        if lit_rows:
            df = pd.DataFrame(lit_rows, columns=["uid", "text"])
            df["shard"] = shard
            df.to_csv(args.out_literal, mode="a", header=write_literal_header, index=False)
            write_literal_header = False
            total_literal += len(lit_rows)
            log(f"  *** literal 'wikiart' hit(s) in {shard}: {len(lit_rows)} ***")

        for kw, count, example in count_rows:
            kw_counts[kw] = kw_counts.get(kw, 0) + count
            total_kw += count
            if len(kw_examples.setdefault(kw, [])) < 3:
                kw_examples[kw].append(example)

        if (i + 1) % 20 == 0 or i == len(shards) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(shards) - i - 1)
            log(f"  [{i + 1}/{len(shards)}] shards scanned  literal={total_literal}  "
                f"keyword_hits={total_kw}  {elapsed / 60:.1f}min elapsed  ETA {eta / 60:.0f}min")

        if (i + 1) % 100 == 0 or i == len(shards) - 1:
            kw_df = pd.DataFrame(
                [{"keyword": k, "count": c, "examples": " | ".join(kw_examples.get(k, []))}
                 for k, c in sorted(kw_counts.items(), key=lambda x: -x[1])]
            )
            kw_df.to_csv(args.out_keywords, index=False)

    log(f"Done. literal 'wikiart' hits={total_literal} -> {args.out_literal}  "
        f"keyword occurrences={total_kw} ({len(kw_counts)} distinct) -> {args.out_keywords}")


if __name__ == "__main__":
    main()
