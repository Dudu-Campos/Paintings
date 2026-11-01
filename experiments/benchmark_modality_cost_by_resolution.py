"""Custo por imagem de cada modalidade contra a resolução de entrada.

Complementa o custo medido em dataset (10,6 h CPU no Multitask), que mistura
complexidade do algoritmo com a distribuição de tamanhos das imagens daquele
corpus. Aqui o conteúdo é fixo (as mesmas 100 pinturas) e só N varia, então o
que sobra é a escala do algoritmo -- um número transferível para quem tem
imagens de outro tamanho.

**Pinturas reais redimensionadas, não ruído.** Segmentação é fortemente
dependente de conteúdo: em ruído branco o Felzenszwalb produz uma contagem de
regiões que não se parece com nada real, e o custo do grafo depende do tamanho
do RAG. Ruído mediria um regime que nunca acontece.

O que se espera medir, e que é a razão do experimento: o pipeline é **limitado
em resolução por construção** -- Descriptors/Gabor/Hilbert reamostram para
128 px, Fourier para 256, os multigrafos para lado máximo 1024. Acima disso o
custo do algoritmo é constante e só a decodificação do JPEG cresce. É uma
diferença estrutural em relação a uma CNN, cujo custo acompanha a resolução.

`--workers 1` em todos os extratores: com mais de um worker mediríamos vazão de
escalonamento, não custo por imagem.

**Dois pontos por medição, não um.** Cada extrator paga ~7 s fixos de import do
Python antes de tocar em imagem alguma. Com 100 imagens isso são 70 ms/imagem
de contaminação, o bastante para inverter a ordem entre resoluções (visto no
teste: 388 ms a 128 px contra 74 ms a 512 px, puro overhead). Medindo com
`n_base` e `n_full` imagens e tomando a inclinação
`(t_full - t_base) / (n_full - n_base)`, o custo fixo cancela por construção.

**Contagem de imagens por passo, e mínimo sobre repetições.** O jitter de
lançar o subprocesso é ~1 s. Para SLIC/Felzenszwalb (~0,7 s por imagem) isso é
irrelevante, mas para Gabor (~2 ms) engole o sinal inteiro -- na primeira
tentativa saiu custo NEGATIVO. Então as modalidades leves rodam com centenas de
imagens e as pesadas com dezenas, e cada medição é o **mínimo** de 3
repetições, que é o estimador padrão para tempo (a execução menos contaminada
por escalonamento alheio).
"""
from __future__ import annotations

import os

# §1: travar BLAS antes de qualquer import pesado. O eixo da figura diz "1 core";
# sem isto nada garante isso -- `--workers 1` limita o Pool, nao as threads de
# OpenBLAS/MKL dentro do numpy. As variaveis sao herdadas pelos subprocessos dos
# extratores, que e onde o tempo e medido.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
from PIL import Image

# A coorte tem varreduras legítimas de centenas de megapixels (uma delas com
# 669 MP), que a proteção anti-bomba do PIL recusa por padrão. Só as
# redimensionamos para um lado fixo, então o tamanho da origem não entra na
# medição -- liberar o limite aqui é seguro e não vaza para o pipeline.
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".ABSvenv/bin/python3"

# nome -> (script, argumentos extras). Todos recebem --image_dir/--image-dir,
# --meta e uma saída; a diferença de convenção (underscore x hífen) é histórica.
# passo -> (n_base, n_full). As leves precisam de muitas imagens para o sinal
# superar o jitter do subprocesso; as pesadas, de poucas para não levar horas.
COUNTS = {
    "Descriptors": (150, 600),
    "Hilbert": (150, 600),
    "Gabor": (150, 600),
    "Fourier": (150, 600),
    "SLIC": (15, 60),
    "Felzenszwalb": (15, 60),
}

# O quarto campo e a FLAG DE TAMANHO de cada extrator. Sem ela o benchmark media
# I/O, nao complexidade: `extract_descriptors_full` (TARGET=128),
# `extract_hilbert_features` (N_SIZE=128) e `extract_gabor_features` (TARGET=128)
# reduzem a imagem internamente, entao rodar sobre um JPEG de 2048 media decode +
# downscale e nao o custo da representacao. Os alfa de 0,07 / 0,74 / 0,46 que a
# figura publicava para essas tres eram expoentes de leitura de arquivo.
STEPS = {
    "Descriptors": ("src/extractors/extract_descriptors_full.py", "underscore", [], "--target"),
    "Hilbert": ("src/extractors/extract_hilbert_features.py", "underscore", [], "--n-size"),
    "Gabor": ("src/extractors/extract_gabor_features.py", "underscore", [], "--target"),
    # O Fourier tem convencao propria: exige `--sample` (CSV com stem+style) alem
    # de `--meta`, e grava num arquivo (`--output`), nao num diretorio. O meta que
    # o benchmark materializa ja tem stem/filename/style, entao serve de sample.
    "Fourier": ("src/extractors/extract_fourier_full_features.py", "fourier", [], "--size"),
    "SLIC": ("src/extractors/segmentation_multigraph_features.py", "dash",
             ["--method", "slic", "--segments", "2048", "--compactness", "10.0",
              "--sigma", "0.0", "--k", "10", "--sample-per-style", "0"], "--max-side"),
    "Felzenszwalb": ("src/extractors/segmentation_multigraph_features.py", "dash",
                     ["--method", "felzenszwalb", "--felz-scale", "50.0",
                      "--felz-min-size", "50", "--felz-sigma", "0.8",
                      "--segments", "2048", "--k", "10",
                      "--sample-per-style", "0"], "--max-side"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_cohort(meta: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    sample = meta.sample(n=count, random_state=seed).reset_index(drop=True)
    return sample


def materialise(sample: pd.DataFrame, side: int, base: Path) -> Path:
    """Escreve as imagens redimensionadas em <style>/<filename>, que é o layout
    que `resolve_wikiart_records` indexa."""
    target = base / f"n{side}"
    if target.exists():
        shutil.rmtree(target)
    for row in sample.itertuples():
        directory = target / str(row.style)
        directory.mkdir(parents=True, exist_ok=True)
        with Image.open(row.path) as image:
            image.convert("RGB").resize((side, side), Image.BILINEAR).save(
                directory / row.filename, quality=92)
    return target


def run_once(name: str, image_dir: Path, meta_path: Path, output: Path,
             side: int, seg_max_side: int | None) -> float:
    script, convention, extra, size_flag = STEPS[name]
    # `seg_max_side=None` (o padrao) faz cada extrator operar NA RESOLUCAO DO
    # ponto, que e o que a figura afirma medir. Fixar o cap em 1024 achatava
    # Felz e SLIC em 2048 -- aquele trecho media o cap, nao saturacao.
    target = side if seg_max_side is None else min(side, seg_max_side)
    extra = [*extra, size_flag, str(target)]
    if convention == "fourier":
        command = [str(PYTHON), script, "--sample", str(meta_path),
                   "--meta", str(meta_path), "--image-dir", str(image_dir),
                   "--output", str(output) + ".parquet", "--workers", "1", *extra]
    elif convention == "underscore":
        command = [str(PYTHON), script, "--image_dir", str(image_dir),
                   "--meta", str(meta_path), "--output", str(output),
                   "--workers", "1", *extra]
    else:
        command = [str(PYTHON), script, "--image-dir", str(image_dir),
                   "--meta", str(meta_path), "--output-dir", str(output),
                   "--workers", "1", *extra]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        log(f"    ! {name} falhou: {result.stderr.strip().splitlines()[-1][:120]}")
        return float("nan")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--image-root", type=Path,
                        default=ROOT / "data/multitask_painting100k/original/images")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seg-max-side", type=int, default=None,
                        help="teto de reamostragem da segmentação. O default é "
                             "o de produção; passar um valor >= ao maior lado "
                             "testado mede o custo do ALGORITMO sem o teto, "
                             "que é o que separa complexidade de escolha de "
                             "engenharia.")
    parser.add_argument("--sides", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    parser.add_argument("--work-dir", type=Path,
                        default=ROOT / "outputs/resolution_benchmark")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/modality_cost_by_resolution.csv")
    args = parser.parse_args()

    meta = pd.read_csv(args.meta)
    meta["path"] = (str(args.image_root) + "/" + meta["style"].astype(str)
                    + "/" + meta["filename"].astype(str))
    needed = max(COUNTS[name][1] for name in args.steps)
    sample = build_cohort(meta, needed, args.seed)
    log(f"{len(sample)} imagens materializadas, lados {args.sides}, passos {args.steps}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    meta_paths = {}
    for name in args.steps:
        for count in COUNTS[name]:
            if count in meta_paths:
                continue
            path = args.work_dir / f"meta_{count}.csv"
            sample.head(count)[["stem", "filename", "style"]].to_csv(path, index=False)
            meta_paths[count] = path

    done = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        done = {(str(r.step), int(r.side)) for r in previous.itertuples()}
        log(f"retomando: {len(done)} medições já feitas")

    for side in args.sides:
        image_dir = materialise(sample, side, args.work_dir)
        pixels = sum(p.stat().st_size for p in image_dir.rglob("*.jpg"))
        log(f"lado={side}: {pixels / 1e6:.1f} MB no disco")
        for name in args.steps:
            if (name, side) in done:
                continue
            stem = name.replace("(", "").replace(")", "")
            n_base, n_full = COUNTS[name]
            timings = {}
            for tag, count in (("base", n_base), ("full", n_full)):
                runs = []
                for repeat in range(args.repeats):
                    output = args.work_dir / f"out_{stem}_{side}_{tag}"
                    shutil.rmtree(output, ignore_errors=True)
                    runs.append(run_once(name, image_dir, meta_paths[count],
                                         output, side, args.seg_max_side))
                    shutil.rmtree(output, ignore_errors=True)
                timings[tag] = min(runs)
            per_image = 1000 * (timings["full"] - timings["base"]) / (n_full - n_base)
            overhead = timings["base"] - (timings["full"] - timings["base"]) \
                * n_base / (n_full - n_base)
            pd.DataFrame([{"step": name, "side": side,
                           "n_base": n_base, "n_full": n_full,
                           "seg_max_side": args.seg_max_side,
                           "seconds_base": timings["base"],
                           "seconds_full": timings["full"],
                           "fixed_overhead_s": overhead,
                           "ms_per_image": per_image}]).to_csv(
                args.output, mode="a", index=False,
                header=not args.output.exists(), float_format="%.4f")
            log(f"  {name:<14} lado={side:>5}: {per_image:>8.1f} ms/imagem "
                f"(overhead fixo {overhead:.1f}s)")
        shutil.rmtree(image_dir, ignore_errors=True)

    table = pd.read_csv(args.output).pivot_table(
        index="step", columns="side", values="ms_per_image")
    print()
    print("ms por imagem (1 worker):")
    print(table.round(1).to_string())
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
