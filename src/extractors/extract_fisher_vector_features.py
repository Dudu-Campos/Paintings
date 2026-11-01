#!/usr/bin/env python3
"""Fisher Vector sobre SIFT denso -- Sanchez, Perronnin, Mensink & Verbeek
(IJCV 105(3), 2013, "Image Classification with the Fisher Vector: Theory and
Practice").

POR QUE ESTE BANCO EXISTE NA COMPARACAO. A matriz de bancos hand-designed tinha
Haralick (1973), HOG (2005) e WND-CHARM (2010) e **omitia o metodo que era
estado da arte imediatamente antes das CNNs**. Pior: o §7.11b registra, da
Tabela 2 dos proprios autores do Food-101, `IFV = 38,88` contra o nosso
`hand-8` em 36,74 na mesma banda `Global` -- ou seja ja ha numero publicado em
que o FV nos vence. Deixar de fora o unico classico que nos bate seria o
movimento espelhado do que o §7.11 proibe ("citar so a CNN nos deixaria
melhor"). Entra por honestidade de baseline, nao por expectativa de ganho.

Segundo motivo: o FV agrega patches LOCAIS densos, entao cai na banda `Local`
da taxonomia do §7.11b, como os tiles de 150x150 do WND-CHARM, enquanto o
nosso banco e global. Ele e o segundo ponto daquele eixo.

A RECEITA, e cada escolha vai no meta.json porque todas movem o numero:
  1. SIFT denso em `--scales` diametros de keypoint, passo `--step`;
  2. PCA 128 -> `--pca-dim` (64 no artigo), ajustada em amostra;
  3. GMM diagonal de `--gmm-k` componentes (256 no artigo), ajustada em amostra;
  4. gradientes em relacao a media E variancia -> 2*K*D dims (32.768 no padrao);
  5. normalizacao de POTENCIA (sqrt assinada) e depois L2 -- e o "improved" do
     titulo, e sem ela o FV perde vários pontos.

**A PCA E A GMM SAO AJUSTADAS SO NO TREINO quando o split e publicado**
(`--fit-split train`). Elas nao veem rotulo, entao ajustar na coorte inteira
seria apenas transdutivo e nao vazamento -- mas o §10.6 registra essa ressalva
em aberto noutro experimento, e aqui ela custa nada para fechar.

**RESOLUCAO: lado maior `--max-side` (512), SEM recorte.** Escolha declarada, e
diferente do WND-CHARM de proposito: lá o protocolo publicado manda recortar
600x600 e isso descarta ~26% da area em imagem nao-quadratica (§5.19). O artigo
do FV nao especifica recorte, entao nao recortar e a leitura que nao joga
conteudo fora. 512 e a resolucao nativa do Food-101, onde esta o numero
publicado que serve de validacao externa desta implementacao.

Saida: `outputs/{dataset}_fisher_vector/` com `features.npy` (n, 2*K*D) float32,
`stems.npy`, `meta.json`. Retomavel por shard.

Uso:
    python src/extractors/extract_fisher_vector_features.py \
        --dataset food101 --image-dir data/food101/food-101/images --workers 28
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import zlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_wndcharm_tiled_features import build_stem_map, log  # noqa: E402

SHARD = 1000
_W: dict = {}


def cinza(path: str, max_side: int) -> np.ndarray:
    """Lado maior -> max_side, sem recorte. PIL para decodificar (robusto a
    CMYK/paleta), OpenCV para o resto."""
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        escala = max_side / max(w, h)
        if escala < 1.0:
            im = im.resize((max(1, round(w * escala)), max(1, round(h * escala))),
                           Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def sift_denso(img: np.ndarray, step: int, scales: tuple[int, ...],
               sift) -> tuple[np.ndarray, np.ndarray]:
    """Grade de keypoints em varios diametros.

    Devolve `(T, 128)` float32 E as posicoes `(T, 2)` em coordenada RELATIVA
    `(y/h, x/w)`. As posicoes existem para a piramide espacial; em coordenada
    relativa porque a celula e fracao da imagem, e assim o mesmo codigo serve a
    qualquer resolucao. O `sift.compute` pode DESCARTAR keypoint (borda), entao
    a posicao tem de sair da lista que ele devolve, nunca da que entrou.
    """
    h, w = img.shape
    pontos = []
    for d in scales:
        raio = d / 2.0
        ys = np.arange(raio, h - raio, step)
        xs = np.arange(raio, w - raio, step)
        pontos += [cv2.KeyPoint(float(x), float(y), float(d))
                   for y in ys for x in xs]
    vazio = (np.empty((0, 128), dtype=np.float32), np.empty((0, 2), dtype=np.float32))
    if not pontos:
        return vazio
    kps, desc = sift.compute(img, pontos)
    if desc is None:
        return vazio
    pos = np.array([[k.pt[1] / h, k.pt[0] / w] for k in kps], dtype=np.float32)
    return desc, pos


def aumenta_xy(desc: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Anexa a posicao normalizada ao descritor JA projetado pela PCA.

    Chatfield, Simonyan, Vedaldi & Zisserman (BMVC 2014), §3.1: o descritor
    `x_i` e estendido, DEPOIS da projecao, com `[x_i, x_i/W - 0.5, y_i/H - 0.5]`.
    Como a GMM passa a quantizar aparencia E posicao, a informacao espacial
    entra pela propria soft-quantization. Na tabela deles isto vale 41.984 dims
    contra as 327.680 da piramide de 8 celulas e MELHORA o resultado
    (64,36 contra 64,18 de mAP no VOC-2007, linhas (d) e (b)).

    `pos` vem de `sift_denso` em `(y/h, x/w)`; a ordem deles e x depois y.
    """
    if len(desc) == 0:
        return np.empty((0, desc.shape[1] + 2), dtype=desc.dtype)
    return np.hstack([desc, (pos[:, 1:2] - 0.5), (pos[:, 0:1] - 0.5)])


def parse_piramide(spec: str) -> list[tuple[int, int]]:
    """'1x1+3x1' -> [(1,1), (3,1)]: RxC = linhas x colunas. `3x1` sao tres
    FAIXAS HORIZONTAIS, que e o layout de Sanchez et al. 2013 (1x1 + 3x1)."""
    niveis = []
    for parte in spec.split("+"):
        r, c = parte.lower().split("x")
        niveis.append((int(r), int(c)))
    return niveis


def celulas(pos: np.ndarray, niveis: list[tuple[int, int]]) -> list[np.ndarray]:
    """Indice de celula de cada descritor, um vetor por nivel."""
    saida = []
    for r, c in niveis:
        if r == 1 and c == 1:
            saida.append(np.zeros(len(pos), dtype=np.int32))
            continue
        li = np.clip((pos[:, 0] * r).astype(np.int32), 0, r - 1)
        co = np.clip((pos[:, 1] * c).astype(np.int32), 0, c - 1)
        saida.append(li * c + co)
    return saida


def raiz_sift(desc: np.ndarray) -> np.ndarray:
    """RootSIFT (Arandjelovic & Zisserman 2012): L1 e depois sqrt. O artigo do FV
    usa L2 + sqrt como 'improved'; a variante por L1 e a canonica e e a que o
    `--rootsift` liga."""
    s = np.abs(desc).sum(axis=1, keepdims=True)
    return np.sqrt(desc / np.maximum(s, 1e-12))


def fisher(desc: np.ndarray, w: np.ndarray, mu: np.ndarray, var: np.ndarray,
           cel: list[np.ndarray] | None = None,
           n_cel: list[int] | None = None, intra: bool = False) -> np.ndarray:
    """FV improved, com piramide espacial opcional.

    Posteriores e momentos por MATMUL: materializar (T,K,D) custaria 640 MB por
    imagem com T=10k, K=256, D=64.

    **A PIRAMIDE NAO RE-EXTRAI NADA.** Os posteriores `g` saem UMA vez sobre os
    T descritores; cada nivel so reparticiona quem ja esta calculado, e somar
    sobre as celulas de um nivel custa o mesmo que o momento global. Medido em
    11/out: SIFT 99,0% do custo, momentos 0,3%, entao `1x1+3x1` sai por +0,3%.

    `cel[l][t]` = celula do descritor t no nivel l; `n_cel[l]` = quantas celulas
    aquele nivel tem. O numero vem da ESPECIFICACAO e nao do dado, senao uma
    imagem sem descritor numa faixa encurtaria o vetor em silencio.

    Normalizacao: potencia + L2 **por celula**, e L2 global no fim. E a pratica
    do VLFeat; sem o L2 por celula a faixa mais populosa domina a norma.

    Com `cel=None` (ou um nivel 1x1 unico) reproduz BIT A BIT a versao sem
    piramide -- conferido contra o cache de 101.000 imagens do food101.
    """
    K, D = mu.shape
    if cel is None:
        cel, n_cel = [np.zeros(len(desc), dtype=np.int32)], [1]
    assert n_cel is not None
    total = int(sum(n_cel))
    if len(desc) == 0:
        return np.zeros(2 * K * D * total, dtype=np.float32)

    X = desc.astype(np.float64)
    X2 = X * X
    inv = 1.0 / var                                        # (K, D)
    # log N(x | mu_k, var_k) sem o termo comum -0.5*D*log(2pi)
    const = -0.5 * (np.log(var).sum(axis=1) + (mu * mu * inv).sum(axis=1))
    logp = (np.log(w) + const)[None, :] \
        - 0.5 * (X2 @ inv.T) + (X @ (mu * inv).T)          # (T, K)
    logp -= logp.max(axis=1, keepdims=True)
    g = np.exp(logp)
    g /= g.sum(axis=1, keepdims=True)                      # (T, K)
    sig = np.sqrt(var)
    raiz_w = np.sqrt(w)[:, None]
    raiz_2w = np.sqrt(2.0 * w)[:, None]

    partes = []
    for nivel, quantas in zip(cel, n_cel):
        for c in range(quantas):
            m = nivel == c if quantas > 1 else slice(None)
            Xc, gc = X[m], g[m]
            T = len(Xc)
            if T == 0:
                partes.append(np.zeros(2 * K * D))
                continue
            X2c = X2[m]
            S0 = gc.sum(axis=0)
            S1 = gc.T @ Xc
            S2 = gc.T @ X2c
            g_mu = (S1 - mu * S0[:, None]) / (sig * T * raiz_w)
            g_var = (S2 - 2 * mu * S1 + (mu * mu - var) * S0[:, None]) \
                / (var * T * raiz_2w)
            if intra:
                # INTRA-NORMALIZACAO (Arandjelovic & Zisserman 2013 para VLAD,
                # adotada no §3.1 do Chatfield 2014): L2 em cada sub-bloco
                # (u_k, v_k) do vetor, para aliviar a burstiness de feature
                # local. O bloco k sao as linhas k de g_mu E de g_var juntas.
                nk = np.sqrt((g_mu * g_mu).sum(axis=1) + (g_var * g_var).sum(axis=1))
                nk = np.where(nk > 0, nk, 1.0)[:, None]
                g_mu, g_var = g_mu / nk, g_var / nk
            v = np.concatenate([g_mu.ravel(), g_var.ravel()])
            v = np.sign(v) * np.sqrt(np.abs(v))            # potencia (alpha=0,5)
            n = np.linalg.norm(v)
            partes.append(v / n if n > 0 else v)
    out = np.concatenate(partes)
    n = np.linalg.norm(out)
    return (out / n if n > 0 else out).astype(np.float32)


def _init(cfg: dict, pca: dict, gmm: dict) -> None:
    _W["sift"] = cv2.SIFT_create()
    _W["cfg"] = cfg
    _W["media"] = pca["media"]
    _W["comp"] = pca["comp"]
    _W["w"], _W["mu"], _W["var"] = gmm["w"], gmm["mu"], gmm["var"]


def descritores(path: str, cfg: dict, sift) -> tuple[np.ndarray, np.ndarray]:
    img = cinza(path, cfg["max_side"])
    d, pos = sift_denso(img, cfg["step"], tuple(cfg["scales"]), sift)
    if cfg["rootsift"] and len(d):
        d = raiz_sift(d)
    return d, pos


def _uma(task: tuple[str, str]) -> tuple[str, np.ndarray | None, str | None]:
    stem, path = task
    try:
        cfg = _W["cfg"]
        d, pos = descritores(path, cfg, _W["sift"])
        if len(d) > cfg["max_desc"]:
            rng = np.random.default_rng(zlib.crc32(stem.encode()))
            # a POSICAO acompanha o sorteio: reamostrar so `d` desalinharia a
            # piramide sem nada na saida denunciar (§5.18b)
            idx = rng.choice(len(d), cfg["max_desc"], replace=False)
            d, pos = d[idx], pos[idx]
        d = (d.astype(np.float64) - _W["media"]) @ _W["comp"].T
        if cfg["augment_xy"]:
            d = aumenta_xy(d, pos)
        niveis = cfg["piramide"]
        cel = celulas(pos, niveis)
        n_cel = [r * c for r, c in niveis]
        return stem, fisher(d, _W["w"], _W["mu"], _W["var"], cel, n_cel,
                            intra=cfg["intra_norm"]), None
    except Exception as erro:
        return stem, None, f"{type(erro).__name__}: {erro}"


def _amostra(task: tuple[str, str, int]) -> np.ndarray:
    """Devolve (n, 130): os 128 do SIFT MAIS a posicao relativa nas duas ultimas
    colunas. A posicao viaja junto porque a GMM e ajustada no espaco AUMENTADO
    quando `--augment-xy` esta ligado, e a PCA no espaco de 128 -- ajustar a GMM
    sem a posicao e depois codificar com ela daria um codebook de 80 dims para
    um descritor de 82, e o erro apareceria so como acuracia ruim."""
    path, _, por_img = task[1], task[0], task[2]
    d, pos = descritores(path, _W["cfg"], _W["sift"])
    if len(d) > por_img:
        rng = np.random.default_rng(zlib.crc32(path.encode()))
        idx = rng.choice(len(d), por_img, replace=False)
        d, pos = d[idx], pos[idx]
    if len(d) == 0:
        return np.empty((0, 130), dtype=np.float32)
    return np.hstack([d, pos]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--scales", default="16,24,32,48")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--gmm-k", type=int, default=256)
    ap.add_argument("--max-desc", type=int, default=10000)
    ap.add_argument("--piramide", default="1x1",
                    help="niveis RxC separados por '+', RxC = linhas x colunas. "
                         "'1x1' (default) reproduz a versao sem piramide BIT A "
                         "BIT; '1x1+3x1' e o layout de Sanchez et al. 2013 "
                         "(imagem inteira + tres faixas horizontais).")
    ap.add_argument("--augment-xy", action="store_true", default=False,
                    help="anexa a posicao normalizada (x/W-0.5, y/H-0.5) ao "
                         "descritor DEPOIS da PCA, como no §3.1 do Chatfield et "
                         "al. (BMVC 2014). A GMM passa a quantizar aparencia E "
                         "posicao; na tabela deles isto da 41.984 dims contra as "
                         "327.680 da piramide de 8 celulas e vai MELHOR.")
    ap.add_argument("--intra-norm", action="store_true", default=False,
                    help="L2 em cada sub-bloco (u_k, v_k) antes da potencia. Com "
                         "ela o artigo aplica a raiz UMA vez, nao duas.")
    ap.add_argument("--rootsift", action="store_true", default=True)
    ap.add_argument("--no-rootsift", dest="rootsift", action="store_false")
    ap.add_argument("--fit-images", type=int, default=2000,
                    help="imagens amostradas para ajustar PCA e GMM")
    ap.add_argument("--fit-desc-per-image", type=int, default=600)
    ap.add_argument("--fit-threads", type=int, default=32,
                    help="threads de BLAS no ajuste de PCA/GMM (processo principal)")
    ap.add_argument("--fit-split", default="train",
                    help="split em que PCA/GMM sao ajustadas, se o metadado tiver "
                         "a coluna; 'any' usa a coorte inteira")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-stems", default="")
    args = ap.parse_args()

    out = args.out_dir or ROOT / "outputs" / f"{args.dataset}_fisher_vector"
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    niveis = parse_piramide(args.piramide)
    n_celulas = sum(r * c for r, c in niveis)
    cfg = {"max_side": args.max_side, "step": args.step,
           "scales": [int(s) for s in args.scales.split(",")],
           "max_desc": args.max_desc, "rootsift": bool(args.rootsift),
           "piramide": niveis, "augment_xy": bool(args.augment_xy),
           "intra_norm": bool(args.intra_norm)}
    d_gmm = args.pca_dim + (2 if args.augment_xy else 0)

    from registry import LABELS, load_blocks
    log(f"coorte de producao de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset)
    stems = [str(s) for s in cohort.stems]
    if args.exclude_stems:
        fora = set(args.exclude_stems.split(","))
        stems = [s for s in stems if s not in fora]
    if args.limit:
        stems = stems[:args.limit]
    label_path, _, _ = LABELS[args.dataset]
    meta = pd.read_csv(ROOT / label_path, dtype={"stem": str})
    smap = build_stem_map(args.image_dir, meta)
    falta = [s for s in stems if s not in smap]
    if falta:
        raise SystemExit(f"{len(falta)} stems sem imagem (ex.: {falta[:3]}); "
                         f"extrair com buraco daria bloco desalinhado (§5.18b)")
    log(f"  {len(stems):,} imagens")

    # ---- fase 1: PCA + GMM numa amostra (sem rotulo) ----
    base = out / "codebook.npz"
    if base.exists():
        z = np.load(base)
        pca = {"media": z["media"], "comp": z["comp"]}
        gmm = {"w": z["w"], "mu": z["mu"], "var": z["var"]}
        log(f"  codebook em cache: PCA {z['comp'].shape} GMM K={len(z['w'])}")
    else:
        pool_stems = stems
        if args.fit_split != "any" and "split" in meta.columns:
            no_split = set(meta.loc[meta["split"] == args.fit_split, "stem"].astype(str))
            so_treino = [s for s in stems if s in no_split]
            if so_treino:
                pool_stems = so_treino
                log(f"  ajustando PCA/GMM so no split '{args.fit_split}': "
                    f"{len(pool_stems):,} candidatas")
        rng = np.random.default_rng(42)
        sel = [pool_stems[i] for i in rng.choice(
            len(pool_stems), min(args.fit_images, len(pool_stems)), replace=False)]
        log(f"  amostrando descritores de {len(sel):,} imagens "
            f"({args.fit_desc_per_image}/imagem)")
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(cfg, {"media": np.zeros(128), "comp": np.eye(128)},
                               {"w": np.ones(1), "mu": np.zeros((1, 128)),
                                "var": np.ones((1, 128))})) as pool:
            partes = pool.map(_amostra, [(s, str(smap[s]), args.fit_desc_per_image)
                                         for s in sel], chunksize=4)
        AMO = np.concatenate([p for p in partes if len(p)]).astype(np.float64)
        D, POS = AMO[:, :128], AMO[:, 128:]      # o `_amostra` traz a posicao
        log(f"  {len(D):,} descritores em {time.time() - t0:.0f}s")

        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        # O limite de 1 thread existe para os WORKERS (§1: evitar
        # sobre-inscricao com 96 processos). O ajuste da PCA e da GMM roda no
        # processo PRINCIPAL e sozinho, entao aqui o limite so faria o K=256 em
        # ~1M descritores levar horas a toa.
        try:
            from threadpoolctl import threadpool_limits
            ctx = threadpool_limits(limits=args.fit_threads)
            log(f"  ajuste com ate {args.fit_threads} threads de BLAS")
        except Exception:
            ctx = None
        p = PCA(n_components=args.pca_dim, random_state=42).fit(D)
        Dp = (D - p.mean_) @ p.components_.T
        log(f"  PCA 128 -> {args.pca_dim} (variancia retida "
            f"{p.explained_variance_ratio_.sum() * 100:.1f}%)")
        if args.augment_xy:
            # a GMM TEM de ser ajustada no mesmo espaco em que vai codificar
            Dp = aumenta_xy(Dp, POS)
            log(f"  posicao anexada: GMM ajustada em {Dp.shape[1]} dims")
        g = GaussianMixture(n_components=args.gmm_k, covariance_type="diag",
                            max_iter=100, random_state=42, reg_covar=1e-6,
                            verbose=0).fit(Dp)
        log(f"  GMM K={args.gmm_k} diagonal ajustada "
            f"(convergiu={g.converged_}, {g.n_iter_} iteracoes)")
        if ctx is not None:
            ctx.unregister()
        pca = {"media": p.mean_, "comp": p.components_}
        gmm = {"w": g.weights_, "mu": g.means_, "var": g.covariances_}
        with open(base, "wb") as h:
            np.savez(h, media=pca["media"], comp=pca["comp"],
                     w=gmm["w"], mu=gmm["mu"], var=gmm["var"])

    dim = 2 * args.gmm_k * d_gmm * n_celulas
    log(f"  dimensao do FV: 2 x {args.gmm_k} x {d_gmm} x "
        f"{n_celulas} celulas ({args.piramide}) = {dim:,}")
    if gmm["mu"].shape[1] != d_gmm:
        raise SystemExit(f"codebook em cache tem GMM de {gmm['mu'].shape[1]} dims "
                         f"e a configuracao pede {d_gmm} -- apagar o codebook.npz "
                         f"ou corrigir as flags (um mudo daria acuracia ruim sem erro)")

    # ---- fase 2: codificar ----
    shards = [(i, stems[i:i + SHARD]) for i in range(0, len(stems), SHARD)]
    log(f"{len(shards)} shards de ate {SHARD}; {args.workers} workers")
    t0 = time.time()
    for ordem, (off, chunk) in enumerate(shards, 1):
        sp = shard_dir / f"shard_{off:07d}.npz"
        if sp.exists():
            continue
        vals = np.empty((len(chunk), dim), dtype=np.float32)
        falhas: list[tuple[str, str]] = []
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(cfg, pca, gmm)) as pool:
            for pos, (stem, v, erro) in enumerate(
                    pool.imap(_uma, [(s, str(smap[s])) for s in chunk], chunksize=2)):
                if v is None:
                    falhas.append((stem, erro))
                else:
                    vals[pos] = v
        if falhas:
            raise SystemExit(f"{len(falhas)} falhas no shard {off} "
                             f"(ex.: {falhas[:2]}); consertar e repetir")
        tmp = sp.with_suffix(".npz.part")
        with open(tmp, "wb") as h:
            np.savez(h, values=vals, stems=np.asarray(chunk))
        tmp.rename(sp)
        feito = off + len(chunk)
        taxa = feito / max(1e-9, time.time() - t0)
        log(f"  shard {ordem}/{len(shards)} ok  ({taxa:.1f} img/s, "
            f"restam ~{(len(stems) - feito) / max(1e-9, taxa) / 60:.0f} min)")

    log("montando os arrays finais")
    M = np.lib.format.open_memmap(out / "features.npy", mode="w+",
                                  dtype=np.float32, shape=(len(stems), dim))
    cursor, ordem_ok = 0, []
    for off, _ in shards:
        z = np.load(shard_dir / f"shard_{off:07d}.npz", allow_pickle=False)
        n = len(z["stems"])
        M[cursor:cursor + n] = z["values"]
        ordem_ok += [str(s) for s in z["stems"]]
        cursor += n
    M.flush()
    if ordem_ok != stems:
        raise SystemExit("ordem dos shards divergiu da coorte -- abortando")
    with open(out / "stems.npy", "wb") as h:
        np.save(h, np.asarray(stems))
    (out / "meta.json").write_text(json.dumps({
        "protocol": "Sanchez et al., IJCV 2013 (improved Fisher vector)",
        "descriptor": "dense SIFT (OpenCV), RootSIFT" if cfg["rootsift"]
                      else "dense SIFT (OpenCV)",
        "max_side": args.max_side, "crop": "nenhum",
        "step": args.step, "scales": cfg["scales"],
        "max_desc_per_image": args.max_desc,
        "pca_dim": args.pca_dim, "gmm_k": args.gmm_k,
        "gmm_covariance": "diag", "fv_dim": dim,
        "spatial_pyramid": args.piramide, "pyramid_cells": n_celulas,
        "augment_xy": bool(args.augment_xy), "gmm_dim": d_gmm,
        "intra_normalisation": bool(args.intra_norm),
        "normalisation": (("intra-L2 por (u_k,v_k) + " if args.intra_norm else "")
                          + ("power(alpha=0.5) + L2 por celula, L2 global"
                             if n_celulas > 1 else "power(alpha=0.5) + L2")),
        "fit_split": args.fit_split, "fit_images": args.fit_images,
        "fit_desc_per_image": args.fit_desc_per_image,
    }, indent=2))
    log(f"gravado {out / 'features.npy'}  ({len(stems):,}, {dim:,})")
    log(f"custo total: {(time.time() - t0) / 60:.1f} min de relogio")


if __name__ == "__main__":
    main()
