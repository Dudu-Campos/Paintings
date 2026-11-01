#!/usr/bin/env python3
"""Fisher Vector com SIFT denso na GPU -- a reescrita que torna a densidade
PUBLICADA alcancavel.

POR QUE ESTE SCRIPT EXISTE. O `extract_fisher_vector_features.py` chama
`cv2.SIFT.compute(img, lista_de_keypoints)`, que trata cada ponto como esparso:
extrai patch de gradiente, resolve orientacao e faz interpolacao trilinear, UMA
VEZ POR KEYPOINT. O artigo do Food-101 diz "as implemented in VlFeat", e o
`vl_dsift` explora a grade REGULAR -- gradiente uma vez, 8 canais de orientacao,
convolucao separavel. Medido em 11/out, 5 imagens do food101 a 512 px:

    cv2.SIFT.compute  step 8    7.134 ms/img    12.215 desc    584,03 us/desc
    kornia GPU        stride 8      8,1 ms/img   13.926 desc     0,581 us/desc
    kornia GPU        stride 3      7,1 ms/img   99.317 desc     0,072 us/desc

**880x no mesmo passo, e com stride 3 o custo por descritor cai 8.100x.** O
sintoma estava no proprio artigo e eu li errado: eles reportam "70% do tempo e
gasto no encoding" e nos mediamos 99% no SIFT. O perfil invertido era a
denuncia.

CONSEQUENCIA DE METODO, e ela e desconfortavel: este script NAO entra em
comparacao de relogio contra o `hand-8`, que e CPU puro e nao otimizado. Seria
o erro espelhado. E o achado joga suspeita sobre os nossos proprios numeros de
custo -- o §6 ja listava SNIC, gSLICr e radix sort no Felzenszwalb (54% do
custo, §9.7) como alavancas conhecidas e NUNCA medidas. O que sobrevive do §5.4b
e o argumento ESTRUTURAL (extracao O(n) paga uma vez e reusada), que nao depende
de implementacao.

O kornia NAO e bit-identico ao VLFeat nem ao OpenCV -- e outra implementacao do
mesmo algoritmo. Os caches do script antigo continuam validos como o que sao, e
este grava em diretorio proprio.

Uso (configuracao publicada, Chatfield et al. BMVC 2014 §3.1):
    python src/extractors/extract_fisher_vector_gpu.py --dataset food101 \
        --image-dir data/food101/food-101/images --upscale 2 --stride 3 \
        --bin-sizes 4,6,8,11,16,23,32 --pca-dim 80 --gmm-k 256 --augment-xy
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from kornia.feature import DenseSIFTDescriptor
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_wndcharm_tiled_features import build_stem_map, log  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHARD = 2000


class Imagens(Dataset):
    """Decodifica e redimensiona na CPU (workers), para a GPU nao esperar I/O."""

    def __init__(self, caminhos: list[str], max_side: int, upscale: int) -> None:
        self.caminhos, self.max_side, self.upscale = caminhos, max_side, upscale

    def __len__(self) -> int:
        return len(self.caminhos)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.caminhos[i]) as im:
            im = im.convert("L")
            w, h = im.size
            e = self.max_side / max(w, h)
            if e < 1.0:
                im = im.resize((max(1, round(w * e)), max(1, round(h * e))),
                               Image.BILINEAR)
            if self.upscale != 1:
                im = im.resize((im.size[0] * self.upscale, im.size[1] * self.upscale),
                               Image.BILINEAR)
            return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())


def dsift(t: torch.Tensor, descs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(T,128) descritores e (T,2) posicoes relativas (y/H, x/W).

    As posicoes saem do PROPRIO grid de saida por `linspace` em [0,1]: o kornia
    devolve um mapa denso, entao a coordenada de cada celula e a posicao dela no
    mapa. Isto e aproximacao de meio-pixel contra o centro do campo receptivo, e
    e irrelevante -- o aumento (x,y) normaliza por W,H de qualquer forma.
    """
    x = t.to(DEV, non_blocking=True).float().div_(255.0)[None, None]
    saida, pos = [], []
    for d in descs.values():
        o = d(x)[0]                                  # (128, H', W')
        _, hh, ww = o.shape
        saida.append(o.reshape(128, -1).T)
        yy = torch.linspace(0.0, 1.0, hh, device=DEV)
        xx = torch.linspace(0.0, 1.0, ww, device=DEV)
        g = torch.stack(torch.meshgrid(yy, xx, indexing="ij"), dim=-1)
        pos.append(g.reshape(-1, 2))
    return torch.cat(saida), torch.cat(pos)


def fisher_gpu(X: torch.Tensor, w: torch.Tensor, mu: torch.Tensor,
               var: torch.Tensor, intra: bool) -> torch.Tensor:
    """O mesmo FV improved do script de CPU, em float32 na GPU.

    Os posteriores saem por matmul em LOTES de descritores: com T=611.000 e
    K=256 a matriz (T,K) sozinha e 626 MB, e materializar (T,K,D) seria
    impensavel. Os momentos S0/S1/S2 acumulam entre lotes, que e exato -- sao
    somas.
    """
    K, D = mu.shape
    inv = 1.0 / var
    const = -0.5 * (torch.log(var).sum(1) + (mu * mu * inv).sum(1))
    base = (torch.log(w) + const)[None, :]
    mui = (mu * inv).T
    S0 = torch.zeros(K, device=DEV)
    S1 = torch.zeros(K, D, device=DEV)
    S2 = torch.zeros(K, D, device=DEV)
    T = X.shape[0]
    for i in range(0, T, 65536):
        Xi = X[i:i + 65536]
        X2 = Xi * Xi
        lp = base - 0.5 * (X2 @ inv.T) + (Xi @ mui)
        lp -= lp.max(1, keepdim=True).values
        g = torch.exp(lp)
        g /= g.sum(1, keepdim=True)
        S0 += g.sum(0)
        S1 += g.T @ Xi
        S2 += g.T @ X2
    sig = torch.sqrt(var)
    g_mu = (S1 - mu * S0[:, None]) / (sig * T * torch.sqrt(w)[:, None])
    g_var = (S2 - 2 * mu * S1 + (mu * mu - var) * S0[:, None]) \
        / (var * T * torch.sqrt(2.0 * w)[:, None])
    if intra:
        nk = torch.sqrt((g_mu * g_mu).sum(1) + (g_var * g_var).sum(1)).clamp_min(1e-12)
        g_mu, g_var = g_mu / nk[:, None], g_var / nk[:, None]
    v = torch.cat([g_mu.reshape(-1), g_var.reshape(-1)])
    v = torch.sign(v) * torch.sqrt(torch.abs(v))
    return v / v.norm().clamp_min(1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--upscale", type=int, default=1,
                    help="o §3.1 do Chatfield comeca por 'upscaling the image by "
                         "a factor of 2'")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--bin-sizes", default="4,6,8,11,16,23,32",
                    help="7 escalas em progressao ~sqrt(2), como no artigo. O "
                         "suporte do descritor e 4x o bin size.")
    ap.add_argument("--pca-dim", type=int, default=80)
    ap.add_argument("--gmm-k", type=int, default=256)
    ap.add_argument("--augment-xy", action="store_true", default=False)
    ap.add_argument("--intra-norm", action="store_true", default=False)
    ap.add_argument("--max-desc", type=int, default=0,
                    help="0 = sem cap. Com a GPU o encoding deixa de ser caro, "
                         "entao subamostrar virou escolha e nao necessidade.")
    ap.add_argument("--fit-images", type=int, default=1500)
    ap.add_argument("--fit-desc-per-image", type=int, default=600)
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--fit-stems", type=Path, default=None,
                    help="restringe o pool do codebook a uma lista de stems (um "
                         "por linha). E o controle INDUTIVO: nos corpora de CV o "
                         "metadado nao tem coluna `split`, entao o default ajusta "
                         "PCA/GMM numa amostra da COORTE INTEIRA -- transdutivo "
                         "(sem rotulo, mas o teste esta no pool). Com esta flag o "
                         "pool e o treino de um fold, e a diferenca entre os dois "
                         "mede o tamanho da transducao em vez de declara-la.")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = args.out_dir or ROOT / "outputs" / f"{args.dataset}_fisher_vector_gpu"
    (out / "shards").mkdir(parents=True, exist_ok=True)
    bins = [int(b) for b in args.bin_sizes.split(",")]
    d_gmm = args.pca_dim + (2 if args.augment_xy else 0)
    dim = 2 * args.gmm_k * d_gmm
    descs = {b: DenseSIFTDescriptor(spatial_bin_size=b, stride=args.stride,
                                    rootsift=True).to(DEV).eval() for b in bins}

    from registry import LABELS, load_blocks
    log(f"device {DEV} | coorte de {args.dataset} pelo registry")
    cohort = load_blocks(args.dataset)
    stems = [str(s) for s in cohort.stems]
    meta = pd.read_csv(ROOT / LABELS[args.dataset][0], dtype={"stem": str})
    smap = build_stem_map(args.image_dir, meta)
    falta = [s for s in stems if s not in smap]
    if falta:
        raise SystemExit(f"{len(falta)} stems sem imagem (ex.: {falta[:3]})")
    log(f"  {len(stems):,} imagens | bins {bins} stride {args.stride} "
        f"upscale {args.upscale}x | FV {dim:,}d")

    def carrega(sel: list[str], bs: int = 1) -> DataLoader:
        return DataLoader(Imagens([str(smap[s]) for s in sel], args.max_side,
                                  args.upscale),
                          batch_size=bs, num_workers=args.workers,
                          collate_fn=lambda b: b, pin_memory=False)

    # ---------------- fase 1: PCA + GMM numa amostra, sem rotulo ----------------
    cb = out / "codebook.npz"
    if cb.exists():
        z = np.load(cb)
        media, comp = z["media"], z["comp"]
        w_, mu_, var_ = z["w"], z["mu"], z["var"]
        log(f"  codebook em cache: PCA {comp.shape} GMM K={len(w_)} D={mu_.shape[1]}")
    else:
        pool = stems
        if args.fit_stems is not None:
            querido = {l.strip() for l in args.fit_stems.read_text().splitlines() if l.strip()}
            fora = querido - set(stems)
            if fora:
                raise SystemExit(f"--fit-stems: {len(fora)} stems fora da coorte "
                                 f"(ex.: {sorted(fora)[:3]})")
            pool = [x for x in stems if x in querido]
            log(f"  pool do codebook RESTRITO a {len(pool):,} stems (controle indutivo)")
        elif args.fit_split != "any" and "split" in meta.columns:
            no = set(meta.loc[meta["split"] == args.fit_split, "stem"].astype(str))
            so = [s for s in stems if s in no]
            if so:
                pool = so
                log(f"  ajustando PCA/GMM so no split '{args.fit_split}': {len(pool):,}")
        rng = np.random.default_rng(42)
        sel = [pool[i] for i in rng.choice(len(pool),
                                           min(args.fit_images, len(pool)), replace=False)]
        t0 = time.time()
        amo, amp = [], []
        with torch.no_grad():
            for lote in carrega(sel):
                d, p = dsift(lote[0], descs)
                k = min(args.fit_desc_per_image, len(d))
                idx = torch.randperm(len(d), device=DEV)[:k]
                amo.append(d[idx].cpu().numpy())
                amp.append(p[idx].cpu().numpy())
        D = np.concatenate(amo).astype(np.float64)
        POS = np.concatenate(amp).astype(np.float64)
        log(f"  {len(D):,} descritores em {time.time() - t0:.0f}s")
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        p = PCA(n_components=args.pca_dim, random_state=42).fit(D)
        Dp = (D - p.mean_) @ p.components_.T
        log(f"  PCA 128 -> {args.pca_dim} (variancia retida "
            f"{p.explained_variance_ratio_.sum() * 100:.1f}%)")
        if args.augment_xy:
            Dp = np.hstack([Dp, POS[:, 1:2] - 0.5, POS[:, 0:1] - 0.5])
            log(f"  posicao anexada: GMM ajustada em {Dp.shape[1]} dims")
        g = GaussianMixture(args.gmm_k, covariance_type="diag", max_iter=100,
                            random_state=42, reg_covar=1e-6).fit(Dp)
        log(f"  GMM K={args.gmm_k} (convergiu={g.converged_}, {g.n_iter_} it)")
        media, comp = p.mean_, p.components_
        w_, mu_, var_ = g.weights_, g.means_, g.covariances_
        with open(cb, "wb") as h:
            np.savez(h, media=media, comp=comp, w=w_, mu=mu_, var=var_)

    if mu_.shape[1] != d_gmm:
        raise SystemExit(f"codebook tem GMM de {mu_.shape[1]} dims e a config pede "
                         f"{d_gmm} -- apagar o codebook.npz ou corrigir as flags")
    tm = torch.tensor(media, dtype=torch.float32, device=DEV)
    tc = torch.tensor(comp, dtype=torch.float32, device=DEV).T
    tw = torch.tensor(w_, dtype=torch.float32, device=DEV)
    tmu = torch.tensor(mu_, dtype=torch.float32, device=DEV)
    tvar = torch.tensor(var_, dtype=torch.float32, device=DEV)

    # ---------------- fase 2: codificar ----------------
    shards = [(i, stems[i:i + SHARD]) for i in range(0, len(stems), SHARD)]
    log(f"{len(shards)} shards de ate {SHARD}")
    t0, feitos, ts = time.time(), 0, []
    for ordem, (off, chunk) in enumerate(shards, 1):
        sp = out / "shards" / f"shard_{off:07d}.npz"
        if sp.exists():
            feitos += len(chunk)
            continue
        vals = np.empty((len(chunk), dim), dtype=np.float32)
        pos_i = 0
        with torch.no_grad():
            for lote in carrega(chunk):
                d, p = dsift(lote[0], descs)
                if args.max_desc and len(d) > args.max_desc:
                    idx = torch.randperm(len(d), device=DEV)[:args.max_desc]
                    d, p = d[idx], p[idx]
                ts.append(len(d))
                X = (d - tm) @ tc
                if args.augment_xy:
                    X = torch.cat([X, p[:, 1:2] - 0.5, p[:, 0:1] - 0.5], dim=1)
                vals[pos_i] = fisher_gpu(X, tw, tmu, tvar,
                                         args.intra_norm).cpu().numpy()
                pos_i += 1
        if pos_i != len(chunk):
            raise SystemExit(f"shard {off}: {pos_i} de {len(chunk)} -- abortando")
        tmp = sp.with_suffix(".npz.part")
        with open(tmp, "wb") as h:
            np.savez(h, values=vals, stems=np.asarray(chunk))
        tmp.rename(sp)
        feitos += len(chunk)
        taxa = feitos / max(1e-9, time.time() - t0)
        log(f"  shard {ordem}/{len(shards)} ok  ({taxa:.1f} img/s, "
            f"restam ~{(len(stems) - feitos) / max(1e-9, taxa) / 60:.0f} min, "
            f"T mediano {int(np.median(ts)) if ts else 0:,})")

    log("montando os arrays finais")
    M = np.lib.format.open_memmap(out / "features.npy", mode="w+",
                                  dtype=np.float32, shape=(len(stems), dim))
    cur, ordem_ok = 0, []
    for off, _ in shards:
        z = np.load(out / "shards" / f"shard_{off:07d}.npz", allow_pickle=False)
        n = len(z["stems"])
        M[cur:cur + n] = z["values"]
        ordem_ok += [str(s) for s in z["stems"]]
        cur += n
    M.flush()
    if ordem_ok != stems:
        raise SystemExit("ordem dos shards divergiu da coorte -- abortando")
    with open(out / "stems.npy", "wb") as h:
        np.save(h, np.asarray(stems))
    (out / "meta.json").write_text(json.dumps({
        "protocol": "Chatfield et al. BMVC 2014 §3.1 (IFV), dense SIFT na GPU",
        "descriptor": "kornia DenseSIFTDescriptor (vl_dsift-like), RootSIFT",
        "implementacao": "kornia -- NAO bit-identico a VLFeat nem a OpenCV",
        "max_side": args.max_side, "upscale": args.upscale,
        "stride": args.stride, "bin_sizes": bins,
        "max_desc_per_image": args.max_desc or "sem cap",
        "T_mediano": int(np.median(ts)) if ts else None,
        "pca_dim": args.pca_dim, "gmm_k": args.gmm_k, "gmm_dim": d_gmm,
        "augment_xy": bool(args.augment_xy),
        "intra_normalisation": bool(args.intra_norm),
        "fv_dim": dim, "fit_split": args.fit_split,
        "nota_de_custo": "GPU -- NAO comparar relogio com o hand-8, que e CPU",
    }, indent=2))
    log(f"gravado {out / 'features.npy'}  ({len(stems):,}, {dim:,})")
    log(f"custo total: {(time.time() - t0) / 60:.1f} min de relogio")


if __name__ == "__main__":
    main()
