"""Posteriors de VALIDACAO do backbone, por inferencia nos checkpoints salvos.

O §4.1f registra que a rota limpa de selecao do peso -- escolher no `val`
oficial -- estava pendente. O lado do backbone custa **inferencia, nao treino**:
ha 190 checkpoints em `outputs/final_finetune_grid/`. Este script carrega cada
um e roda o `val` oficial.

**SO DOIS CORPORA SAO ELEGIVEIS**, e a razao e de contaminacao, nao de custo:
qualquer fatia recortada do treino JA PASSOU PELO GRADIENTE do backbone, e o
§4.1c mede o tamanho disso (96,65% nas linhas de treino contra 67,32% no teste).
Os unicos conjuntos fora do gradiente sao o `val` OFICIAL e o teste, e `val`
oficial so existe no `minc2500` (2.875) e no `places_lt` (7.300) -- no
`food101` o split publicado e train/test, e nos tres de arte o "val" e fatia do
proprio treino por fold.

**VALIDACAO OBRIGATORIA antes de confiar na saida** (§5.6: um braco de controle
que nao reproduz o valor conhecido nao e controle): com `--check-test` o script
roda a MESMA pipeline no conjunto de TESTE e compara com o
`fold_NN_posteriors.npz` ja gravado pelo trainer. Se as probabilidades nao
baterem, a transformacao de avaliacao ou o carregamento do checkpoint estao
errados -- e o §8 mede que a transformacao sozinha vale 4,91pp.

O checkpoint e um ENVELOPE, nao um `state_dict` (§8): carregar o arquivo direto
da 152 chaves faltando e um modelo ALEATORIO, sem erro.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.train_resnet_wikiart import (build_model, build_transforms,  # noqa: E402
                                                evaluate, load_records, make_loader)

# corpus -> (metadado, diretorio de imagens). So os dois com `val` oficial.
ELEGIVEIS = {
    "minc2500": ("data/minc2500_meta.csv", "/tmp/prism_data/minc/minc-2500/images"),
    "places_lt": ("data/places_lt_meta.csv", "/tmp/prism_data/places_lt/images"),
}
ARQ = {"resnet50": "resnet50", "vit_b_16": "vit_b_16"}


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=list(ELEGIVEIS))
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit_b_16"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-test", action="store_true",
                    help="roda o TESTE e confere contra o posterior ja gravado")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = build_transforms(224, "strong")     # a receita da grade inteira (§9)

    for corpus in args.corpora:
        meta, image_dir = ELEGIVEIS[corpus]
        # O `load_records` aborta se QUALQUER linha do metadado nao resolver para
        # um arquivo, e no places_lt 528 linhas de TREINO nao resolvem: sao as 5
        # classes permutadas do §5.18d (`d_desert_road` e companhia), onde o
        # metadado atribui a uma classe arquivos que pertencem a outra. Este
        # script so precisa de `val` e `test`, e os dois estao LIMPOS -- conferido
        # classe a classe contra `Places_LT_val.txt`, 0 de 365 divergem. Entao
        # filtra-se o metadado para os splits usados, em vez de tolerar linha
        # nao resolvida (o que esconderia um erro de verdade noutro corpus).
        frame = pd.read_csv(ROOT / meta, dtype={"stem": str})
        usados = frame[frame["split"].isin(("val", "test"))]
        if len(usados) < len(frame):
            meta_path = Path(tempfile.gettempdir()) / f"{corpus}_meta_val_test.csv"
            usados.to_csv(meta_path, index=False)
            log(f"{corpus}: metadado filtrado para val+test "
                f"({len(usados):,} de {len(frame):,} linhas; "
                f"{len(frame) - len(usados):,} de treino ficam de fora)")
        else:
            meta_path = ROOT / meta
        records, classes = load_records(meta_path, Path(image_dir), "style", 5)
        # O `record` e {path, stem, label} -- o split NAO vem nele. O trainer
        # monta `split_by_stem` do metadado e mapeia por stem (linha 680); fazer
        # igual, em vez de supor, e o que o §5.6 exige de um braco de controle.
        frame = pd.read_csv(ROOT / meta, dtype={"stem": str})
        split_by_stem = dict(zip(frame["stem"].astype(str), frame["split"].astype(str)))
        splits = np.asarray([split_by_stem[r["stem"]] for r in records], dtype=str)
        train = [r for r, s in zip(records, splits) if s == "train"]
        val = [r for r, s in zip(records, splits) if s == "val"]
        test = [r for r, s in zip(records, splits) if s == "test"]
        # §8: todo script de split oficial IMPRIME o tamanho das particoes, e a
        # conferencia contra o numero publicado entra na rotina antes de reportar.
        log(f"{corpus}: split oficial ({len(train):,} train / {len(val):,} val / "
            f"{len(test):,} test) | {len(classes)} classes")
        if not val:
            raise SystemExit(f"{corpus}: sem linhas de `val` -- corpus nao elegivel")

        for model_name in args.models:
            for seed in args.seeds:
                base = (ROOT / "outputs/final_finetune_grid" / corpus / model_name /
                        f"seed_{seed}")
                ckpts = sorted(base.glob("fold_*/fold_*_best.pt"))
                if not ckpts:
                    log(f"  {model_name}/seed {seed}: sem checkpoint, pulando")
                    continue
                ckpt = ckpts[0]
                out = ckpt.parent / "val_posteriors.npz"
                blob = torch.load(ckpt, map_location="cpu", weights_only=False)
                model = build_model("imagenet", len(classes), architecture=ARQ[model_name])
                model.load_state_dict(blob["model_state_dict"], strict=True)
                model.to(device)

                if args.check_test:
                    loader = make_loader(test, eval_tf, batch_size=args.batch_size,
                                         workers=args.workers, shuffle=False,
                                         seed=seed, device=device)
                    with torch.no_grad():
                        _, stems_t, _, _, prob_t = evaluate(model, loader, device)
                    ref = np.load(ckpt.parent / f"{ckpt.stem.replace('_best','')}_posteriors.npz",
                                  allow_pickle=True)
                    order = pd.Index(list(ref["stems"].astype(str))).get_indexer(
                        np.asarray(stems_t, dtype=str))
                    gap = float(np.abs(prob_t - ref["probabilities"][order]).max())
                    log(f"  {model_name}/seed {seed}: CONFERENCIA no teste, "
                        f"maior diferenca de probabilidade = {gap:.2e}")
                    if gap > 1e-4:
                        raise SystemExit(
                            "a pipeline reconstruida NAO reproduz o posterior gravado -- "
                            "transformacao de avaliacao ou checkpoint errados (§8)")

                loader = make_loader(val, eval_tf, batch_size=args.batch_size,
                                     workers=args.workers, shuffle=False,
                                     seed=seed, device=device)
                with torch.no_grad():
                    metrics, stems_v, y_true_v, _, prob_v = evaluate(model, loader, device)
                with out.open("wb") as handle:
                    np.savez_compressed(handle, stems=np.asarray(stems_v, dtype=str),
                                        probabilities=prob_v, y_true=y_true_v)
                log(f"  {model_name}/seed {seed}: val BAcc "
                    f"{100*metrics['balanced_accuracy']:.2f} -> {out.name}")
    log("FIM")


if __name__ == "__main__":
    main()
