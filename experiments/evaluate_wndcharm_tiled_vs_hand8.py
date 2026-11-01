"""WND-CHARM no protocolo TILED/CANONICO (Shamir et al., ACM TAP 7(2), 2010)
contra o hand-8. Consumidor de `extract_wndcharm_tiled_features.py`, cuja
extracao (99.816 x 16 x 2895) terminou em 29/set -- este script e a peca que
faltava (nenhum avaliador do tiled existia ainda).

O extrator cobre padronizacao 600x600 e os 16 tiles de 150x150 do artigo; a
selecao de Fisher fica AQUI, DENTRO de cada fold (selecionar na coorte inteira
seria vazamento -- o mecanismo do §6 que derrubou o SNF, e o proprio docstring
do extrator ja avisa disso).

**Classificador: o MESMO MLP de producao do resto do projeto, NAO o WND5 do
artigo original.** O WND5 (Orlov et al. 2008) e um kNN ponderado por escore de
Fisher com uma formula de agregacao por classe que nao esta determinada com
certeza suficiente pra reivindicar reproducao -- e o §7.11b deste projeto ja
registra o custo de chamar de reproducao algo que nao era (o HOG do Food-101).
Em vez de arriscar um numero plausivel-mas-errado, usa-se `evaluate_mlp` de
producao (mesmo import de `evaluate_wndcharm_vs_hand8.py`), que mantem hand-8 e
WND-CHARM no MESMO classificador -- comparacao apples-to-apples, e e o padrao
de todo o resto do artigo (nenhuma view externa e avaliada por classificador
proprio).

O que o protocolo do artigo contribui aqui, fielmente:
  1. padronizacao 600x600 e tiles de 150x150 (ja feito na extracao);
  2. reducao de dimensao por escore de Fisher (ANOVA F via `sklearn.f_classif`,
     que E o "Fisher score" da literatura de selecao de feature), rejeitando os
     85% mais fracos, computado SO no treino de cada fold;
  3. "averaging the 16 similarity vectors of the tiles" e adaptado para espaco
     de FEATURE (nao ha vetor de similaridade sem o WND5): media dos 16 vetores
     de feature por imagem, que e pooling padrao de descritor local -- operacao
     bem definida, ao contrario de reimplementar o classificador do artigo.

**UPSCALE, sensibilidade.** 35,3% da coorte tem lado menor < 600px (medido na
coorte INTEIRA aqui, nao amostra como no docstring do extrator) e portanto foi
upscaled pra caber no protocolo. Reporta-se BAcc restrita ao subconjunto NATIVO
do teste como sensibilidade -- MESMO modelo treinado (o split de treino nao
muda), so o teste e filtrado, entao o retreino e so custo de wall-clock, nao
reintroduz nenhuma escolha nova.

Uso:
    python experiments/evaluate_wndcharm_tiled_vs_hand8.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from corpora import HAND8, clean                     # noqa: E402
from protocol import SEED, evaluate_mlp               # noqa: E402
from registry import LABELS, load_blocks                      # noqa: E402

FISHER_KEEP_FRAC = 0.15  # "the weakest 85% ... are rejected" (Shamir et al. 2010)

# Corpora com particao PUBLICADA. Rodar 5-fold CV neles e o bug que o §8
# registra no `evaluate_preprocess_effect_oilpainting` -- a constante OFFICIAL
# existia e nunca era lida. No Places-LT a CV derruba o desbalanceamento de 996x
# para 41x e apaga o regime que motiva o corpus (§7.5); no NCT-CRC apaga a
# separacao por paciente do teste externo, que vale 11pp (§7.14).
OFFICIAL = {"food101", "nct_crc", "minc2500", "places_lt"}
TEST_COHORT = {"nct_crc": "nct_crc_val"}   # teste e coorte SEPARADA, nao fatia

# Referencia do `hand-8` sob BranchMLP estreito, do §5.18e (mesma cabeca, mesmo
# protocolo que este script usa). A margarita reproduz ~0,6pp abaixo da absinto
# (§9.7: o mesmo hand-8 da 38,53/38,71/38,74 entre execucoes), entao a banda e
# larga de proposito -- ela existe para pegar caminho ERRADO, nao para validar a
# terceira casa. Uma guarda que dispara em todo corpus ninguem le: a versao
# anterior tinha a banda do multitask fixa e gritava nos outros seis.
BANDA_HAND8 = {
    "multitask":   (37.0, 39.5),   # 38,74 no §5.18e
    "oilpainting": (42.0, 44.5),   # 43,12
    "rijksdraw":   (54.0, 56.5),   # 55,47
    "food101":     (35.5, 38.5),   # 37,07  (split oficial)
    "minc2500":    (54.0, 57.0),   # 55,57  (split oficial 1)
    "nct_crc":     (85.0, 89.0),   # 86,97  (teste externo CRC-VAL-HE-7K)
    "places_lt":   (10.5, 14.5),   # 12,00  (split oficial; la sao 362 classes
}                                  #         contra as 365 daqui, §5.18e)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def algorithm_of(raw: str) -> str:
    """`Haralick Textures (Fourier (Wavelet ())) [7]` -> `Haralick Textures`."""
    match = re.match(r"^(.*?)\s*\(", raw)
    return match.group(1) if match else raw


def fisher_select(train_x: np.ndarray, train_y: np.ndarray, keep_frac: float) -> np.ndarray:
    """Escore de Fisher (ANOVA F, `sklearn.f_classif`) computado SO no treino.

    Devolve os indices das `keep_frac` colunas de maior F (ordenados), i.e.
    rejeita as `1 - keep_frac` mais fracas -- e a definicao do artigo.
    """
    f_values, _ = f_classif(train_x, train_y)
    # coluna quase-constante DENTRO do treino (mas viva na coorte inteira) da
    # F indefinido (0/0) -- sklearn ja devolve NaN; tratar como a mais fraca.
    f_values = np.nan_to_num(f_values, nan=-np.inf)
    n_keep = max(1, int(np.ceil(keep_frac * len(f_values))))
    order = np.argsort(f_values)[::-1]
    return np.sort(order[:n_keep])


def run_condition(name: str, blocks: list[np.ndarray], labels: np.ndarray,
                   splits: list, rows: list[dict], output: Path,
                   n_folds: int) -> None:
    for fold, (train, test) in enumerate(splits[:n_folds], start=1):
        started = time.perf_counter()
        metrics = evaluate_mlp(blocks, labels, train, test, fold, epochs=150, patience=15)
        rows.append({
            "condition": name, "fold": fold,
            "n_blocks": len(blocks),
            "dimensions": sum(b.shape[1] for b in blocks),
            "seconds": time.perf_counter() - started,
            "bacc": 100 * metrics["bacc"],
            "accuracy": 100 * metrics["accuracy"],
            "macro_f1": 100 * metrics["macro_f1"],
        })
        log(f"  {name:<42} f{fold}  {rows[-1]['n_blocks']:>2}b "
            f"{rows[-1]['dimensions']:>5}d  BAcc {rows[-1]['bacc']:5.2f}  "
            f"({rows[-1]['seconds']:.0f}s)")
        pd.DataFrame(rows).to_csv(output, index=False, float_format="%.4f")


def load_side(dataset: str, tiled_dir: Path) -> dict:
    """Coorte do registry + tiled poolado, ALINHADOS POR STEM.

    Separado em funcao porque o `nct_crc` precisa de DOIS lados: a coorte do
    registry e so o treino (100.000) e o teste e a coorte externa `nct_crc_val`
    (7.180, pacientes diferentes, §7.14).
    """
    log(f"coorte de {dataset} pelo registry")
    cohort = load_blocks(dataset, list(HAND8))
    labels_raw = np.asarray(cohort.labels, dtype=str)
    stems = [str(x) for x in cohort.stems]
    log(f"  {len(stems)} imagens, {cohort.dimensions}d, "
        f"{len(np.unique(labels_raw))} classes, blocos {list(cohort.blocks)}")

    log(f"lendo {tiled_dir}")
    tiled_stems = [str(x) for x in np.load(tiled_dir / "stems.npy")]
    if tiled_stems == stems:
        log("  stems.npy confere com cohort.stems (mesma ordem, sem reindex)")
        take = np.arange(len(stems))
    else:
        # ALINHAMENTO POR STEM, nunca por posicao (§4.1f). A extracao pode ser
        # SUBCONJUNTO da coorte -- no places_lt as 528 imagens que o §5.18d
        # registra como nunca materializadas sairam via `--exclude-stems`. O que
        # continua sendo abortar e o caso em que o tiled tem stem que a coorte
        # nao tem: ali as duas fontes sao coortes DIFERENTES, nao um recorte.
        tiled_at = {x: i for i, x in enumerate(tiled_stems)}
        if len(tiled_at) != len(tiled_stems):
            raise SystemExit(f"{dataset}: stems.npy do tiled tem repetidos")
        extra = sorted(set(tiled_stems) - set(stems))
        if extra:
            raise SystemExit(
                f"{dataset}: o tiled tem {len(extra)} stems que a coorte nao tem "
                f"(ex.: {extra[:3]}) -- sao coortes diferentes, nao um recorte")
        keep = np.array([i for i, x in enumerate(stems) if x in tiled_at])
        take = np.array([tiled_at[stems[i]] for i in keep])
        log(f"  o tiled e SUBCONJUNTO da coorte: {len(keep)} de {len(stems)} "
            f"({len(stems) - len(keep)} sem extracao). Reindexando por stem e "
            f"recortando os dois lados na intersecao.")
        for name in HAND8:
            cohort.blocks[name] = np.asarray(cohort.blocks[name])[keep]
        labels_raw = labels_raw[keep]
        stems = [stems[i] for i in keep]
        log(f"  depois do recorte: {len(stems)} imagens, "
            f"{len(np.unique(labels_raw))} classes")

    upscaled = np.load(tiled_dir / "upscaled.npy")[take]
    raw_names = json.loads((tiled_dir / "meta.json").read_text())["raw_names"]
    log(f"  {int(upscaled.sum())} de {len(upscaled)} imagens upscaled "
        f"({100 * upscaled.mean():.1f}%)")

    # MEMMAP + pooling em lotes. O arquivo do places_lt tem 19,6 GB e o §1 manda
    # dimensionar para maquina compartilhada: ler inteiro na RAM disputaria com o
    # que estiver na fila. Em lotes o pico e o do `pooled` (n x 2895 float32).
    log("abrindo features.npy (n, 16, 2895) e fazendo pooling por media dos tiles")
    started = time.time()
    tiles = np.load(tiled_dir / "features.npy", mmap_mode="r")
    log(f"  memmap {tiles.shape}, {tiles.nbytes / 1e9:.1f} GB em disco")
    pooled = np.empty((len(take), tiles.shape[2]), dtype=np.float32)
    for start in range(0, len(take), 4096):
        chunk = take[start:start + 4096]
        pooled[start:start + len(chunk)] = np.asarray(
            tiles[chunk], dtype=np.float32).mean(axis=1)
    del tiles
    pooled = clean(pooled)
    log(f"  pooled: {pooled.shape} em {time.time() - started:.0f}s")
    return {"blocks": cohort.blocks, "labels_raw": labels_raw, "stems": stems,
            "pooled": pooled, "upscaled": upscaled, "raw_names": raw_names,
            "dimensions": cohort.dimensions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--tiled-dir", type=Path,
                        default=ROOT / "outputs" / "multitask_wndcharm_tiled")
    parser.add_argument("--held-tiled-dir", type=Path, default=None,
                        help="tiled da coorte de TESTE separada (so nct_crc); "
                             "default outputs/<coorte_de_teste>_wndcharm_tiled")
    parser.add_argument("--force-cv", action="store_true",
                        help="forca 5-fold CV num corpus de split oficial. "
                             "Existe so para reproduzir a execucao antiga; o §7.5 "
                             "proibe isto no places_lt e o §7.14 no nct_crc.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "wndcharm_tiled_vs_hand8.csv")
    args = parser.parse_args()

    main_side = load_side(args.dataset, args.tiled_dir)
    held_side = None
    if args.dataset in TEST_COHORT and not args.force_cv:
        held_name = TEST_COHORT[args.dataset]
        held_dir = args.held_tiled_dir or (ROOT / "outputs" /
                                           f"{held_name}_wndcharm_tiled")
        if not (held_dir / "features.npy").exists():
            raise SystemExit(
                f"{args.dataset}: o teste e a coorte EXTERNA {held_name} (§7.14) e "
                f"{held_dir} nao existe. Extrair primeiro, ou passar --force-cv "
                f"para rodar a CV INTERNA -- que o §7.14 diz nao ser o numero "
                f"principal, porque nao controla paciente/slide.")
        log(f"\ncoorte de TESTE separada: {held_name}")
        held_side = load_side(held_name, held_dir)
        if held_side["raw_names"] != main_side["raw_names"]:
            raise SystemExit("os dois tiled tem conjuntos de feature diferentes")

    if held_side is None:
        blocks_by_name = main_side["blocks"]
        labels_raw = main_side["labels_raw"]
        stems = main_side["stems"]
        pooled = main_side["pooled"]
        upscaled = main_side["upscaled"]
        n_train_rows = None
    else:
        # Concatena treino e teste numa coorte so; o split abaixo e por POSICAO,
        # que aqui e seguro porque a fronteira e esta concatenacao e nao uma
        # reindexacao (cada lado ja foi alinhado por stem dentro do load_side).
        blocks_by_name = {n: np.vstack([np.asarray(main_side["blocks"][n]),
                                        np.asarray(held_side["blocks"][n])])
                          for n in HAND8}
        labels_raw = np.concatenate([main_side["labels_raw"], held_side["labels_raw"]])
        stems = main_side["stems"] + held_side["stems"]
        pooled = np.vstack([main_side["pooled"], held_side["pooled"]])
        upscaled = np.concatenate([main_side["upscaled"], held_side["upscaled"]])
        n_train_rows = len(main_side["stems"])
        log(f"  concatenado: {n_train_rows:,} treino + "
            f"{len(held_side['stems']):,} teste externo = {len(stems):,}")

    labels = LabelEncoder().fit_transform(labels_raw)
    raw_names = main_side["raw_names"]
    dimensions = main_side["dimensions"]

    alive = pooled.std(axis=0) > 0
    log(f"  colunas constantes na coorte (pooled): {int((~alive).sum())} de {pooled.shape[1]}")
    raw_alive = [n for n, keep in zip(raw_names, alive) if keep]
    pooled = pooled[:, alive]

    groups: dict[str, list[int]] = {}
    for position, raw in enumerate(raw_alive):
        groups.setdefault(algorithm_of(raw), []).append(position)
    log(f"  {len(groups)} familias de algoritmo: "
        + ", ".join(f"{k} ({len(v)})" for k, v in
                    sorted(groups.items(), key=lambda kv: -len(kv[1]))))

    hand8 = [blocks_by_name[name] for name in HAND8]
    charm15 = [pooled[:, positions] for positions in groups.values()]

    # Mesmo recorte de cor do avaliador single-tile (§10.2), conferido por
    # dentro: as colunas 99-116 do Descriptors sao LBP density e somam 1,0.
    descriptors = blocks_by_name["Descriptors"]
    COR = slice(0, 99)
    LBP = slice(99, 117)
    soma_lbp = descriptors[:, LBP].sum(axis=1)
    if not np.allclose(soma_lbp, 1.0, atol=1e-3):
        raise SystemExit(
            "colunas 99-116 do Descriptors nao somam 1,0 -- o layout do §10.2 "
            "nao vale neste cohort e o recorte de cor estaria errado")
    sem_cor = np.hstack([descriptors[:, LBP], descriptors[:, 117:]])
    cor = descriptors[:, COR]
    hand8_sem_cor = [sem_cor if name == "Descriptors" else blocks_by_name[name]
                     for name in HAND8]

    # PARTICAO: split publicado onde existe, CV onde nao existe. Nos corpora
    # oficiais a particao e UMA, e as 5 unidades sao 5 SEMENTES sobre ela -- o
    # `evaluate_mlp` usa `SEED + fold` tanto no recorte de validacao quanto na
    # inicializacao, entao repetir a mesma particao com fold=1..5 e exatamente o
    # protocolo de 5 sementes que o §5.18e usa. Sem isso o delta sairia sem
    # barra e sem teste pareado.
    if held_side is not None:
        # Teste e a coorte EXTERNA inteira; a fronteira e a concatenacao acima.
        train_idx = np.arange(n_train_rows)
        test_idx = np.arange(n_train_rows, len(labels))
        splits = [(train_idx, test_idx)] * 5
        log(f"coorte EXTERNA: {len(train_idx):,} treino / {len(test_idx):,} teste "
            f"-- 1 particao x 5 sementes (conferir contra o numero publicado)")
    elif args.force_cv or args.dataset not in OFFICIAL:
        splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(labels)), labels))
        log(f"CV: 5 folds sobre {len(labels):,} imagens"
            + ("  (--force-cv num corpus de split oficial!)" if args.force_cv else ""))
    else:
        where = (pd.read_csv(ROOT / LABELS[args.dataset][0], dtype={"stem": str})
                 .set_index("stem")["split"].reindex(stems).to_numpy(str))
        if (where == "nan").any():
            raise SystemExit(f"{args.dataset}: {int((where == 'nan').sum())} stems "
                             f"da coorte nao resolvem na coluna `split`")
        if not (where == "test").any():
            raise SystemExit(f"{args.dataset}: coluna `split` sem linhas de teste")
        if (where == "val").any():
            log(f"  o `val` oficial ({int((where == 'val').sum()):,} linhas) fica "
                f"FORA: o protocolo publicado nao o usa como treino, e o proprio "
                f"fit separa 10% do treino para validacao (§5.18e)")

        # GUARDA DE CLASSE RARA, refeita DEPOIS do corte por falta de imagem --
        # contar no metadado nao serve (§5.18e). No places_lt as 528 faltantes
        # sao todas de TREINO e deixam 3 classes com <2 linhas la: o split
        # estratificado de 10% do `evaluate_mlp` quebra, e manter a classe so no
        # teste daria recall zero numa classe que o modelo nao pode acertar,
        # deprimindo a BAcc por artefato de contagem em vez de desempenho.
        counts = np.bincount(labels[where == "train"], minlength=labels.max() + 1)
        thin = np.flatnonzero(counts < 2)
        if len(thin):
            keep_rows = np.flatnonzero(~np.isin(labels, thin))
            log(f"  !! {len(thin)} CLASSES com <2 linhas de treino saem de treino "
                f"E teste: {sorted(np.unique(labels_raw[np.isin(labels, thin)]))}")
            for name in HAND8:
                blocks_by_name[name] = np.asarray(blocks_by_name[name])[keep_rows]
            pooled = pooled[keep_rows]
            upscaled = upscaled[keep_rows]
            stems = [stems[i] for i in keep_rows]
            where = where[keep_rows]
            labels = LabelEncoder().fit_transform(labels_raw[keep_rows])
            labels_raw = labels_raw[keep_rows]

        train_idx = np.flatnonzero(where == "train")
        test_idx = np.flatnonzero(where == "test")
        splits = [(train_idx, test_idx)] * 5
        # Imprimir os tamanhos E o antidoto do §8: o bug antigo sobreviveu
        # porque NADA na saida denunciava.
        tr_counts = np.bincount(labels[train_idx])
        log(f"split OFICIAL: {len(train_idx):,} treino / {len(test_idx):,} teste / "
            f"{len(np.unique(labels))} classes -- 1 particao x 5 sementes "
            f"(conferir contra o numero publicado)")
        log(f"  desbalanceamento no treino: {tr_counts.max() / tr_counts.min():.0f}x "
            f"(min {tr_counts.min()}, max {tr_counts.max()}) -- no places_lt a CV "
            f"derrubaria isto de 996x para 41x e apagaria o regime (§7.5)")
    rows: list[dict] = []

    fixed_conditions = {
        "hand-8": hand8,
        "hand-8 sem cor": hand8_sem_cor,
        "WND-CHARM tiled (1 bloco, pooled)": [pooled],
        "WND-CHARM tiled (15 blocos, pooled)": charm15,
        "WND-CHARM tiled (15) + cor do Descriptors": charm15 + [cor],
        "hand-8 + WND-CHARM tiled (15, pooled)": hand8 + charm15,
    }
    for name, blocks in fixed_conditions.items():
        run_condition(name, blocks, labels, splits, rows, args.output, args.folds)

    log(f"\nFisher (ANOVA F, so no treino de cada fold), top {100*FISHER_KEEP_FRAC:.0f}% -- "
        f"o passo de reducao de dimensao do protocolo canonico")
    fisher_dims = []
    for fold, (train, test) in enumerate(splits[:args.folds], start=1):
        keep = fisher_select(pooled[train], labels[train], FISHER_KEEP_FRAC)
        fisher_dims.append(len(keep))
        block = pooled[:, keep]

        started = time.perf_counter()
        metrics = evaluate_mlp([block], labels, train, test, fold, epochs=150, patience=15)
        rows.append({
            "condition": "WND-CHARM tiled (Fisher top15%, pooled)", "fold": fold,
            "n_blocks": 1, "dimensions": block.shape[1],
            "seconds": time.perf_counter() - started,
            "bacc": 100 * metrics["bacc"], "accuracy": 100 * metrics["accuracy"],
            "macro_f1": 100 * metrics["macro_f1"],
        })
        log(f"  WND-CHARM tiled (Fisher top15%, pooled)      f{fold}   1b "
            f"{block.shape[1]:>5}d  BAcc {rows[-1]['bacc']:5.2f}  "
            f"({rows[-1]['seconds']:.0f}s)")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

        started = time.perf_counter()
        metrics = evaluate_mlp(hand8 + [block], labels, train, test, fold,
                               epochs=150, patience=15)
        rows.append({
            "condition": "hand-8 + WND-CHARM tiled (Fisher top15%)", "fold": fold,
            "n_blocks": 9, "dimensions": dimensions + block.shape[1],
            "seconds": time.perf_counter() - started,
            "bacc": 100 * metrics["bacc"], "accuracy": 100 * metrics["accuracy"],
            "macro_f1": 100 * metrics["macro_f1"],
        })
        log(f"  hand-8 + WND-CHARM tiled (Fisher top15%)      f{fold}   9b "
            f"{rows[-1]['dimensions']:>5}d  BAcc {rows[-1]['bacc']:5.2f}  "
            f"({rows[-1]['seconds']:.0f}s)")
        pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

        # Sensibilidade: mesmo modelo (o split de TREINO nao muda), teste
        # restrito ao subconjunto NATIVO (nao upscaled). So custa retreino de
        # wall-clock -- nao reintroduz escolha nova, e o §5.18g so confia em
        # decomposicao assim quando a direcao do teste esta declarada antes.
        native_test = test[~upscaled[test]]
        if fold == 1 and (not len(native_test)
                          or len(np.unique(labels[native_test])) <= 1):
            # Condicao que desaparece sem aviso e o modo de falha do §8. No
            # places_lt a coorte e 100% upscaled (as imagens sao de 256 px e o
            # protocolo padroniza em 600), entao o subconjunto nativo e VAZIO e
            # esta sensibilidade nao existe ali -- tem de estar dito.
            log(f"  [so NATIVO] PULADO: {len(native_test)} imagens nativas no "
                f"teste de {len(test)} -- sem sensibilidade de upscale neste corpus")
        if len(native_test) and len(np.unique(labels[native_test])) > 1:
            started = time.perf_counter()
            metrics_native = evaluate_mlp(hand8 + [block], labels, train, native_test,
                                          fold, epochs=150, patience=15)
            rows.append({
                "condition": "hand-8 + WND-CHARM tiled (Fisher top15%) [so NATIVO no teste]",
                "fold": fold, "n_blocks": 9, "dimensions": dimensions + block.shape[1],
                "seconds": time.perf_counter() - started,
                "bacc": 100 * metrics_native["bacc"],
                "accuracy": 100 * metrics_native["accuracy"],
                "macro_f1": 100 * metrics_native["macro_f1"],
                "n_test_native": len(native_test), "n_test_total": len(test),
            })
            log(f"  ...[so NATIVO, {len(native_test)}/{len(test)} do teste]     f{fold}"
                f"       BAcc {rows[-1]['bacc']:5.2f}")
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")

    log(f"\ndimensoes pos-Fisher por fold: {fisher_dims} "
        f"(de {pooled.shape[1]} vivas; alvo canonico do artigo original era 548, "
        f"mas o total de features difere -- 2895 monocromatico aqui contra ~3659-4008 "
        f"do artigo com cor, entao 548 nao e o numero esperado aqui)")

    table = pd.DataFrame(rows)
    summary = table.groupby("condition", sort=False).agg(
        blocos=("n_blocks", "first"), dims=("dimensions", "first"),
        bacc=("bacc", "mean"), sd=("bacc", "std"),
        acc=("accuracy", "mean"), f1=("macro_f1", "mean"),
        s=("seconds", "mean"))
    log("\n" + summary.round(2).to_string())

    base = table[table.condition.eq("hand-8")].set_index("fold")["bacc"]
    banda = BANDA_HAND8.get(args.dataset)
    if banda is None:
        log(f"\nhand-8 em {base.mean():.2f}; sem referencia publicada para "
            f"{args.dataset}, nada a conferir")
    elif args.force_cv and args.dataset in OFFICIAL:
        log(f"\nhand-8 em {base.mean():.2f}; a banda {banda} e do SPLIT OFICIAL e "
            f"isto rodou com --force-cv -- protocolos diferentes, nao comparar")
    elif not banda[0] <= base.mean() <= banda[1]:
        log(f"\n!! hand-8 em {base.mean():.2f}, fora da banda {banda} do §5.18e -- "
            f"o caminho NAO e o de producao; nao reportar os deltas")
    else:
        log(f"\nhand-8 em {base.mean():.2f}: dentro da banda {banda} do §5.18e, "
            f"caminho validado")

    unidade = "fold" if (args.force_cv or args.dataset not in OFFICIAL) else "semente"
    log(f"\ndelta contra hand-8, pareado por {unidade} (so condicoes com 5 medicoes):")
    counts = table.groupby("condition")["fold"].count()
    for name in counts.index:
        if name == "hand-8" or counts[name] != len(base):
            continue
        other = table[table.condition.eq(name)].set_index("fold")["bacc"]
        delta = other.reindex(base.index) - base
        p = stats.ttest_rel(other.reindex(base.index), base).pvalue if len(delta) > 1 else np.nan
        log(f"  {name:<55} {delta.mean():+6.2f} +- {delta.std():4.2f}  "
            f"p={p:.4f}  {int((delta > 0).sum())}/{len(delta)}")

    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
