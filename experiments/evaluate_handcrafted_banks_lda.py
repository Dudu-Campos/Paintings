"""Haralick x HOG x WND-CHARM x hand-8, sob sonda LINEAR.

Os quatro sao bancos hand-designed, entao a comparacao e DENTRO da familia e o
§7.19 diz que a sonda linear e o instrumento (o §5.9d valida: a ordenacao se
conserva nos cinco classificadores, `sozinha (LDA) x Shapley (LDA)` = +1,000).

**DUAS VARIANTES DE LDA, e aqui isso NAO e zelo excessivo.** O §7.19 mediu que o
shrinkage de Ledoit-Wolf nao e neutro entre representacoes: +1,98pp num
embedding denso de 2.048d e -0,32pp no hand-8 multi-bloco de 617d. Nesta tabela
as dimensionalidades vao de **13 a 2.895 (222x)** e os bancos diferem em
estrutura (Haralick e 13 escalares; hand-8 sao 8 blocos; HOG e WND-CHARM sao
densos e altos), ou seja e exatamente o regime onde a escolha da sonda pode
ordenar a tabela. Rodar as duas e conferir se a ORDEM muda e a guarda.

  svd            = `protocol.evaluate_lda` de producao, sem shrinkage
  lsqr+shrinkage = `evaluate_minc2500_official.fit_lda`, o do §7.15

Protocolo por corpus, o mesmo do resto do projeto (§7.5, §9.1):
  split PUBLICADO onde existe · coorte de teste SEPARADA no nct_crc (§7.14) ·
  5 folds x N sementes nos tres de arte

WND-CHARM entra no protocolo TILED (16 tiles de 150x150 sobre padronizacao
600x600, Shamir et al. 2010), com os tiles agregados por media -- e a agregacao
por media e adaptacao declarada, nao o WND5 do artigo.

Uso:
    python experiments/evaluate_handcrafted_banks_lda.py --dataset oilpainting
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
from corpora import HAND8                      # noqa: E402
from registry import load_blocks               # noqa: E402

SEED, N_FOLDS = 42, 5
OFFICIAL = {"minc2500", "places_lt", "food101", "nct_crc"}
TEST_COHORT = {"nct_crc": "nct_crc_val"}
HARALICK_COLS, HOG_COLS = slice(0, 13), slice(13, None)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def by_stem(path: Path, stems: list[str]) -> np.ndarray | None:
    """Carrega `features.npy` + `stems.npy` e REINDEXA por stem (§4.1f)."""
    if not (path / "features.npy").exists():
        return None
    values = np.load(path / "features.npy", mmap_mode="r")
    at = {str(s): i for i, s in enumerate(np.load(path / "stems.npy", allow_pickle=True))}
    faltam = [s for s in stems if s not in at]
    if faltam:
        log(f"  !! {path.name}: {len(faltam)} stems ausentes (ex. {faltam[:2]}) -- pulando")
        return None
    return np.asarray(values[np.array([at[s] for s in stems])], dtype=np.float32)


def pooled_wndcharm(ds: str, stems: list[str]) -> np.ndarray | None:
    """Media dos 16 tiles, com CACHE -- repoolar 19,6 GB a cada corpus nao paga."""
    tiled = ROOT / f"outputs/{ds}_wndcharm_tiled"
    if not (tiled / "features.npy").exists():
        return None
    cache = ROOT / f"outputs/{ds}_wndcharm_tiled/pooled.npy"
    cache_stems = ROOT / f"outputs/{ds}_wndcharm_tiled/pooled_stems.npy"
    if cache.exists() and cache_stems.exists():
        at = {str(s): i for i, s in enumerate(np.load(cache_stems, allow_pickle=True))}
        if all(s in at for s in stems):
            return np.asarray(np.load(cache, mmap_mode="r")
                              [np.array([at[s] for s in stems])], dtype=np.float32)
    log(f"  poolando {tiled.name} (uma vez; depois fica em cache)")
    src = np.load(tiled / "features.npy", mmap_mode="r")
    out = np.empty((src.shape[0], src.shape[2]), dtype=np.float32)
    for i in range(0, src.shape[0], 4096):
        out[i:i + 4096] = np.asarray(src[i:i + 4096], dtype=np.float32).mean(axis=1)
    np.save(cache, out)
    np.save(cache_stems, np.load(tiled / "stems.npy", allow_pickle=True))
    at = {str(s): i for i, s in enumerate(np.load(tiled / "stems.npy", allow_pickle=True))}
    faltam = [s for s in stems if s not in at]
    if faltam:
        log(f"  !! WND-CHARM: {len(faltam)} stems ausentes -- pulando")
        return None
    return out[np.array([at[s] for s in stems])]


def _limpo(X: np.ndarray, train: np.ndarray, chunk: int = 2048) -> bool:
    """Todo finito E nenhuma coluna constante no treino? Em pedacos de colunas,
    para nao alocar um bool do tamanho da matriz."""
    for i in range(0, X.shape[1], chunk):
        bloco = X[:, i:i + chunk]
        if not np.isfinite(bloco).all():
            return False
        if not bool((bloco[train].std(axis=0) > 0).all()):
            return False
    return True


def fit_probe(X, y, train, test, variante: str, n_classes: int,
              priors: str = "uniform") -> dict:
    """Imputa, padroniza e ajusta. Colunas constantes no TREINO saem antes:
    com 2.895 features ha muitas, e o `svd` do LDA avisa colinearidade nelas."""
    # +-inf vira NaN ANTES do imputer: ele trata NaN e NAO trata infinito, e o
    # WND-CHARM poolado tem os dois (razoes e logs de estatisticas de tile que
    # degeneram em tile uniforme). O avaliador antigo usava `corpora.clean` para
    # isto; aqui a mediana do TREINO substitui, que e estritamente melhor porque
    # nao vaza o teste.
    # CAMINHO LEAN, para banco ja finito e sem coluna constante no treino: o
    # imputer e as fatias seguintes dao QUATRO copias, e com os 33.385d da fusao
    # `hand-8 + Fisher vector` cada copia e 20 GB em float64 -- o chain ingenuo
    # pediria ~70 GB de pico para devolver exatamente a mesma matriz. Mesma
    # guarda (e mesma justificativa) do `limpar` do §5.19b. A checagem de
    # finitude vai em pedacos: `np.isfinite(X)` inteiro aloca 3,4 GB de bool.
    if _limpo(X, train):
        tr = np.array(X[train], dtype=np.float64, order="C")
        te = np.array(X[test], dtype=np.float64, order="C")
        med, dp = tr.mean(axis=0), tr.std(axis=0)
        for A in (tr, te):                       # padroniza NO LUGAR
            A -= med
            A /= dp
        pri = None if priors == "empirical" else np.ones(n_classes) / n_classes
        model = (LDA(priors=pri) if variante == "svd"
                 else LDA(solver="lsqr", shrinkage="auto", priors=pri))
        pred = model.fit(tr, y[train]).predict(te)
        del tr, te
        truth = y[test]
        return ({"dims_vivas": int(X.shape[1]),
                 "bacc": 100 * balanced_accuracy_score(truth, pred),
                 "accuracy": 100 * accuracy_score(truth, pred),
                 "macro_f1": 100 * f1_score(truth, pred, average="macro",
                                            zero_division=0)}, pred, truth)

    tr = np.where(np.isfinite(X[train]), X[train], np.nan)
    te = np.where(np.isfinite(X[test]), X[test], np.nan)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(tr)
    tr, te = imp.transform(tr), imp.transform(te)
    viva = tr.std(axis=0) > 0
    tr, te = tr[:, viva], te[:, viva]
    sc = StandardScaler().fit(tr)
    tr, te = sc.transform(tr), sc.transform(te)
    # `priors` NAO e o `class_weight` do §7.12: o LDA nao minimiza risco empirico
    # de metrica nenhuma. Medido em 07/out, por solver:
    #   svd            o priors cancela na DIRECAO da fronteira (cos = 1,000000)
    #                  e sobra `log(priors)` no intercepto -- limiar puro
    #   lsqr+shrinkage o priors tambem PONDERA a covariancia agrupada
    #                  (`_class_cov(X, y, priors, shrinkage)`), usada em Σ⁻¹μ,
    #                  entao a fronteira roda (cos = 0,991 num brinquedo 30x)
    # `uniform` e a regra de Bayes plug-in para classes equiprovaveis, otima para
    # BAcc SE o modelo gaussiano estivesse correto -- esta mal especificado.
    # `empirical` e a regra orientada a Acc, e e o outro lado do 2x2 do §7.12.
    pri = None if priors == "empirical" else np.ones(n_classes) / n_classes
    model = (LDA(priors=pri) if variante == "svd"
             else LDA(solver="lsqr", shrinkage="auto", priors=pri))
    pred = model.fit(tr, y[train]).predict(te)
    truth = y[test]
    met = {"dims_vivas": int(viva.sum()),
           "bacc": 100 * balanced_accuracy_score(truth, pred),
           "accuracy": 100 * accuracy_score(truth, pred),
           "macro_f1": 100 * f1_score(truth, pred, average="macro", zero_division=0)}
    return met, pred, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="oilpainting")
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--variants", nargs="+", default=["svd", "shrinkage"],
                    choices=["svd", "shrinkage"])
    ap.add_argument("--priors", default="uniform", choices=["uniform", "empirical"],
                    help="uniform = regra de Bayes para classes equiprovaveis (o "
                         "publicado); empirical = a regra orientada a Acc. NAO e "
                         "class_weight -- ver o comentario em fit_probe.")
    ap.add_argument("--fusoes", nargs="*", default=[],
                    help="bancos a concatenar com o hand-8, um por condicao "
                         "(ex.: --fusoes 'Fisher vector'). A fusao e montada "
                         "JUST-IN-TIME e liberada depois: materializar as duas "
                         "ao mesmo tempo custaria 27 GB no food101.")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--pred-dir", type=Path, default=None,
                    help="grava a predicao por imagem de cada celula. Sem ela nao ha "
                         "como reamostrar o teste, que e a UNICA fonte de incerteza "
                         "nos corpora de split fixo (o LDA e deterministico).")
    args = ap.parse_args()
    ds = args.dataset
    out = args.output or ROOT / f"results/{ds}_handcrafted_banks_lda.csv"

    cohort = load_blocks(ds, list(HAND8))
    stems = [str(s) for s in cohort.stems]
    labels_raw = np.asarray(cohort.labels, dtype=str)
    blocks = {n: np.asarray(cohort.blocks[n]) for n in HAND8}
    n_train_rows = None

    if ds in TEST_COHORT:
        held = load_blocks(TEST_COHORT[ds], list(HAND8), allow_external=True)
        n_train_rows = len(stems)
        held_stems = [str(s) for s in held.stems]
        blocks = {n: np.vstack([blocks[n], np.asarray(held.blocks[n])]) for n in HAND8}
        labels_raw = np.concatenate([labels_raw, np.asarray(held.labels, dtype=str)])
        log(f"{ds}: coorte EXTERNA {TEST_COHORT[ds]} como teste "
            f"({n_train_rows:,} + {len(held_stems):,})")

    y = LabelEncoder().fit_transform(labels_raw)
    C = len(np.unique(y))
    hand8 = np.hstack([blocks[n] for n in HAND8]).astype(np.float32)

    # --- os outros tres bancos, cada um alinhado por stem ---
    bancos: dict[str, np.ndarray] = {"hand-8": hand8}
    if ds in TEST_COHORT:
        cls_tr = by_stem(ROOT / f"outputs/{ds}_haralick_hog", stems)
        cls_te = by_stem(ROOT / f"outputs/{TEST_COHORT[ds]}_haralick_hog", held_stems)
        classico = np.vstack([cls_tr, cls_te]) if cls_tr is not None and cls_te is not None else None
        wnd_tr = pooled_wndcharm(ds, stems)
        wnd_te = pooled_wndcharm(TEST_COHORT[ds], held_stems)
        wnd = np.vstack([wnd_tr, wnd_te]) if wnd_tr is not None and wnd_te is not None else None
        todos_stems = stems + held_stems
    else:
        classico = by_stem(ROOT / f"outputs/{ds}_haralick_hog", stems)
        wnd = pooled_wndcharm(ds, stems)
        todos_stems = stems
    if classico is not None:
        bancos["Haralick"] = classico[:, HARALICK_COLS]
        bancos["HOG"] = classico[:, HOG_COLS]
    if wnd is not None:
        bancos["WND-CHARM"] = wnd
    # Fisher vector: entra pelo mesmo caminho dos outros (alinhado por stem) e
    # so se o cache existir, igual ao WND-CHARM. §7.19 autoriza a sonda linear
    # aqui porque ele e hand-designed -- a PCA e a GMM nao veem rotulo.
    if ds in TEST_COHORT:
        fv_tr = by_stem(ROOT / f"outputs/{ds}_fisher_vector", stems)
        fv_te = by_stem(ROOT / f"outputs/{TEST_COHORT[ds]}_fisher_vector", held_stems)
        fv = np.vstack([fv_tr, fv_te]) if fv_tr is not None and fv_te is not None else None
    else:
        fv = by_stem(ROOT / f"outputs/{ds}_fisher_vector", stems)
    if fv is not None:
        bancos["Fisher vector"] = fv
    log(f"{ds}: {len(y):,} linhas, {C} classes | bancos: "
        + ", ".join(f"{k} {v.shape[1]}d" for k, v in bancos.items()))

    # --- particao, com os tamanhos IMPRESSOS (o antidoto do §8) ---
    if ds in TEST_COHORT:
        parts = [(args.seeds[0], 1, np.arange(n_train_rows), np.arange(n_train_rows, len(y)))]
        tipo = "coorte externa"
    elif ds in OFFICIAL:
        meta = pd.read_csv(ROOT / f"data/{ds}_meta.csv", dtype={"stem": str})[["stem", "split"]]
        where = meta.set_index("stem")["split"].reindex(todos_stems).to_numpy(str)
        if (where == "nan").any():
            raise SystemExit(f"{ds}: {(where == 'nan').sum()} stems fora do metadado")
        # guarda de classe rara DEPOIS do corte (§5.18e): no places_lt 3 classes
        # ficam com <2 linhas de treino e quebrariam a estratificacao
        cnt = np.bincount(y[where == "train"], minlength=y.max() + 1)
        thin = np.flatnonzero(cnt < 2)
        if len(thin):
            keep = np.flatnonzero(~np.isin(y, thin))
            log(f"  !! {len(thin)} classes com <2 linhas de treino saem de treino E teste")
            bancos = {k: v[keep] for k, v in bancos.items()}
            where, y = where[keep], LabelEncoder().fit_transform(labels_raw[keep])
            # os stems acompanham o corte: `y`/`where`/`bancos` passam a ser o
            # subconjunto, e um `todos_stems` cheio faria os indices de teste
            # apontarem para o stem errado -- silenciosamente, porque o tamanho
            # nao e conferido em lugar nenhum (§4.1f item 7: alinhar por stem,
            # nunca por posicao). So o places_lt chega aqui (3 classes saem).
            todos_stems = [todos_stems[i] for i in keep]
            C = len(np.unique(y))
        parts = [(args.seeds[0], 1, np.flatnonzero(where == "train"),
                  np.flatnonzero(where == "test"))]
        tipo = "split OFICIAL"
    else:
        parts = [(sd, f, tr, te) for sd in args.seeds
                 for f, (tr, te) in enumerate(
                     StratifiedKFold(N_FOLDS, shuffle=True, random_state=sd)
                     .split(np.zeros(len(y)), y), start=1)]
        tipo = f"CV {N_FOLDS} folds x {len(args.seeds)} sementes"
    log(f"  [{tipo}] treino={len(parts[0][2]):,} teste={len(parts[0][3]):,} classes={C}")
    if ds in OFFICIAL and len(args.seeds) > 1:
        log(f"  !! particao FIXA e LDA deterministico: as sementes extras dariam "
            f"numero identico; rodando so a {args.seeds[0]}")

    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.banco, r.variante, int(r.seed), int(r.fold)) for r in prev.itertuples()}
        log(f"  retomando {len(done)} linhas")
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []

    def caminho_pred(nome: str, var: str, seed: int, fold: int) -> Path | None:
        if args.pred_dir is None:
            return None
        slug = re.sub(r"[^0-9A-Za-z]+", "_", nome).strip("_")
        return args.pred_dir / f"{ds}__{slug}__{var}__s{seed}_f{fold}.npz"

    # Uma linha no CSV NAO basta para considerar a celula feita quando se pede
    # predicao: as sete execucoes de 07/out gravaram metrica e descartaram
    # `pred`, entao retomar pelo CSV sozinho deixaria o bootstrap sem entrada e
    # sem nada na saida denunciando (§8).
    if args.pred_dir is not None:
        args.pred_dir.mkdir(parents=True, exist_ok=True)
        done = {k for k in done if (caminho_pred(k[0], k[1], k[2], k[3]) or Path("/")).exists()}
        log(f"  com --pred-dir, {len(done)} celulas tem metrica E predicao em disco")

    faltam = [f for f in args.fusoes if f not in bancos]
    if faltam:
        raise SystemExit(f"--fusoes: {faltam} nao esta entre os bancos carregados "
                         f"({list(bancos)})")
    condicoes = [(n, [n]) for n in bancos] + \
                [(f"hand-8 + {f}", ["hand-8", f]) for f in args.fusoes]

    for seed, fold, tr, te in parts:
        for nome, partes in condicoes:
            for var in args.variants:
                if (nome, var, seed, fold) in done:
                    continue
                t0 = time.perf_counter()
                # just-in-time: so a condicao corrente existe materializada
                X = (bancos[partes[0]] if len(partes) == 1
                     else np.hstack([bancos[q] for q in partes]))
                dims_cond = X.shape[1]
                m, pred, truth = fit_probe(X, y, tr, te, var, C,
                                           priors=args.priors)
                rows = [r for r in rows if not (r["banco"] == nome
                        and r["variante"] == var and int(r["seed"]) == seed
                        and int(r["fold"]) == fold)]
                rows.append({"dataset": ds, "banco": nome, "variante": var,
                             "seed": seed, "fold": fold, "dims": dims_cond,
                             "seconds": time.perf_counter() - t0, **m})
                pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
                alvo = caminho_pred(nome, var, seed, fold)
                if alvo is not None:
                    # stems junto: sem eles o pareamento entre bancos seria por
                    # POSICAO, que e o bug de alinhamento do §4.1f item 7.
                    np.savez_compressed(
                        str(alvo)[:-4] if str(alvo).endswith(".npz") else str(alvo),
                        pred=pred, truth=truth,
                        stems=np.asarray(todos_stems, dtype=object)[te])
                if len(partes) > 1:
                    del X
                log(f"  s{seed} f{fold} {nome:11s} {var:10s} {dims_cond:5d}d "
                    f"({m['dims_vivas']:5d} vivas)  BAcc {m['bacc']:5.2f}  "
                    f"({rows[-1]['seconds']:.0f}s)")

    t = pd.DataFrame(rows)
    log("\n=== BAcc por banco e variante de sonda ===")
    piv = t.pivot_table(index="banco", columns="variante", values="bacc")
    piv["dims"] = t.groupby("banco")["dims"].first()
    log(piv.sort_values(args.variants[0], ascending=False).round(2).to_string())
    if len(args.variants) > 1:
        a = t[t.variante.eq("svd")].groupby("banco")["bacc"].mean().sort_values(ascending=False)
        b = t[t.variante.eq("shrinkage")].groupby("banco")["bacc"].mean().sort_values(ascending=False)
        log(f"\nordem svd       : {list(a.index)}")
        log(f"ordem shrinkage : {list(b.index)}")
        log(f"A ORDEM {'SE CONSERVA' if list(a.index) == list(b.index) else 'MUDA'} "
            f"entre as duas sondas.")
    log(f"\n{out}")


if __name__ == "__main__":
    main()
