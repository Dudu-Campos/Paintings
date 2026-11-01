"""hand-8 x WND-CHARM sob MLP SEM BRANCH: o LDA tem capacidade para features heterogeneas?

Pergunta do usuario (07/out): *"somos mais rapidos e temos menos features, mas
nao gostaria de empatar com o wnd, avalie os dois com um mlp sem branch, quero
ver se o lda tem pouca capacidade de extrair as relacoes entre features
heterogeneas"*.

**O §8 tem a previsao registrada, e ela e exatamente essa hipotese:** *"LDA e MLP
concordam em <=0,35pp em vetores densos e divergem ~9pp em vetores multi-bloco"*,
com seis casos de "LDA extrai, MLP destroi". O WND-CHARM e UM banco denso de
2.895 colunas homogeneas; o hand-8 sao OITO blocos de medidas diferentes. Se o
empate do rijksdraw (§5.19) for falta de capacidade da sonda, o MLP separa.

**SEM BRANCH e exigencia do usuario, e ela e metodologicamente necessaria.** A
primeira camada bloco-diagonal vale +0,93pp sobre o denso de mesma largura
(§5.16) e existe PORQUE o hand-8 tem blocos -- dar branches a nos e nao ao
WND-CHARM mediria a arquitetura, nao os bancos. Com um bloco unico as duas
condicoes rodam a MESMA rede.

Dois bracos de largura, porque o gargalo nao e simetrico:
  denso-128   Linear(d,128) -> 512-256-128   (BranchMLP com 1 bloco)
              comprime o hand-8 4,8x e o WND-CHARM **22,6x**
  denso-512   Linear(d,512) -> 2048-1024-1024 (WideBranchMLP com 1 bloco)
              comprime 1,2x e 5,7x
Sem o segundo braco, um resultado a nosso favor seria confundido com "o encoder
de 128 estrangula o banco maior". O §4.1e mede que largura extra num bloco nao
compra nada (+0,15pp n.s.) e que a fatia proporcional **piora** (-0,87pp), mas
aquilo foi com enc=512 e compressao de 4x -- nao cobre 22,6x.

Pre-processamento IDENTICO ao `fit_probe` do §5.19 (nao-finitos -> NaN, mediana
do TREINO externo, colunas constantes fora), para que a unica diferenca entre as
duas sondas seja a sonda. Diferenca inerente e declarada: o LDA padroniza no
treino externo inteiro e o MLP no 90% interno, porque ele precisa de validacao
para escolher a epoca.

As TRES metricas saem juntas: o §5.19 mede que a BAcc inverte de sinal contra o
WND-CHARM em 2 de 6 corpora, entao reportar so ela aqui repetiria o erro.

Uso:
    python experiments/evaluate_banks_dense_mlp.py --datasets rijksdraw
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import re
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
from corpora import HAND8                                      # noqa: E402
from models import BranchMLP, LinearProbe, WideBranchMLP         # noqa: E402
from protocol import evaluate_mlp                               # noqa: E402
from registry import load_blocks                                # noqa: E402
from evaluate_handcrafted_banks_lda import (                    # noqa: E402
    HARALICK_COLS, HOG_COLS, OFFICIAL, TEST_COHORT, by_stem, pooled_wndcharm)

SEED, N_FOLDS, EPOCHS, PATIENCE = 42, 5, 150, 15
BANCOS = ["hand-8", "Haralick", "HOG", "WND-CHARM", "Fisher vector",
          "Fisher vector (celula global)"]

ARQ = {
    "denso-128": BranchMLP,
    "denso-512": partial(WideBranchMLP, enc=512, head=(2048, 1024, 1024)),
    # braco LINEAR: a sonda do protocolo publicado do Fisher vector, pelo mesmo
    # caminho de treino dos outros dois (ver LinearProbe em models.py).
    "linear": LinearProbe,
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def limpar(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Mesmo tratamento do `fit_probe`: nao-finitos -> NaN, mediana do TREINO,
    colunas constantes no treino fora. Sem isto o MLP recebe inf e a loss vira
    NaN em silencio -- o WND-CHARM poolado tem os dois."""
    # CAMINHO RAPIDO: um banco ja finito e sem coluna constante no treino nao
    # precisa de nenhuma das tres copias. Nao e otimizacao gratuita -- o Fisher
    # vector sao 13,2 GB por copia no food101, e o caminho ingenuo pediria ~40 GB
    # de pico para devolver exatamente a mesma matriz.
    if np.isfinite(X).all() and bool((X[train].std(axis=0) > 0).all()):
        return np.ascontiguousarray(X, dtype=np.float32)
    Z = np.where(np.isfinite(X), X, np.nan)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Z[train])
    Z = imp.transform(Z)
    viva = Z[train].std(axis=0) > 0
    return np.ascontiguousarray(Z[:, viva], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["rijksdraw"])
    ap.add_argument("--arquiteturas", nargs="+", default=list(ARQ),
                    choices=list(ARQ))
    ap.add_argument("--bancos", nargs="+", default=BANCOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="avalia so estes folds (1..5). Existe para o controle "
                         "indutivo: o codebook ajustado no treino do fold 1 so e "
                         "honesto NO fold 1 -- rodar os outros quatro com ele "
                         "seria transdutivo de novo, e sem nada na saida denunciar.")
    ap.add_argument("--fusoes", nargs="*", default=[],
                    help="bancos a concatenar com o hand-8, um por condicao. "
                         "Montada JUST-IN-TIME e liberada depois: o Fisher vector "
                         "sozinho sao 8,1 GB no rijksdraw.")
    ap.add_argument("--pred-dir", type=Path, default=None,
                    help="grava pred/truth/stems por celula. Sem isso nao ha como "
                         "medir sobreposicao de erro (Q de Yule) nem decompor "
                         "quem acerta o que -- e a pergunta 'eles sao iguais?' e "
                         "exatamente essa.")
    ap.add_argument("--fv-sufixo", default="",
                    help="le o Fisher vector de `outputs/{ds}_fisher_vector{sufixo}` "
                         "e ROTULA o banco com o sufixo. Existe para comparar "
                         "configuracoes de FV sem sobrescrever cache nenhum -- cada "
                         "uma custa ~4,8 h de SIFT no food101.")
    ap.add_argument("--descartar-faltantes", action="store_true",
                    help="quando o banco nao cobre a coorte inteira, REMOVE as "
                         "linhas sem banco em vez de pular o corpus. So com UM "
                         "banco pedido: com varios, cortar por um dessincroniza "
                         "os outros. Imprime o que caiu, por split -- o §5.18e "
                         "faz exatamente este corte no places_lt e declara.")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/banks_dense_mlp.csv")
    args = ap.parse_args()

    rows = pd.read_csv(args.output).to_dict("records") if args.output.exists() else []
    done = {(r["dataset"], r["banco"], r["arquitetura"],
             int(r.get("seed", SEED)), int(r["fold"])) for r in rows}

    def caminho_pred(ds: str, nome: str, arq: str, seed: int, fold: int) -> Path:
        slug = re.sub(r"[^0-9A-Za-z]+", "_", f"{nome}__{arq}").strip("_")
        return args.pred_dir / f"{ds}__{slug}__s{seed}_f{fold}.npz"

    # Uma linha no CSV NAO basta quando se pede predicao: as corridas de hoje
    # gravaram metrica e descartaram `pred`, entao retomar pelo CSV deixaria a
    # sobreposicao de erro sem entrada e SEM NADA NA SAIDA DENUNCIANDO (§8). O
    # script de LDA ja tinha esta guarda; este nao.
    if args.pred_dir is not None:
        args.pred_dir.mkdir(parents=True, exist_ok=True)
        antes = len(done)
        done = {k for k in done if caminho_pred(k[0], k[1], k[2], k[3], k[4]).exists()}
        log(f"  com --pred-dir: {len(done)} de {antes} celulas tem metrica E "
            f"predicao; as outras {antes - len(done)} serao refeitas")

    for ds in args.datasets:
        c = load_blocks(ds, list(HAND8))
        stems = list(c.stems)
        labels_raw = np.asarray(c.labels, dtype=str)
        n_train_rows = len(stems)
        # coorte de TESTE separada no registry (nct_crc -> nct_crc_val, §5.18e):
        # o ramo OFFICIAL sozinho aborta ali com "zero linhas de teste"
        # As duas coortes do nct_crc tem CACHE SEPARADO (`nct_crc_wndcharm_tiled`
        # e `nct_crc_val_wndcharm_tiled`), e o mesmo vale para o haralick_hog.
        # Passar a lista COMBINADA ao carregador de um corpus so da cache miss,
        # repoola 19,6 GB e DEPOIS falha -- o worker terminaria com hand-8
        # sozinho e sem erro (§8). Carregar separado e empilhar e o que o
        # `evaluate_handcrafted_banks_lda.py:185-191` faz.
        held_stems: list[str] = []
        if ds in TEST_COHORT:
            held = load_blocks(TEST_COHORT[ds], list(HAND8), allow_external=True)
            held_stems = [str(s) for s in held.stems]
            labels_raw = np.concatenate([labels_raw,
                                         np.asarray(held.labels, dtype=str)])
            blocos = {b: np.vstack([c.blocks[b], held.blocks[b]]) for b in c.columns}
        else:
            blocos = {b: c.blocks[b] for b in c.columns}
        todos = stems + held_stems
        # `todos` e a lista alinhada com `y` LINHA A LINHA nas duas coortes, e e
        # a unica que serve para gravar predicao: `stems` sozinho tem so a
        # coorte de treino, entao em TEST_COHORT os indices de teste (100.000 a
        # 107.179 no nct_crc) estouram. Ficou DEFINIDA E NUNCA USADA ate 12/out,
        # e por isso o --pred-dir funcionou em 6 corpora e crashou no setimo.
        assert len(todos) == len(labels_raw), (
            f"{ds}: {len(todos)} stems contra {len(labels_raw)} rotulos")

        bancos: dict[str, np.ndarray] = {}
        if "hand-8" in args.bancos:
            bancos["hand-8"] = np.hstack([blocos[b] for b in c.columns])

        def dois(carrega):
            """Aplica o carregador por COORTE e empilha; None se faltar um lado."""
            a = carrega(ds, stems)
            if not held_stems:
                return a
            b = carrega(TEST_COHORT[ds], held_stems)
            return np.vstack([a, b]) if a is not None and b is not None else None

        if {"Haralick", "HOG"} & set(args.bancos):
            hh = dois(lambda d, st: by_stem(ROOT / f"outputs/{d}_haralick_hog", st))
            if hh is None:
                log(f"!! {ds}: sem Haralick/HOG -- pulando os dois bancos")
            else:
                if "Haralick" in args.bancos:
                    bancos["Haralick"] = np.ascontiguousarray(hh[:, HARALICK_COLS])
                if "HOG" in args.bancos:
                    bancos["HOG"] = np.ascontiguousarray(hh[:, HOG_COLS])
        # CLIP CONGELADO: o backbone moderno que faltava no bloco de congelados.
        # Entra aqui, e nao num script proprio, porque este produtor ja resolve
        # split oficial, TEST_COHORT, sementes e as tres metricas -- e o §2
        # proibe criar caminho novo. O cache vem do
        # `extract_clip_cache_model.py`, que grava NA ORDEM DA COORTE, entao o
        # alinhamento e posicional e a guarda e o tamanho.
        #
        # §7.19 refinado pelo §5.20: o eixo da sonda e *se a representacao foi
        # otimizada para separabilidade linear*, e o CLIP foi, por alinhamento
        # contrastivo. Logo MLP, e a sensibilidade de cabeca do §7.8 declarada
        # junto -- no §5.18f ela INVERTE o sinal da nossa complementaridade
        # (+0,91 estreita contra -0,94 larga).
        if "CLIP" in args.bancos:
            def clip(d: str, st: list[str]):
                """ALINHA POR STEM. O cache e um .npy sem stems, gravado na ordem
                do `results/<ds>_student_captions.csv` e NAO na da coorte do
                registry -- o consumidor publicado (`evaluate_clip_text_arms.py`)
                le os rotulos do mesmo CSV e por isso e auto-consistente, mas
                aqui as duas ordens divergem. Casar por POSICAO deu BAcc 0,0017
                contra os 54,39 do §5.18f: ruido puro, com o tamanho batendo
                exatamente. E o §5.18b outra vez, e a guarda de tamanho nao pega.
                """
                q = ROOT / f"outputs/{d}_clip_vitb32.npy"
                leg = ROOT / f"results/{d}_student_captions.csv"
                if not (q.exists() and leg.exists()):
                    return None
                ordem = pd.read_csv(leg, dtype={"stem": str})["stem"].tolist()
                a = np.load(q)
                if a.shape[0] != len(ordem):
                    log(f"!! {d}: cache CLIP {a.shape[0]} x CSV de legendas "
                        f"{len(ordem)} -- fontes diferentes, pulando")
                    return None
                pos = {t: i for i, t in enumerate(ordem)}
                falta = [t for t in st if t not in pos]
                if falta and not args.descartar_faltantes:
                    log(f"!! {d}: {len(falta)} de {len(st)} stems da coorte fora "
                        f"do cache CLIP -- pulando (nao avaliar em subconjunto, "
                        f"que deixaria a celula incomparavel). "
                        f"--descartar-faltantes corta e declara.")
                    return None
                # NaN nas que faltam; quem corta e o bloco de `parts`, nao o
                # imputador -- preencher com mediana fabricaria exemplo de
                # treino com rotulo real, que e pior que o buraco.
                idx = np.array([pos.get(t, -1) for t in st])
                out = np.full((len(st), a.shape[1]), np.nan, dtype=a.dtype)
                ok = idx >= 0
                out[ok] = a[idx[ok]]
                return out
            cl = dois(clip)
            if cl is None:
                log(f"!! {ds}: sem CLIP alinhado -- pulando o banco")
            else:
                bancos["CLIP ViT-B/32 congelado"] = cl
        if "WND-CHARM" in args.bancos:
            wnd = dois(pooled_wndcharm)
            if wnd is None:
                log(f"!! {ds}: sem WND-CHARM -- pulando o banco")
            else:
                bancos["WND-CHARM"] = wnd
        quer_fv = {"Fisher vector", "Fisher vector (celula global)"} & set(args.bancos)
        if quer_fv:
            sfx = args.fv_sufixo
            fv = dois(lambda d, st: by_stem(
                ROOT / f"outputs/{d}_fisher_vector{sfx}", st))
            if fv is None:
                log(f"!! {ds}: sem Fisher vector -- pulando o(s) banco(s)")
            else:
                rot = f"Fisher vector{' ' + sfx.lstrip('_') if sfx else ''}"
                if "Fisher vector" in args.bancos:
                    bancos[rot] = fv
                # A PIRAMIDE E NINHADA: cada celula e normalizada por
                # potencia+L2 e o vetor inteiro leva um L2 global, entao com 4
                # celulas cada bloco tem norma exatamente 0,5 (medido) e as
                # primeiras 2*K*D dimensoes SAO o FV sem piramide -- mesmo
                # codebook, mesmos descritores, mesmo K, mesma PCA. Fatiar da o
                # par pareado "com piramide x sem" em todo corpus, sem extracao.
                # O fator 1/2 nao e desfeito de proposito: `scaled_splits`
                # padroniza POR COLUNA, e uma constante positiva comum a todas
                # as colunas nao muda nada depois disso.
                if "Fisher vector (celula global)" in args.bancos:
                    meta_fv = json.loads(
                        (ROOT / f"outputs/{ds}_fisher_vector{sfx}/meta.json").read_text())
                    bloco = 2 * meta_fv["gmm_k"] * meta_fv["pca_dim"]
                    if meta_fv.get("pyramid_cells", 1) == 1:
                        log(f"!! {ds}: o cache do FV nao tem piramide -- a celula "
                            f"global seria o proprio banco; pulando")
                    else:
                        bancos[f"{rot} (celula global)"] = fv[:, :bloco]

        # CORTE das linhas que o banco nao cobre (§5.18e faz o mesmo no
        # places_lt). So com UM banco: com varios, cortar por um deles
        # dessincronizaria os outros sem nada na saida denunciar.
        viva_linha = np.ones(len(labels_raw), bool)
        if args.descartar_faltantes and bancos:
            # INTERSECAO da cobertura de TODOS os bancos pedidos: cortar por um
            # so dessincronizaria os outros. Com a mesma mascara nos dois lados
            # o bloco fica internamente comparavel, que e o que importa aqui --
            # ele deixa de ser comparavel com a MESMA celula noutras tabelas, e
            # isso vai declarado.
            viva_linha = np.logical_and.reduce(
                [np.isfinite(v).all(axis=1) for v in bancos.values()])
            if not viva_linha.all():
                fora = ~viva_linha
                onde = ""
                if ds in OFFICIAL:
                    m2 = pd.read_csv(ROOT / f"data/{ds}_meta.csv", dtype={"stem": str})
                    w2 = m2.set_index("stem")["split"].reindex(todos).to_numpy(str)
                    onde = " | por split: " + ", ".join(
                        f"{q} {int((fora & (w2 == q)).sum())}"
                        for q in ("train", "val", "test"))
                log(f"  --descartar-faltantes: {int(fora.sum()):,} de "
                    f"{len(fora):,} linhas sem banco foram REMOVIDAS{onde}")
                bancos = {k: v[viva_linha] for k, v in bancos.items()}
                labels_raw = labels_raw[viva_linha]
                todos = [t for t, q in zip(todos, viva_linha) if q]
                stems = [t for t in todos if t not in set(held_stems)]
                n_train_rows = len(stems)

            # GUARDA DE CLASSE RARA, refeita DEPOIS do corte -- contar no
            # metadado nao serve (§5.18e: e exatamente o caso do
            # `d_desert_road`, que fica com 1 de 176 linhas de treino). A classe
            # sai de treino E de teste: mante-la so no teste daria recall zero
            # numa classe que o modelo nao pode acertar, deprimindo a BAcc por
            # artefato de contagem em vez de por desempenho.
            if ds in OFFICIAL:
                m3 = pd.read_csv(ROOT / f"data/{ds}_meta.csv", dtype={"stem": str})
                w3 = m3.set_index("stem")["split"].reindex(todos).to_numpy(str)
                cont = pd.Series(labels_raw[w3 == "train"]).value_counts()
                rara = set(cont[cont < 2].index) | (set(labels_raw) - set(cont.index))
                if rara:
                    fica = ~pd.Series(labels_raw).isin(rara).to_numpy()
                    log(f"  guarda de classe rara: {len(rara)} classe(s) com <2 "
                        f"linhas de treino removidas de treino E teste "
                        f"({int((~fica).sum()):,} linhas) -> "
                        f"{len(set(labels_raw[fica]))} classes")
                    bancos = {k: v[fica] for k, v in bancos.items()}
                    labels_raw = labels_raw[fica]
                    todos = [t for t, q in zip(todos, fica) if q]
                    stems = [t for t in todos if t not in set(held_stems)]
                    n_train_rows = len(stems)

        y = LabelEncoder().fit_transform(labels_raw)
        C = len(np.unique(y))

        if ds in TEST_COHORT:
            parts = [(1, np.arange(n_train_rows), np.arange(n_train_rows, len(y)))]
            tipo = "coorte externa"
        elif ds in OFFICIAL:
            meta = pd.read_csv(ROOT / f"data/{ds}_meta.csv", dtype={"stem": str})
            where = meta.set_index("stem")["split"].reindex(stems).to_numpy(str)
            parts = [(1, np.flatnonzero(where == "train"),
                      np.flatnonzero(where == "test"))]
            tipo = "split OFICIAL"
        else:
            parts = [(f, tr, te) for f, (tr, te) in enumerate(
                StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                .split(np.zeros(len(y)), y), start=1)]
            tipo = f"CV {N_FOLDS} folds"
        log(f"{ds}: [{tipo}] treino={len(parts[0][1]):,} teste={len(parts[0][2]):,} "
            f"classes={C} | " + ", ".join(f"{k} {v.shape[1]}d"
                                          for k, v in bancos.items()))

        faltam = [f for f in args.fusoes if f not in bancos]
        if faltam:
            raise SystemExit(f"--fusoes: {faltam} nao esta entre os bancos "
                             f"carregados ({list(bancos)})")
        condicoes = [(n, [n]) for n in bancos] + \
                    [(f"hand-8 + {f}", ["hand-8", f]) for f in args.fusoes]
        if args.folds:
            parts = [q for q in parts if q[0] in set(args.folds)]
            log(f"  filtro de fold: {sorted(args.folds)} -> {len(parts)} particao(oes)")
        for fold, tr, te in parts:
            # `limpar` depende so das linhas de TREINO (o fold), nao da semente
            limpos: dict[str, np.ndarray] = {}
            for seed in args.seeds:
                # o `evaluate_mlp` usa SEED+fold como semente de treino E de
                # split de validacao, entao a chave tem de encodar os dois.
                # seed=42 da fold_key = fold, que REPRODUZ a corrida de 07/out
                # -- e braco de controle, nao conveniencia (§5.6).
                fold_key = (seed - SEED) * 10 + fold
                for nome, partes in condicoes:
                    for arq in args.arquiteturas:
                        if (ds, nome, arq, seed, fold) in done:
                            continue
                        if nome not in limpos:
                            bruto = (bancos[partes[0]] if len(partes) == 1
                                     else np.hstack([bancos[q] for q in partes]))
                            limpos[nome] = limpar(bruto, tr)
                            if len(partes) > 1:
                                del bruto
                        Z = limpos[nome]
                        t0 = time.perf_counter()
                        m = evaluate_mlp([Z], y, tr, te, fold_key, EPOCHS,
                                         PATIENCE, label_smoothing=0.05,
                                         model_factory=ARQ[arq],
                                         return_probs=args.pred_dir is not None)
                        # remover a linha superada ANTES de acrescentar: com
                        # --pred-dir uma celula pode ser refeita, e so acrescentar
                        # duplicaria (o script de LDA ja fazia isto)
                        rows = [r for r in rows
                                if not (r["dataset"] == ds and r["banco"] == nome
                                        and r["arquitetura"] == arq
                                        and int(r.get("seed", SEED)) == seed
                                        and int(r["fold"]) == fold)]
                        rows.append({
                            "dataset": ds, "banco": nome, "arquitetura": arq,
                            "seed": seed, "fold": fold,
                            "dims": int(Z.shape[1]),
                            "dims_vivas": int(Z.shape[1]),
                            "bacc": 100 * m["bacc"],
                            "accuracy": 100 * m["accuracy"],
                            "macro_f1": 100 * m["macro_f1"],
                            "validation_bacc": 100 * m["validation_bacc"],
                            "best_epoch": m.get("best_epoch", np.nan),
                            "seconds": time.perf_counter() - t0})
                        pd.DataFrame(rows).to_csv(args.output, index=False,
                                                  float_format="%.4f")
                        if args.pred_dir is not None:
                            args.pred_dir.mkdir(parents=True, exist_ok=True)
                            # os STEMS vao junto: sem eles o pareamento entre
                            # bancos seria por posicao, que e o bug do §4.1f
                            np.savez_compressed(
                                str(caminho_pred(ds, nome, arq, seed, fold))[:-4],
                                pred=np.asarray(m["test_probs"]).argmax(1),
                                # os POSTERIORS vao junto: a fusao de decisao
                                # (media de log-probs, §4.1f) precisa deles, e
                                # gravar so o argmax obriga a re-treinar tudo
                                probs=np.asarray(m["test_probs"], dtype=np.float32),
                                truth=y[te],
                                stems=np.asarray(todos, dtype=object)[te])
                        log(f"  s{seed} f{fold} {nome:<10} {arq:<10} "
                            f"{Z.shape[1]:>5}d  BAcc {rows[-1]['bacc']:5.2f}  "
                            f"Acc {rows[-1]['accuracy']:5.2f}  "
                            f"(ep {rows[-1]['best_epoch']}, "
                            f"{rows[-1]['seconds']:.0f}s)")

    t = pd.DataFrame(rows)
    for arq in args.arquiteturas:
        g = t[t.arquitetura.eq(arq)]
        if not len(g):
            continue
        log(f"\n=== {arq} ===")
        for ds in g.dataset.unique():
            h = g[g.dataset.eq(ds)]
            a = h[h.banco.eq("hand-8")].set_index("fold")
            b = h[h.banco.eq("WND-CHARM")].set_index("fold")
            if not len(a) or not len(b):
                continue
            saida = [f"{ds:<12}"]
            for met in ("bacc", "accuracy", "macro_f1"):
                x, yy = a[met].align(b[met], join="inner")
                d = x - yy
                saida.append(f"{met[:4]} {x.mean():5.2f} vs {yy.mean():5.2f} "
                             f"= {d.mean():+5.2f} ({(d>0).sum()}/{len(d)})")
            log("  ".join(saida))
    log(f"\n{args.output}")


if __name__ == "__main__":
    main()
