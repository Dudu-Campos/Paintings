"""Fecha a cadeia de producao: hand-8 + grafo em `logit-add` + canal de texto.

O §9.7 registra que os dois nunca foram medidos JUNTOS -- o marginal de texto do
§5.18 (+2,31pp) tem como base o grafo **concatenado**, e o `logit-add` do §5.6
foi medido sem texto. Encadear os dois numeros seria inventar um resultado.

Roda o 2x2 completo (grafo aditivo x texto), nao so a celula que falta, porque a
pergunta interessante e a INTERACAO: o §9.7 mede que o texto rende 2,7x mais com
o grafo concatenado presente (+2,31 contra +0,86 sem), e nao ha mecanismo
proposto. Com o grafo no canal aditivo -- que nao gasta dimensao nenhuma -- a
interacao pode desaparecer ou nao, e isso e informativo nos dois sentidos.

NADA e reimplementado aqui: `train_arm` e `build_oof` vem do script do logit-add
e `nested_text_block` do script de texto. O §5.6 registra o custo de reescrever
um braco de controle (0,47pp de divergencia por `label_smoothing` e semente
omitidos), e a regra que ficou foi importar em vez de redefinir.

O aninhamento do texto e obrigatorio: `build_text_scores` produz OOF sobre a
coorte INTEIRA, entao para o fold externo k os escores das linhas de treino vem
de modelos que viram os rotulos do fold k. E o mecanismo do SNF (§6), diluido.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src/pipeline"))
sys.path.insert(0, str(ROOT / "experiments/PRISM"))

from build_text_scores import make_stop_words  # noqa: E402
from evaluate_logit_add_multitask import (HAND8, N_FOLDS, SEED,  # noqa: E402
                                          build_oof, train_arm)
from evaluate_text_marginal_multitask import (fit_text,  # noqa: E402
                                              load_captions, nested_text_block)
from registry import LABELS, load_blocks, log  # noqa: E402

torch.set_num_threads(16)
CAPTIONS = {"estudante": "results/{dataset}_student_captions.jsonl",
            "catr": "outputs/affect_captions/catr_coco_{dataset}.jsonl"}


OFFICIAL = {"food101", "nct_crc", "minc2500", "places_lt"}
TEST_COHORT = {"nct_crc": "nct_crc_val"}


def captioned_stems(path):
    """Stems que TEM legenda, sem carregar o texto -- so para decidir o corte."""
    import json as _json
    with path.open(encoding="utf-8") as h:
        return {str(_json.loads(l)["stem"]) for l in h}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="multitask")
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--captions", default="estudante", choices=sorted(CAPTIONS))
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="semente de TREINO; a particao NAO muda")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    args.meta = args.meta or ROOT / f"data/{args.dataset}_meta.csv"
    # A PARTICAO fica fixa -- ela e o split publicado (§7.5) ou os 5 folds da
    # convencao do §2. O que `--seed` move e a semente de TREINO: divisao de
    # validacao e inicializacao de pesos. O §5.6 mediu que este experimento e
    # limitado por ruido de treino (~±0,5pp por unidade) e nao por numero de
    # folds -- "o conserto e semente, nao fold" --, e nos quatro corpora de split
    # oficial a semente e a UNICA fonte de barra de erro possivel, porque a
    # particao e uma so.
    if args.output is None:
        sufixo = "" if args.seed == SEED else f"_seed{args.seed}"
        args.output = ROOT / f"results/{args.dataset}_logit_add_text{sufixo}.csv"
    caption_path = ROOT / CAPTIONS[args.captions].format(dataset=args.dataset)
    if not caption_path.exists():
        raise SystemExit(f"legendas ausentes: {caption_path}")

    cohort = load_blocks(args.dataset, HAND8, allow_external=True)
    # Corpora cujo TESTE e uma coorte SEPARADA no registry, nao uma fatia da
    # coluna `split` (§4.1d). No nct_crc a coorte `nct_crc` e so treino (100.000)
    # e o teste externo sao os 7.180 tiles de PACIENTES DIFERENTES em
    # `nct_crc_val` -- e essa separacao vale 11pp (99,08 interno contra 87,64
    # externo, §7.14). Fatiar a coluna `split` aqui daria zero linhas de teste.
    held = None
    if args.dataset in TEST_COHORT:
        held = load_blocks(TEST_COHORT[args.dataset], HAND8, allow_external=True)
        log(f"coorte de teste SEPARADA: {TEST_COHORT[args.dataset]} "
            f"({len(held.stems):,} imagens)")
    # Os stems vem do COHORT, nunca reconstruidos do metadado. A primeira versao
    # filtrava `data/<ds>_meta.csv` por estilo e obtinha os mesmos stems em
    # ORDEM DIFERENTE: `load_captions` casa por stem e nao reclama, entao o bloco
    # de texto entra embaralhado em relacao as imagens -- ruido puro. Custou
    # -2,92pp no fold 1 (36,04 contra 38,96) onde o §5.18 mede +2,31.
    stems = np.asarray(cohort.stems, dtype=str)
    labels_str = np.asarray(cohort.labels, dtype=str)
    labels = LabelEncoder().fit_transform(labels_str)
    n_classes = len(np.unique(labels))
    classes = np.unique(labels_str)
    index_of = {c: i for i, c in enumerate(classes)}
    blocks = [cohort.blocks[name] for name in HAND8]

    # Corpora de split oficial: usar SO `train` e `test`. O `val` fica de fora
    # do treino porque o protocolo publicado nao o usa como dado de ajuste --
    # o §7.5 treina o Places-LT nas 62.500 e testa nas 36.500, e o §7.15 usa o
    # val do MINC apenas para escolher a epoca. A validacao interna sai de 10%
    # do treino, como no resto do projeto.
    if args.dataset in OFFICIAL and args.dataset not in TEST_COHORT:
        meta0 = pd.read_csv(ROOT / LABELS[args.dataset][0], dtype={"stem": str})
        w0 = meta0.set_index("stem")["split"].reindex(stems).to_numpy(str)
        keep = np.isin(w0, ("train", "test"))
        if not keep.all():
            log(f"  fora do protocolo: {int((~keep).sum()):,} linhas de `val` "
                f"descartadas (§7.5/§7.15)")
            stems, labels = stems[keep], labels[keep]
            labels_str = labels_str[keep]
            blocks = [b[keep] for b in blocks]

    # Linhas da coorte sem imagem local (logo sem legenda) saem, com a contagem
    # DECLARADA por split. No Places-LT sao 528 de `train` que nao constam da
    # lista `Places_LT_train.txt` -- estao no metadado, nao no benchmark.
    have = captioned_stems(caption_path)
    ok = np.isin(stems, list(have))
    if not ok.all():
        falta = int((~ok).sum())
        if falta > 0.02 * len(stems):
            raise SystemExit(f"{falta:,} de {len(stems):,} sem legenda (>2%): "
                             f"investigar antes de descartar")
        log(f"  sem legenda: {falta:,} linhas descartadas ({100*falta/len(stems):.2f}%)")
        stems, labels, labels_str = stems[ok], labels[ok], labels_str[ok]
        blocks = [b[ok] for b in blocks]

    # Classe com menos de 2 linhas de TREINO quebra o split estratificado de 10%
    # que o `evaluate_mlp` faz internamente. Isso so acontece depois de um corte,
    # entao a contagem tem de ser refeita AQUI e nao no metadado: no places_lt as
    # 528 linhas sem imagem estao concentradas em 5 classes (§5.18d), e
    # `d_desert_road` fica com 1 de 176. Dropar a classe inteira -- de treino E de
    # teste -- e a unica saida que mantem o teste avaliavel; deixar so no teste
    # daria recall 0 numa classe que o modelo nunca pode acertar, o que depreciaria
    # a BAcc por artefato de contagem.
    if args.dataset in OFFICIAL and args.dataset not in TEST_COHORT:
        w1 = meta0.set_index("stem")["split"].reindex(stems).to_numpy(str)
        vivas, mortas = [], []
        for c in np.unique(labels_str):
            (vivas if int(((labels_str == c) & (w1 == "train")).sum()) >= 2
             else mortas).append(c)
        if mortas:
            log(f"  classes com <2 linhas de treino apos os cortes: {len(mortas)} "
                f"descartadas -> {sorted(mortas)}")
            keep2 = np.isin(labels_str, vivas)
            stems, labels_str = stems[keep2], labels_str[keep2]
            blocks = [b[keep2] for b in blocks]
            classes = np.unique(labels_str)
            index_of = {c: i for i, c in enumerate(classes)}
            labels = np.array([index_of[c] for c in labels_str])
            n_classes = len(classes)
            log(f"  coorte final: {len(stems):,} imagens, {n_classes} classes")

    texts = load_captions(caption_path, stems)
    n_train_rows = len(stems)
    if held is not None:
        held_stems = np.asarray(held.stems, dtype=str)
        held_labels = np.asarray(held.labels, dtype=str)
        desconhecidas = set(held_labels) - set(classes)
        if desconhecidas:
            raise SystemExit(f"classes so no teste: {sorted(desconhecidas)}")
        blocks = [np.vstack([b, held.blocks[n]]) for b, n in zip(blocks, HAND8)]
        labels = np.concatenate([labels, [index_of[c] for c in held_labels]])
        # `labels_str` tambem: a guarda de alinhamento indexa por ele, e
        # esquece-lo da IndexError so DEPOIS de carregar as duas coortes.
        labels_str = np.concatenate([labels_str, held_labels])
        held_caps = ROOT / f"results/{TEST_COHORT[args.dataset]}_student_captions.jsonl"
        if not held_caps.exists():
            raise SystemExit(f"legendas da coorte de teste ausentes: {held_caps}")
        texts = np.concatenate([texts, load_captions(held_caps, held_stems)])
        stems = np.concatenate([stems, held_stems])
    space = np.hstack(blocks)
    # As stop words de NOME DE ESTILO sao a defesa do §5.18 contra o captioner
    # nomear o rotulo (o estudante diz "abstract" em 5,93% das legendas). Passar
    # lista vazia aqui removeria a defesa em silencio -- e `set()` nem e aceito
    # pelo TfidfVectorizer, que so admite list, "english" ou None.
    stop_words = list(make_stop_words(classes))
    log(f"{args.dataset}: {len(labels):,} imagens, {n_classes} classes, "
        f"hand-8 {space.shape[1]}d, legendas '{args.captions}', "
        f"{len(stop_words)} stop words")

    # GUARDA DE ALINHAMENTO. O pareamento legenda<->imagem e implicito (ordem das
    # linhas), entao trocar a ordem nao levanta erro nenhum -- foi exatamente o
    # bug da primeira versao. O teste tem de ser sobre o que importa: as legendas
    # ainda predizem o rotulo? Se estiverem embaralhadas isto cai no acaso.
    rng = np.random.default_rng(SEED)
    probe = rng.permutation(len(labels))[:12000]
    cut = int(0.8 * len(probe))

    def probe_accuracy(sample_texts):
        vec, model = fit_text(sample_texts[probe[:cut]], labels_str[probe[:cut]],
                              stop_words)
        return float((model.predict(vec.transform(sample_texts[probe[cut:]]))
                      == labels_str[probe[cut:]]).mean())

    # O NULO e medido, nao chutado: o mesmo probe sobre as legendas embaralhadas
    # e exatamente o modo de falha que se quer pegar. Um limiar contra o acaso
    # teorico nao serve -- a primeira versao exigia 3x o acaso e abortou os dois
    # corpora com o alinhamento CORRETO (12,67% contra 5,03% em 125 classes e
    # 32,75% contra 11,22% em 17 sao sinal, nao ruido).
    real = probe_accuracy(texts)
    null = probe_accuracy(texts[rng.permutation(len(texts))])
    ratio = real / max(null, 1e-9)
    log(f"  guarda de alinhamento: texto sozinho {100*real:.2f}% | "
        f"embaralhado {100*null:.2f}% | razao {ratio:.1f}x")
    # A guarda distingue DESALINHAMENTO de canal fraco, e a fronteira teve de ser
    # medida: no rijksdraw a razao e 1,9x com o alinhamento CORRETO (o rotulo e
    # PERIODO, e conteudo de cena nao diz o seculo -- previsao do §5.18), contra
    # 13,2x no multitask e 4,8x no oilpainting. Um limiar de 2x abortava um
    # resultado valido. Embaralhado de verdade fica em ~1,0x, entao 1,3x separa
    # os dois casos com folga; entre 1,3 e 2,0 vale um aviso, nao um aborto.
    if ratio < 1.3:
        raise SystemExit("ABORTADO: legendas provavelmente desalinhadas com a coorte")
    if ratio < 2.0:
        log(f"  [aviso] canal de texto fraco neste corpus (razao {ratio:.1f}x); "
            f"alinhamento ok, mas esperar marginal pequeno")

    rows = []
    if args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        log(f"retomando {len(rows)} linhas")
    done = {(r["condition"], r["fold"]) for r in rows}

    # Corpora com particao publicada NAO podem rodar 5-fold CV sobre a coorte
    # inteira -- e o bug que o §8 registra no evaluate_preprocess_effect (a
    # constante OFFICIAL existia e nunca era lida). No Places-LT a CV derruba o
    # desbalanceamento de 996x para 41x e apaga o regime que motiva o corpus; no
    # NCT-CRC apaga a separacao por paciente do teste externo, que vale 11pp.
    if args.dataset in TEST_COHORT:
        train_idx = np.arange(n_train_rows)
        test_idx = np.arange(n_train_rows, len(labels))
        splits = [(train_idx, test_idx)]
        log(f"coorte EXTERNA: {len(train_idx):,} treino / {len(test_idx):,} teste "
            f"-- 1 particao (conferir contra o numero publicado)")
    elif args.dataset in OFFICIAL:
        meta_split = pd.read_csv(ROOT / LABELS[args.dataset][0], dtype={"stem": str})
        where = (meta_split.set_index("stem")["split"]
                 .reindex(stems).to_numpy(str))   # `stems` vem do COHORT (l.83)
        train_idx = np.flatnonzero(where == "train")   # `val` ja foi descartado
        test_idx = np.flatnonzero(where == "test")     # acima; o fit separa 10%.
        if len(test_idx) == 0:
            raise SystemExit(f"{args.dataset}: coluna `split` sem linhas de teste")
        splits = [(train_idx, test_idx)]
        # Imprimir os tamanhos e o antidoto do §8: o bug antigo sobreviveu
        # porque NADA na saida denunciava.
        log(f"split OFICIAL: {len(train_idx):,} treino / {len(test_idx):,} teste "
            f"-- 1 particao (conferir contra o numero publicado)")
    else:
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                      .split(np.zeros(len(labels), np.uint8), labels))
        log(f"CV: {N_FOLDS} folds sobre {len(labels):,} imagens")
    for fold, (outer, test) in enumerate(splits, start=1):
        fit, validation = train_test_split(
            outer, test_size=0.10, stratify=labels[outer], random_state=args.seed)

        lift = None
        if any((c, fold) not in done for c in ("+ logit-add", "+ logit-add + texto")):
            started = time.perf_counter()
            scaler = StandardScaler().fit(space[outer])
            scaled = scaler.transform(space).astype(np.float32)
            oof, held = build_oof(scaled[outer], labels[outer], scaled[test],
                                  n_classes, args.k)
            lift = np.zeros((len(labels), n_classes), dtype=np.float32)
            lift[outer], lift[test] = oof, held
            log(f"  fold {fold}: log-lift em {time.perf_counter() - started:.0f}s")

        text = None
        if any((c, fold) not in done for c in ("hand-8 + texto", "+ logit-add + texto")):
            started = time.perf_counter()
            text = nested_text_block(texts, labels_str, classes, index_of,
                                     stop_words, outer, test, fold)
            log(f"  fold {fold}: OOF aninhado do texto em "
                f"{time.perf_counter() - started:.0f}s")

        arms = [("hand-8", blocks, None),
                ("hand-8 + texto", blocks + ([text] if text is not None else []), None),
                ("+ logit-add", blocks, lift),
                ("+ logit-add + texto",
                 blocks + ([text] if text is not None else []), lift)]
        for name, parts, channel in arms:
            if (name, fold) in done:
                continue
            started = time.perf_counter()
            metrics, lam = train_arm(
                parts, channel, labels, fit, validation, test, n_classes,
                None if channel is None else np.zeros(n_classes, dtype=int),
                args.seed, args.epochs, args.patience)
            rows.append({"condition": name, "fold": fold, "seed": args.seed, **metrics,
                         "dimensions": sum(b.shape[1] for b in parts),
                         "lambda": ";".join(f"{v:.4f}" for v in lam),
                         "elapsed_seconds": time.perf_counter() - started})
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
            log(f"  {name:<22} fold {fold} | {rows[-1]['dimensions']:5d}d | "
                f"BAcc {metrics['bacc']:5.2f} | {rows[-1]['elapsed_seconds']:.0f}s")

    table = pd.DataFrame(rows)
    base = table[table.condition == "hand-8"].set_index("fold")["bacc"]
    log("\n=== marginal sobre hand-8 (pareado por fold) ===")
    log(f"  {'hand-8':<22} {base.mean():5.2f} ± {base.std():.2f}")
    for name in ("hand-8 + texto", "+ logit-add", "+ logit-add + texto"):
        arm = table[table.condition == name].set_index("fold")["bacc"]
        common = base.index.intersection(arm.index)
        if len(common) < 2:
            continue
        delta = arm.loc[common] - base.loc[common]
        _, p = stats.ttest_rel(arm.loc[common], base.loc[common])
        log(f"  {name:<22} {arm.loc[common].mean():5.2f} ± {arm.loc[common].std():.2f}"
            f"   Δ {delta.mean():+.2f} ± {delta.std():.2f}pp  p={p:.4f}  "
            f"{int((delta > 0).sum())}/{len(delta)}")
    log(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
