"""O ganho da fusao ponderada sobrevive quando o peso NAO ve o teste?

O §4.1d registra que o peso otimo da fusao foi varrido **no teste** e por isso
nao e reportavel -- no MINC ele recupera +1,58pp mas oscila de 0,05 a 0,35 entre
sementes, o que ja indica ajuste a ruido. A rota legitima seria escolher o peso
no `val` oficial, mas **nao ha posteriors de validacao salvos** (nem do nosso
lado nem do backbone), e refaze-los exigiria re-treinar os dois.

Este script responde a mesma pergunta sem dado novo, por **cross-fitting**: o
teste e particionado em K dobras estratificadas; para cada dobra o peso e
escolhido nas outras K-1 e aplicado so na dobra retida. As predicoes retidas sao
concatenadas e a BAcc e calculada uma vez sobre o teste inteiro. Nenhuma imagem
e avaliada com um peso que a viu.

Isso NAO e equivalente a escolher no `val` oficial -- continua usando a
distribuicao do teste --, mas remove a circularidade que invalida o numero atual
e estima o que interessa: **a seguranca do procedimento**, nao a de um peso.

Tres condicoes por unidade:
    w = 0,5          a fusao publicada, sem parametro
    w cross-fit      o peso escolhido fora da dobra avaliada
    w otimo no teste o teto NAO reportavel, so para dimensionar a folga

ESTENDIDO em 19/set para os TRES corpora de arte. A versao anterior cobria so
os quatro de split oficial, e a ressalva registrada dizia que "os tres de arte
nao tem posteriors nesse formato" -- isso estava errado. Eles tem, em
`outputs/{corpus}_our_oof_cv.npz` (o cache do `evaluate_decision_fusion_cv.py`,
com as mesmas chaves `stems`/`y_true`/`seed_*`), e o lado do backbone se monta
juntando os 5 folds da grade, que e o que `backbone_oof` ja fazia la. Nenhum
treino novo: o cross-fit e aritmetica sobre posteriors em disco.

Diferenca de protocolo que tem de ficar declarada: nos quatro oficiais o
cross-fit particiona um conjunto de TESTE separado; nos tres de arte ele
particiona a coorte inteira em OOF, cujas linhas vem de CINCO modelos (um por
fold da CV). A dobra de selecao de peso mistura linhas de modelos diferentes.
Isso nao quebra a garantia -- nenhuma imagem e avaliada com um peso que a viu --
mas o estimando e "seguranca do procedimento sob OOF", nao sob split fixo.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

CORPORA = ["nct_crc", "minc2500", "places_lt", "food101"]
# corpus -> subdiretorio de outputs/final_finetune_grid. As features do
# multitask vivem sob `multitask` e os posteriors do backbone sob
# `multitask125`; `evaluate_decision_fusion_cv.py` carrega a mesma excecao.
CV_CORPORA = {"rijksdraw": "rijksdraw", "multitask": "multitask125",
              "oilpainting": "oilpainting"}
WEIGHTS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def log(m: str) -> None:
    print(m, flush=True)


def blend(P, B, w):
    return w * np.log(np.maximum(P, 1e-12)) + (1 - w) * np.log(np.maximum(B, 1e-12))


def backbone_oof(grid: Path, model: str, seed: str, stems: np.ndarray,
                 n_classes: int, y: np.ndarray) -> np.ndarray:
    """Junta os 5 folds da grade num posterior de coorte inteira.

    Mesma funcao de `evaluate_decision_fusion_cv.py`, com a checagem de rotulo
    que o §4.1f exige: alinhar POR STEM e conferir depois. Parear por posicao
    produziu numeros plausiveis e errados no MINC.
    """
    values = np.zeros((len(stems), n_classes), dtype=np.float32)
    seen = np.zeros(len(stems), dtype=bool)
    index = pd.Index(list(stems))
    for path in sorted((grid / model / f"seed_{seed}").glob("fold_*/fold_*_posteriors.npz")):
        blob = np.load(path, allow_pickle=True)
        order = index.get_indexer(blob["stems"].astype(str))
        if (order < 0).any():
            raise ValueError(f"{path}: {int((order < 0).sum())} stems fora da coorte")
        if not (blob["y_true"] == y[order]).all():
            raise ValueError(f"{path}: rotulos nao batem depois do alinhamento")
        values[order] = blob["probabilities"]
        seen[order] = True
    if not seen.all():
        raise ValueError(f"{model} seed {seed}: {int((~seen).sum())} linhas sem posterior")
    return values.astype(np.float64)


def yule_q(ok_a, ok_b):
    """Q de Yule sobre a 2x2 de acerto/erro, com a razao de chances que o gera.

    POR QUE ELE ENTROU (02/out): o item 4 desta serie mede o excesso do oraculo
    sobre o nulo de independencia em PONTOS PERCENTUAIS, e pp ainda herdam as
    taxas de acerto dos dois lados -- o mesmo defeito que obrigou o §4.1d a
    abandonar a porcentagem bruta de acertos exclusivos. Q = (ad-bc)/(ad+bc) =
    (OR-1)/(OR+1) e funcao da RAZAO DE CHANCES, invariante a escalar linha e
    coluna, logo comparavel entre corpora de acuracia muito diferente.

      Q = 0  independencia -- o oraculo vale exatamente o nulo
      Q > 0  os dois erram JUNTO, o oraculo fica ABAIXO do nulo
      Q < 0  complementares alem do acaso

    Haldane (+0,5 em toda celula) so quando alguma zera, para Q nao saturar em
    +-1 por contagem em vez de por associacao.
    """
    a = float((ok_a & ok_b).sum()); b = float((ok_a & ~ok_b).sum())
    c = float((~ok_a & ok_b).sum()); d = float((~ok_a & ~ok_b).sum())
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    num, den = a * d - b * c, a * d + b * c
    return (float(num / den) if den else float("nan"),
            float((a * d) / (b * c)) if b * c else float("inf"))


def yule_q_macro(y, ok_a, ok_b) -> float:
    """Q por classe, media macro -- o analogo do que a BAcc faz com a recall.
    O agrupado, com desbalanceamento, mede sobretudo as classes frequentes."""
    qs = [yule_q(ok_a[y == c], ok_b[y == c])[0]
          for c in np.unique(y) if (y == c).sum() >= 2]
    return float(np.nanmean(qs)) if qs else float("nan")


def unit_row(corpus: str, protocol: str, model: str, seed: str,
             P: np.ndarray, B: np.ndarray, y: np.ndarray, folds: int) -> dict:
    """As tres condicoes de peso para uma unidade (corpus, modelo, semente)."""
    bb = 100 * balanced_accuracy_score(y, B.argmax(1))
    half = 100 * balanced_accuracy_score(y, blend(P, B, 0.5).argmax(1))
    # ORACULO NA MESMA METRICA das outras condicoes. O `oracle` do
    # `*_decision_fusion*.csv` e decomposicao por imagem e portanto ACURACIA
    # SIMPLES (`oracle - backbone_accuracy` == `only_ours`); mistura-lo com
    # ganhos em BAcc num mesmo eixo produziu cross-fit ACIMA do teto no
    # multitask. Aqui o oraculo e o classificador que acerta sempre que um dos
    # dois lados acerta, avaliado em BAcc como todo o resto.
    ok_p, ok_b = P.argmax(1) == y, B.argmax(1) == y
    hit = ok_p | ok_b
    oracle_pred = np.where(hit, y, B.argmax(1))
    oracle = 100 * balanced_accuracy_score(y, oracle_pred)
    # nulo de independencia DENTRO da classe, exato em esperanca
    null = 100 * float(np.mean([1 - (1 - ok_p[y == c].mean()) * (1 - ok_b[y == c].mean())
                                for c in np.unique(y)]))
    q, orr = yule_q(ok_p, ok_b)
    q_macro = yule_q_macro(y, ok_p, ok_b)
    # teto nao reportavel: peso escolhido vendo o teste inteiro
    best = max(WEIGHTS, key=lambda w: balanced_accuracy_score(
        y, blend(P, B, w).argmax(1)))
    peek = 100 * balanced_accuracy_score(y, blend(P, B, best).argmax(1))

    pred = np.empty_like(y)
    picked = []
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=42).split(np.zeros(len(y)), y):
        w = max(WEIGHTS, key=lambda w: balanced_accuracy_score(
            y[tr], blend(P[tr], B[tr], w).argmax(1)))
        picked.append(w)
        pred[te] = blend(P[te], B[te], w).argmax(1)
    cross = 100 * balanced_accuracy_score(y, pred)
    return dict(corpus=corpus, protocol=protocol, model=model, seed=int(seed),
                backbone=bb, w_half=half, w_cross=cross, w_peek=peek,
                oracle_bacc=oracle, oracle_bacc_null=null,
                yule_q=q, odds_ratio=orr, yule_q_macro=q_macro,
                only_ours=100 * float((ok_p & ~ok_b).mean()),
                only_backbone=100 * float((~ok_p & ok_b).mean()),
                both=100 * float((ok_p & ok_b).mean()),
                neither=100 * float((~ok_p & ~ok_b).mean()),
                best_weight=best, weight_sd=float(np.std(picked)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--with-text", action="store_true",
                    help="usa os posteriors de `hand-8 + grafo + texto`")
    ap.add_argument("--corpora", nargs="+", default=None,
                    help="subconjunto a rodar; o default e os sete")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/fusion_weight_crossfit.csv")
    args = ap.parse_args()

    # Retomada: as unidades ja gravadas sao relidas e puladas. O §1 exige
    # saida incremental idempotente porque a maquina mata job longo por
    # SIGKILL externo, sem traceback.
    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if args.output.exists():
        previous = pd.read_csv(args.output)
        rows = previous.to_dict("records")
        done = {(r["corpus"], r["model"], int(r["seed"])) for r in rows}
        log(f"retomando: {len(rows)} unidades ja em {args.output.name}")

    wanted = set(args.corpora) if args.corpora else set(CORPORA) | set(CV_CORPORA)
    unknown = wanted - (set(CORPORA) | set(CV_CORPORA))
    if unknown:
        raise SystemExit("corpora desconhecidos: " + ", ".join(sorted(unknown)))

    for corpus in [c for c in CORPORA if c in wanted]:
        sufixo = "_text" if args.with_text else ""
        z = np.load(ROOT / f"outputs/{corpus}_our_posteriors{sufixo}.npz",
                    allow_pickle=True)
        stems, y = z["stems"].astype(str), z["y_true"]
        seeds = [k for k in z.files if k.startswith("seed_")]
        for model in ("resnet50", "vit_b_16"):
            for key in seeds:
                seed = key.split("_")[1]
                if (corpus, model, int(seed)) in done:
                    continue
                found = sorted(glob.glob(str(
                    ROOT / f"outputs/final_finetune_grid/{corpus}/{model}/seed_{seed}/*/*_posteriors.npz")))
                if not found:
                    continue
                zb = np.load(found[0], allow_pickle=True)
                # Alinhamento POR STEM, com conferencia de rotulo: o MINC guarda
                # as linhas em ordem diferente nos dois arquivos, e parear por
                # posicao produz numeros plausiveis e errados.
                order = pd.Index(list(zb["stems"].astype(str))).get_indexer(stems)
                if (order < 0).any():
                    raise ValueError(f"{corpus}/{model}/{seed}: stems ausentes")
                B = zb["probabilities"][order].astype(np.float64)
                if not (zb["y_true"][order] == y).all():
                    raise ValueError(f"{corpus}/{model}/{seed}: rótulos não batem")
                P = z[key].astype(np.float64)

                rows.append(unit_row(corpus, "official", model, seed, P, B, y, args.folds))
                pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
        log(f"  {corpus}: {len([r for r in rows if r['corpus'] == corpus])} unidades")

    # --- corpora de arte: OOF de validacao cruzada, mesmos posteriors ja em disco
    for corpus, grid_name in ((c, g) for c, g in CV_CORPORA.items() if c in wanted):
        ours_path = ROOT / (f"outputs/{corpus}_our_oof_cv"
                            + ("_text" if args.with_text else "") + ".npz")
        if not ours_path.exists():
            log(f"  {corpus}: sem {ours_path.name}, pulando")
            continue
        z = np.load(ours_path, allow_pickle=True)
        stems, y = z["stems"].astype(str), z["y_true"]
        n_classes = int(y.max()) + 1
        grid = ROOT / "outputs/final_finetune_grid" / grid_name
        seeds = [k for k in z.files if k.startswith("seed_")]
        for model in ("resnet50", "vit_b_16"):
            for key in seeds:
                seed = key.split("_")[1]
                if (corpus, model, int(seed)) in done:
                    continue
                if not sorted((grid / model / f"seed_{seed}").glob("fold_*/fold_*_posteriors.npz")):
                    continue
                B = backbone_oof(grid, model, seed, stems, n_classes, y)
                P = z[key].astype(np.float64)
                rows.append(unit_row(corpus, "cv", model, seed, P, B, y, args.folds))
                pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
                log(f"    {corpus}/{model}/seed {seed}: backbone {rows[-1]['backbone']:.2f} "
                    f"w=0,5 {rows[-1]['w_half']:.2f} cross {rows[-1]['w_cross']:.2f}")
        log(f"  {corpus}: {len([r for r in rows if r['corpus'] == corpus])} unidades")

    t = pd.DataFrame(rows)
    log(f"\n{'corpus':<11}{'backbone':>10}{'w=0,5':>9}{'cross-fit':>11}"
        f"{'espiando':>10}   {'delta cross':>12}{'delta 0,5':>11}")
    for c, g in t.groupby("corpus", sort=False):
        log(f"{c:<11}{g.backbone.mean():>10.2f}{g.w_half.mean():>9.2f}"
            f"{g.w_cross.mean():>11.2f}{g.w_peek.mean():>10.2f}"
            f"{g.w_cross.mean()-g.backbone.mean():>+12.2f}"
            f"{g.w_half.mean()-g.backbone.mean():>+11.2f}")
    log(f"\nFIM: {args.output}")


if __name__ == "__main__":
    main()
