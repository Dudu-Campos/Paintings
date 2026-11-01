"""Tabela principal da fusao de decisao: peso FIXO (w=0,5), com e sem texto.

Decisao de reporte de 23/set (§4.1f): a tabela do artigo usa o peso fixo, nao o
cross-fit, e carrega as DUAS linhas -- sem texto e com texto. A linha sem texto
nao e opcional: com texto os dois lados carregam supervisao externa (§7.2), e so
ela sustenta a alegacao de zero supervisao externa do §4.1.

O script NAO treina nada. Ele agrega os posteriors ja gravados pelos
`evaluate_decision_fusion_{cv,official}.py` nos dois bracos.

Duas convencoes que vem do §4.1d e nao podem mudar:
  * Delta e PAREADO POR UNIDADE (corpus x modelo x semente): calcula-se
    `fusao - backbone` dentro da unidade antes de agregar. Fusao e backbone vem
    do mesmo treino, entao parear cancela a variacao comum; agregar cada lado e
    subtrair depois inflaria a barra.
  * A metrica e BAcc. Nos tres corpora de teste balanceado Acc = BAcc.

Reproducao conferida contra os valores registrados no §4.1f (itens 1 e 3): o
script aborta se qualquer celula divergir mais de 0,01pp.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# corpus -> (arquivo sem texto, arquivo com texto)
SOURCES = {
    "rijksdraw":   ("rijksdraw_decision_fusion_cv.csv",   "rijksdraw_decision_fusion_cv_text.csv"),
    "nct_crc":     ("nct_crc_decision_fusion.csv",        "nct_crc_decision_fusion_text.csv"),
    "multitask":   ("multitask_decision_fusion_cv.csv",   "multitask_decision_fusion_cv_text.csv"),
    "oilpainting": ("oilpainting_decision_fusion_cv.csv", "oilpainting_decision_fusion_cv_text.csv"),
    "places_lt":   ("places_lt_decision_fusion.csv",      "places_lt_decision_fusion_text.csv"),
    "minc2500":    ("minc2500_decision_fusion.csv",       "minc2500_decision_fusion_text.csv"),
    "food101":     ("food101_decision_fusion.csv",        "food101_decision_fusion_text.csv"),
}

# §4.1f item 1 (sem texto) e item 3 (com texto). Serve de conferencia, nao de fonte.
ESPERADO = {
    ("rijksdraw",   "sem"): +3.83, ("rijksdraw",   "com"): +3.83,
    ("nct_crc",     "sem"): +0.92, ("nct_crc",     "com"): +1.06,
    ("multitask",   "sem"): +3.99, ("multitask",   "com"): +4.58,
    ("oilpainting", "sem"): +1.47, ("oilpainting", "com"): +2.08,
    ("places_lt",   "sem"): -0.28, ("places_lt",   "com"): +1.42,
    ("minc2500",    "sem"): -1.69, ("minc2500",    "com"): -0.40,
    ("food101",     "sem"): -0.88, ("food101",     "com"): +0.29,
}


# sufixo no CSV -> nome curto da metrica. BAcc e a principal (§2); Acc e macro-F1
# saem juntas do mesmo treino, entao sao a MESMA celula do 2x2 do §7.12 (treino
# ponderado), nao uma configuracao alternativa.
METRICAS = {"bacc": "BAcc", "accuracy": "Acc", "macro_f1": "macro-F1"}


def unidades(path: Path, corpus: str, arm: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    pedaços = []
    for sufixo, rotulo in METRICAS.items():
        cols = {lado: f"{lado}_{sufixo}" for lado in ("ours", "backbone", "fusion")}
        faltando = {"model", "seed", *cols.values()} - set(frame.columns)
        if faltando:
            raise SystemExit(f"{path.name}: faltam colunas {sorted(faltando)}")
        bloco = frame[["model", "seed"]].copy()
        bloco.insert(0, "corpus", corpus)
        bloco.insert(1, "arm", arm)
        bloco.insert(2, "metrica", rotulo)
        for lado, c in cols.items():
            bloco[lado] = frame[c]
        bloco["delta"] = bloco["fusion"] - bloco["backbone"]
        bloco["gap"] = bloco["backbone"] - bloco["ours"]
        pedaços.append(bloco)
    return pd.concat(pedaços, ignore_index=True)


def agrega(bloco: pd.DataFrame) -> dict:
    d = bloco["delta"].to_numpy()
    n = len(d)
    # t pareado equivale ao t de uma amostra sobre os deltas por unidade
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if n > 1 and np.ptp(d) > 0 else float("nan")
    return {
        "n_unidades": n,
        "nosso": bloco["ours"].mean(),
        "backbone": bloco["backbone"].mean(),
        "gap": bloco["gap"].mean(),
        "fusao": bloco["fusion"].mean(),
        "delta": d.mean(),
        "delta_sd": d.std(ddof=1) if n > 1 else float("nan"),
        "p": p,
        "positivas": int((d > 0).sum()),
    }


def main() -> None:
    todas = []
    for corpus, (sem, com) in SOURCES.items():
        for arm, nome in (("sem", sem), ("com", com)):
            path = RESULTS / nome
            if not path.exists():
                raise SystemExit(f"ausente: {path}")
            todas.append(unidades(path, corpus, arm))
    unid = pd.concat(todas, ignore_index=True)

    por_modelo = (unid.groupby(["corpus", "arm", "metrica", "model"], sort=False)
                      .apply(agrega, include_groups=False)
                      .apply(pd.Series).reset_index())
    todas_metricas = (unid.groupby(["corpus", "arm", "metrica"], sort=False)
                          .apply(agrega, include_groups=False)
                          .apply(pd.Series).reset_index())
    # a conferencia e o corpo do relatorio usam a metrica principal
    por_corpus = todas_metricas[todas_metricas.metrica == "BAcc"].reset_index(drop=True)

    # ---- conferencia contra o §4.1f ------------------------------------
    print("CONFERENCIA contra os valores registrados no §4.1f (tolerancia 0,01pp)")
    pior = 0.0
    for (corpus, arm), esperado in ESPERADO.items():
        linha = por_corpus[(por_corpus.corpus == corpus) & (por_corpus.arm == arm)]
        if linha.empty:
            raise SystemExit(f"sem linha para {corpus}/{arm}")
        medido = float(linha["delta"].iloc[0])
        diff = abs(medido - esperado)
        pior = max(pior, diff)
        marca = "ok " if diff <= 0.01 else "DIVERGE"
        print(f"  {marca} {corpus:<12} {arm}  registrado {esperado:+.2f}  medido {medido:+.2f}  ({diff:.3f})")
    print(f"  maior divergencia: {pior:.4f}pp")
    if pior > 0.01:
        raise SystemExit(
            "a agregacao NAO reproduz os valores do §4.1f -- algum CSV foi "
            "regravado desde que a secao foi escrita; investigar antes de reportar")

    # ---- unidades por celula ------------------------------------------
    esperadas = por_corpus["n_unidades"].unique()
    print(f"\nunidades por celula: {sorted(esperadas)} (10 = 2 modelos x 5 sementes)")
    if set(esperadas) != {10}:
        raise SystemExit("alguma celula nao tem 10 unidades -- conferir antes de reportar")

    unid.to_csv(RESULTS / "fusion_main_table_units.csv", index=False)
    por_modelo.to_csv(RESULTS / "fusion_main_table_by_model.csv", index=False)
    todas_metricas.to_csv(RESULTS / "fusion_main_table.csv", index=False)

    # ---- markdown ------------------------------------------------------
    ordem = list(SOURCES)
    print("\n" + "=" * 78)
    print("TABELA PRINCIPAL -- fusao de decisao com peso FIXO w=0,5, 10 unidades por celula")
    print("=" * 78)
    print("\n| corpus | cond. | nosso | backbone | gap | fusao | Δ | p | unid. |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for corpus in ordem:
        for arm, rotulo in (("sem", "sem texto"), ("com", "**com texto**")):
            r = por_corpus[(por_corpus.corpus == corpus) & (por_corpus.arm == arm)].iloc[0]
            d = f"{r.delta:+.2f} ± {r.delta_sd:.2f}"
            print(f"| {corpus} | {rotulo} | {r.nosso:.2f} | {r.backbone:.2f} | "
                  f"{r.gap:.1f} | {r.fusao:.2f} | **{d}** | {r.p:.2g} | "
                  f"{int(r.positivas)}/{int(r.n_unidades)} |")

    print("\n\n| corpus | Δ sem texto | Δ com texto | gap sem → com |")
    print("|---|---:|---:|---:|")
    for corpus in ordem:
        s = por_corpus[(por_corpus.corpus == corpus) & (por_corpus.arm == "sem")].iloc[0]
        c = por_corpus[(por_corpus.corpus == corpus) & (por_corpus.arm == "com")].iloc[0]
        print(f"| {corpus} | {s.delta:+.2f} ({int(s.positivas)}/10) | "
              f"{c.delta:+.2f} ({int(c.positivas)}/10) | {s.gap:.1f} → {c.gap:.1f} |")

    # ---- as tres metricas ----------------------------------------------
    # §7.12: o treino usa class_weight, que otimiza BAcc e SACRIFICA acuracia de
    # proposito. As tres colunas sao a MESMA celula (treino ponderado); nao sao
    # configuracoes alternativas, e a tabela tem de declarar isso.
    BALANCEADOS = {"places_lt", "minc2500", "food101"}
    conf = todas_metricas.pivot_table(index=["corpus", "arm"], columns="metrica",
                                      values="delta")
    pior_bal = max(abs(conf.loc[(c, a), "Acc"] - conf.loc[(c, a), "BAcc"])
                   for c in BALANCEADOS for a in ("sem", "com"))
    print(f"\n\nteste balanceado -> Acc = BAcc: maior divergencia {pior_bal:.4f}pp "
          f"em {sorted(BALANCEADOS)}")
    if pior_bal > 0.005:
        raise SystemExit("Acc != BAcc num corpus de teste balanceado -- investigar")

    print("\nΔ DA FUSAO NAS TRES METRICAS (w=0,5, 10 unidades por celula)")
    print("\n| corpus | cond. | Δ BAcc | Δ Acc | Δ macro-F1 | sinais |")
    print("|---|---|---:|---:|---:|---|")
    discordantes = []
    for corpus in ordem:
        for arm, rotulo in (("sem", "sem texto"), ("com", "com texto")):
            r = conf.loc[(corpus, arm)]
            sinais = {np.sign(r["BAcc"]), np.sign(r["Acc"]), np.sign(r["macro_F1"])} \
                if "macro_F1" in r else {np.sign(r[m]) for m in ("BAcc", "Acc", "macro-F1")}
            marca = "concordam" if len(sinais) == 1 else "**DISCORDAM**"
            if len(sinais) > 1:
                discordantes.append((corpus, arm))
            print(f"| {corpus} | {rotulo} | {r['BAcc']:+.2f} | {r['Acc']:+.2f} | "
                  f"{r['macro-F1']:+.2f} | {marca} |")
    print(f"\ncelulas em que as tres metricas DISCORDAM de sinal: "
          f"{len(discordantes)} de 14  {discordantes if discordantes else ''}")

    print("\nNIVEL absoluto da fusao nas tres metricas (media das 10 unidades)")
    print("\n| corpus | cond. | BAcc | Acc | macro-F1 |")
    print("|---|---|---:|---:|---:|")
    niv = todas_metricas.pivot_table(index=["corpus", "arm"], columns="metrica", values="fusao")
    for corpus in ordem:
        for arm, rotulo in (("sem", "sem texto"), ("com", "com texto")):
            r = niv.loc[(corpus, arm)]
            print(f"| {corpus} | {rotulo} | {r['BAcc']:.2f} | {r['Acc']:.2f} | {r['macro-F1']:.2f} |")

    # ---- efeito do texto sobre a FUSAO, pareado por unidade ------------
    # Nao e o marginal do §5.18e (que e sobre o nosso lado sozinho): aqui o
    # estimando e quanto o canal muda o desfecho DA FUSAO, com o backbone fixo.
    print("\n\nO QUE O TEXTO MUDA NA FUSAO -- pareado por unidade (mesmo corpus/modelo/semente)")
    print("\n| corpus | Δ(com) − Δ(sem) | p | unid. a favor |")
    print("|---|---:|---:|---:|")
    largo = (unid[unid.metrica == "BAcc"]
             .pivot_table(index=["corpus", "model", "seed"], columns="arm",
                          values="delta").reset_index())
    for corpus in ordem:
        sub = largo[largo.corpus == corpus]
        dd = (sub["com"] - sub["sem"]).to_numpy()
        pv = float(stats.ttest_1samp(dd, 0.0).pvalue)
        print(f"| {corpus} | {dd.mean():+.2f} ± {dd.std(ddof=1):.2f} | {pv:.2g} | "
              f"{int((dd > 0).sum())}/{len(dd)} |")

    # ---- celulas que NAO se separam de zero ----------------------------
    frouxas = por_corpus[por_corpus.p > 0.05]
    print("\nCELULAS QUE NAO SE SEPARAM DE ZERO (p > 0,05) -- a legenda tem de dize-lo,")
    print("nos dois sentidos: o negativo que sobra e os positivos pequenos.")
    for _, r in frouxas.iterrows():
        print(f"  {r.corpus:<12} {r.arm} texto   Δ {r.delta:+.2f}  p={r.p:.2f}  "
              f"{int(r.positivas)}/{int(r.n_unidades)}")

    for arm, rotulo in (("sem", "sem texto"), ("com", "com texto")):
        sub = por_corpus[por_corpus.arm == arm]
        u = unid[(unid.arm == arm) & (unid.metrica == "BAcc")]["delta"]
        pos_corpus = int((sub.delta > 0).sum())
        print(f"\n{rotulo:<10} corpora positivos {pos_corpus}/7 | "
              f"unidades positivas {int((u > 0).sum())}/{len(u)} | "
              f"faixa {sub.delta.min():+.2f} a {sub.delta.max():+.2f}")

    print("\nescritos: results/fusion_main_table{,_by_model,_units}.csv")


if __name__ == "__main__":
    main()
