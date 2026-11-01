"""Fixa os numeros conhecidos do pipeline, para que refator quebre ALTO.

**Por que existe.** O valor deste codigo esta em restricoes DECLARADAS, nao na
estrutura, e as restricoes sao invisiveis para quem le so o codigo. Quatro
exemplos que ja morderam:

* `VISUAL` contem `Fourier` e NAO contem `Gabor`; `HAND8` sao as oito. Somar
  `"Fourier"` a `VISUAL` duplica colunas **sem erro nenhum** (§10.8) -- aconteceu
  em 06/set ao derivar o teste de transferencia do Fourier.
* O banco EMITE 639d e USA 617d. As 22 mortas nao sao bug: sao o anel DC do
  Fourier (18) e duas fracoes de folha dos multigrafos (§3).
* `felz_scale` esta em unidades de PIXEL, entao resolucao e granularidade sao o
  mesmo botao -- o §5.17e mediu a contagem de segmentos mudando 8,6x ao variar
  so a resolucao.
* O `BranchMLP` manda todo bloco para 128 dims, entao a fracao do vetor cru NAO
  e o que a cabeca ve (§5.6, correcao de 31/ago).

Um refator "limpo" normaliza as quatro e as tabelas do artigo param de
reproduzir, **em silencio**. Estes testes convertem isso em falha ruidosa.

Rodar antes e depois de qualquer reorganizacao, e antes de entregar o codigo a
quem nao leu o CLAUDE.md:

    .ABSvenv/bin/python3 experiments/test_invariants.py

Os testes que exigem GPU ou features em disco sao PULADOS com aviso, nunca
silenciosamente -- um teste que nao rodou nao e um teste que passou.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in ("experiments", "src/pipeline", "experiments/PRISM", "src/baselines"):
    sys.path.insert(0, str(ROOT / _p))

PASS, FAIL, SKIP = [], [], []


def check(name: str):
    def wrap(fn):
        try:
            msg = fn()
        except SkipTest as exc:
            SKIP.append((name, str(exc)))
        except Exception as exc:
            FAIL.append((name, f"{type(exc).__name__}: {exc}"))
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
        else:
            PASS.append((name, msg or ""))
        return fn
    return wrap


class SkipTest(Exception):
    pass


# --------------------------------------------------------------------------
@check("VISUAL x HAND8: listas distintas, e VISUAL nao tem Gabor")
def _():
    from corpora import HAND8, VISUAL
    assert "Gabor" not in VISUAL, "VISUAL nao pode conter Gabor (§3)"
    assert "Fourier" in VISUAL, "VISUAL tem de conter Fourier"
    assert set(HAND8) == set(VISUAL) | {"Gabor"}, "HAND8 != VISUAL + Gabor"
    assert len(HAND8) == 8 and len(VISUAL) == 7
    return "VISUAL 7, HAND8 8"


@check("esquema do multigrafo: 71 dims, 2 mortas nomeadas")
def _():
    from corpora import DEAD_MULTIGRAPH_COLUMNS, FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 71, f"{len(FEATURE_COLUMNS)} != 71"
    for c in DEAD_MULTIGRAPH_COLUMNS:
        assert c in FEATURE_COLUMNS, f"{c} sumiu do esquema"
    return "71 dims, mortas = " + ", ".join(DEAD_MULTIGRAPH_COLUMNS)


@check("protocolo: SEED 42, N_FOLDS 5")
def _():
    from protocol import N_FOLDS, SEED
    assert (SEED, N_FOLDS) == (42, 5), f"({SEED}, {N_FOLDS}) != (42, 5)"
    return "SEED 42, 5 folds"


@check("BranchMLP: 128 por bloco, 778.905 params no hand-8 de 25 classes")
def _():
    try:
        from models import BranchMLP
    except Exception as exc:
        raise SkipTest(f"torch indisponivel ({type(exc).__name__})")
    dims = [130, 71, 71, 63, 63, 57, 152, 32]          # 639d BRUTO
    assert sum(dims) == 639
    n = sum(p.numel() for p in BranchMLP(dims, 25).parameters())
    assert n == 778905, f"{n:,} != 778.905 (§5.4b) -- arquitetura mudou"
    # a fracao do vetor cru NAO e o que a cabeca ve: todo bloco vai para 128
    m = BranchMLP(dims, 25)
    outs = {e[0].out_features for e in m.encoders}
    assert outs == {128}, f"encoders emitem {outs}, esperado 128 (§5.6)"
    return "778.905 params, todo bloco -> 128"


@check("Fourier: defaults 256/12/12 e 152 dims")
def _():
    import extract_fourier_spectrum_features as s
    assert (s.SIZE, s.RADIAL_BINS, s.ANGULAR_BINS) == (256, 12, 12)
    assert len(s.COMPACT_FEATURE_NAMES) == 152
    n0 = len(s.COMPACT_FEATURE_NAMES)
    s.configure(128, 8, 16)
    assert len(s.COMPACT_FEATURE_NAMES) == 8 * 16 + 8, "nomes nao seguiram as bordas"
    s.configure(256, 12, 12)
    assert len(s.COMPACT_FEATURE_NAMES) == n0, "round-trip nao volta ao default"
    return "256/12/12 -> 152 dims, configure faz round-trip"


@check("registry: multitask hand-8 = 617d usados em 99.816 imagens")
def _():
    from registry import load_blocks
    try:
        c = load_blocks("multitask", list(__import__("corpora").HAND8))
    except Exception as exc:
        raise SkipTest(f"features fora do disco ({type(exc).__name__})")
    d = sum(b.shape[1] for b in c.blocks.values())
    assert d == 617, f"{d} != 617 -- DEAD_COLUMNS mudou (§3: emite 639, usa 617)"
    assert len(c.stems) == 99816, f"{len(c.stems):,} != 99.816"
    return "617d, 99.816 imagens"


@check("view_names: display() e a fonte unica de nome")
def _():
    from view_names import display
    esperado = {"Descriptors": "Image descriptors", "L(G_felz)": "Line Felz",
                "Line graph": "Line SLIC", "Felzenszwalb": "Felzenszwalb"}
    for k, v in esperado.items():
        got = display(k)
        assert got == v, f"display({k!r}) = {got!r}, esperado {v!r} (§7.18)"
    return "4 nomes canonicos conferidos"


@check("paleta: piso de dE 12 entre familias sob os tres dicromatismos")
def _():
    import itertools
    import numpy as np
    from audit_palette_cvd import to_lab, to_linear
    from view_names import FAMILY_SLOT, SLOTS
    CVD = [np.eye(3),
           np.array([[.152286, 1.052583, -.204868], [.114503, .786281, .099216],
                     [-.003882, -.048116, 1.051998]]),
           np.array([[.367322, .860646, -.227968], [.280085, .672501, .047413],
                     [-.011820, .042940, .968881]]),
           np.array([[1.255528, -.076749, -.178779], [-.078411, .930809, .147602],
                     [.004733, .691367, .303900]])]
    cores = [SLOTS[s] for s in sorted(set(FAMILY_SLOT.values()))]
    pior = min(float(np.linalg.norm(to_lab(M @ to_linear(a)) - to_lab(M @ to_linear(b))))
               for a, b in itertools.combinations(cores, 2) for M in CVD)
    assert pior >= 12.0, f"pior dE entre familias = {pior:.1f} < 12,0 (§7.18)"
    return f"pior dE = {pior:.1f}"


def main() -> None:
    print(f"{'':2}{'invariante':<62} resultado")
    print("-" * 92)
    for name, msg in PASS:
        print(f"OK  {name:<62} {msg}")
    for name, msg in SKIP:
        print(f"--  {name:<62} PULADO: {msg}")
    for name, msg in FAIL:
        print(f"XX  {name:<62} {msg}")
    print("-" * 92)
    print(f"{len(PASS)} ok, {len(SKIP)} pulados, {len(FAIL)} FALHAS")
    if SKIP:
        print("\nAVISO: teste pulado NAO e teste que passou. Rodar na maquina com "
              "GPU e features em disco antes de confiar.")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
