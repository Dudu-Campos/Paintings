"""Auditoria de daltonismo da paleta canônica de views.

Roda em segundos e não depende de dado nenhum -- é um teste da paleta, não um
experimento. **Rodar sempre que `view_names.FAMILY_SLOT` ou `_STEPS` mudarem.**

Método: simula os três dicromatismos com as matrizes de Machado, Oliveira &
Fernandes (2009) em severidade 1,0, aplicadas em RGB **linear**, e mede distância
em CIELAB (dE76). dE76 basta para a pergunta aqui, que é "dá para separar de
relance?" e não diferença sub-limiar.

Como ler a saída, porque a distinção é o ponto todo:

* **Pares entre famílias** têm de passar. É onde a cor carrega identidade
  sozinha, e a paleta foi escolhida por busca exaustiva maximizando justamente
  esse mínimo -- hoje **dE 13,3**, contra 9,7 da primeira versão (que tinha
  `Felzenszwalb x Fourier` colapsando sob protanopia).
* **Pares dentro da família** ficam apertados de propósito: a rampa de tom É a
  marca de parentesco, e dois membros da mesma família parecidos comunicam a
  coisa certa. A separação ali é por **luminosidade** (dL 7 a 9), que sobrevive
  ao daltonismo. O piso é dE 7,5 (`Gabor x Fourier`, deuteranopia).

**Consequência de projeto, e ela é obrigatória:** com piso de 7,5 dentro da
família, nenhuma figura pode identificar view **só** por cor. Ou rótulo direto
(o slopegraph), ou rótulo de eixo (os barplots), ou marcador diferente (a figura
de custo por resolução). A paleta agrupa; ela não nomeia.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from view_names import (FAMILY_SLOT, SLOTS, STRENGTH, colour, display,  # noqa: E402
                        family_of)

# Machado, Oliveira & Fernandes (2009), severidade 1,0, para RGB linear.
DICHROMACY = {
    "protanopia": [[0.152286, 1.052583, -0.204868],
                   [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]],
    "deuteranopia": [[0.367322, 0.860646, -0.227968],
                     [0.280085, 0.672501, 0.047413],
                     [-0.011820, 0.042940, 0.968881]],
    "tritanopia": [[1.255528, -0.076749, -0.178779],
                   [-0.078411, 0.930809, 0.147602],
                   [0.004733, 0.691367, 0.303900]],
}
XYZ = np.array([[0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def to_linear(hex_colour: str) -> np.ndarray:
    channels = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return np.where(channels <= 0.04045, channels / 12.92,
                    ((channels + 0.055) / 1.055) ** 2.4)


def to_lab(linear: np.ndarray) -> np.ndarray:
    xyz = XYZ @ np.clip(linear, 0, 1) / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _hue(hex_colour: str) -> float:
    _, a, b = to_lab(to_linear(hex_colour))
    return float(np.degrees(np.arctan2(b, a)) % 360)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floor-cross", type=float, default=12.0,
                    help="dE mínimo exigido ENTRE famílias; abaixo disso falha")
    ap.add_argument("--floor-hue", type=float, default=60.0,
                    help="separação mínima de MATIZ (graus) entre a família "
                         "âncora e as outras. dE e matiz medem coisas "
                         "diferentes e as duas têm de passar -- ver abaixo.")
    ap.add_argument("--anchor", default="Region graphs")
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    views = STRENGTH + ["Similarity graph", "Text"]
    worst_cross = np.inf
    for condition in ["normal", *DICHROMACY]:
        labs = {}
        for name in views:
            linear = to_linear(colour(name))
            if condition != "normal":
                linear = np.clip(np.array(DICHROMACY[condition]) @ linear, 0, 1)
            labs[name] = to_lab(linear)

        rows = sorted((float(np.linalg.norm(labs[a] - labs[b])), a, b)
                      for i, a in enumerate(views) for b in views[i + 1:])
        cross = [r for r in rows if family_of(r[1]) != family_of(r[2])]
        worst_cross = min(worst_cross, cross[0][0])
        print(f"=== {condition}: menor dE {rows[0][0]:.1f} "
              f"(entre famílias {cross[0][0]:.1f})")
        for distance, a, b in rows[:args.show]:
            same = family_of(a) == family_of(b)
            lightness = abs(labs[a][0] - labs[b][0])
            print(f"    dE {distance:5.1f}  dL {lightness:4.1f}  "
                  f"{display(a):18s} x {display(b):18s}"
                  f"{'  (mesma família, esperado)' if same else '  <-- ENTRE FAMÍLIAS'}")
        print()

    print(f"pior par entre famílias, nos quatro cenários: dE {worst_cross:.1f}")
    if worst_cross < args.floor_cross:
        raise SystemExit(f"FALHA: abaixo do piso de {args.floor_cross}")

    # Segundo teste, e ele existe porque o primeiro NÃO pega o defeito que ele
    # pega. Uma versão desta paleta punha `Image statistics` no violeta, a dE 30
    # e 45 dos vizinhos -- folgadíssimo pela métrica -- e mesmo assim lia como
    # "mais um dos azuis". dE mede **discriminabilidade**; a pergunta de uma
    # paleta que codifica família é **categoria**, e categoria é matiz.
    # A checagem é contra a ÂNCORA, não entre todos os pares, e a razão é
    # aritmética: seis famílias num círculo de matiz dão 60° de separação média,
    # então um piso de 60° entre todos os pares é impossível por construção. O
    # melhor alcançável com estes slots é ~25°, e quatro pares ficam abaixo de
    # 60. O que resolve o resíduo NÃO é matiz, é a hachura por família
    # (`view_names.FAMILY_HATCH`) -- canal redundante, que ainda por cima
    # sobrevive a impressão em cinza. Aqui só se exige que ninguém encoste na
    # família âncora, que é a de quatro views e domina as figuras.
    anchor_hue = _hue(SLOTS[FAMILY_SLOT[args.anchor]])
    print(f"\nseparação de matiz contra a âncora ({args.anchor}):")
    worst_hue = 360.0
    for family, slot in FAMILY_SLOT.items():
        if family == args.anchor:
            continue
        delta = abs(_hue(SLOTS[slot]) - anchor_hue) % 360
        delta = min(delta, 360 - delta)
        worst_hue = min(worst_hue, delta)
        print(f"    {family:22s} {SLOTS[slot]}  {delta:6.1f}°"
              f"{'  <-- ABAIXO DO PISO' if delta < args.floor_hue else ''}")
    if worst_hue < args.floor_hue:
        raise SystemExit(f"FALHA: matiz a {worst_hue:.1f}° da âncora, "
                         f"piso {args.floor_hue}°")

    from view_names import hatch
    # `STRENGTH` não lista `Similarity graph` nem `Text` (são filhos únicos da
    # matiz deles e a rampa não os alcança), então os dois entram explícitos --
    # sem isso o laço os pula e a hachura deles fica SEM verificação de colisão.
    marked = {"Text": hatch("Text"), "Similarity graph": hatch("Similarity graph")}
    for family in FAMILY_SLOT:
        members = [v for v in STRENGTH if family_of(v) == family]
        if members:
            marked[family] = hatch(members[0])
    if len(set(marked.values())) != len(marked):
        raise SystemExit(f"FALHA: famílias com a mesma hachura: {marked}")
    print("\nhachura por família (canal redundante, sobrevive a cinza):")
    for family, pattern in marked.items():
        shown = repr(pattern) if pattern else "lisa"
        print(f"    {family:22s} {shown}")
    print("\nok")


if __name__ == "__main__":
    main()
