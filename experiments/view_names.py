"""Nomes de exibição das views, num lugar só.

As chaves internas vêm do registro e de arquivos de resultado já gravados;
renomeá-las quebraria todo CSV em `results/` e os scripts que os leem. Então a
tradução acontece na **borda de apresentação**: os dados guardam a chave, as
figuras mostram o nome.

Duas correções que motivam o arquivo:

  `Line graph` não dizia de qual segmentação era o grafo de linha -- ele é o do
  SLIC, e o do Felzenszwalb chamava-se `L(G_felz)`. Duas convenções diferentes
  para a mesma operação. Agora os quatro blocos de grafo seguem o mesmo padrão:
  segmentador (Felzenszwalb, SLIC) e grafo de linha dele (Line Felz, Line SLIC).

  `Descriptors` é ambíguo num artigo em que tudo é descritor; `Image
  descriptors` diz que são medidas globais da imagem.
"""
from __future__ import annotations

DISPLAY = {
    "Descriptors": "Image descriptors",
    "L(G_felz)": "Line Felz",
    "Line graph": "Line SLIC",
}


# Traducao de EXIBICAO, nao de chave. O §7.18 fixa que as chaves internas vem do
# registro e dos CSVs ja gravados e NAO podem ser renomeadas -- a traducao mora
# na borda de apresentacao, que e aqui. `lang` e opcional e o default e "en",
# entao nenhum consumidor existente muda de comportamento.
#
# Nomes proprios (Felzenszwalb, Gabor, Hilbert, Fourier) e siglas (SLIC, HVG)
# ficam como estao. "view" fica em ingles porque e o termo que o projeto usa em
# portugues no CLAUDE.md inteiro -- traduzi-lo aqui e nao la criaria duas
# palavras para a mesma coisa.
DISPLAY_PT = {
    "Similarity graph": "Grafo de similaridade",
    "Descriptors": "Descritores de imagem",
    "L(G_felz)": "Grafo linha (Felz)",
    "Line graph": "Grafo linha (SLIC)",
    "Random walk": "Passeio aleatório",
    "Text": "Texto",
    "Caption embedding": "Embedding da legenda",
}


def display(name: str, lang: str = "en") -> str:
    if lang == "pt":
        return DISPLAY_PT.get(name, DISPLAY.get(name, name))
    return DISPLAY.get(name, name)


def displays(names, lang: str = "en") -> list[str]:
    return [display(n, lang) for n in names]


# Taxonomia de famílias. Derivada do dendrograma de concordância (clados com
# suporte 1,00 em 200 reamostragens) e depois **refinada por procedência**, que é
# o critério do artigo:
#
#   - `(Gabor, Fourier, Text)` é um clado real (suporte 1,00), mas Text fica à
#     parte: ele é a única fonte fora da imagem, e errar junto com duas views
#     fracas é provavelmente acertar as mesmas classes fáceis, não medir algo
#     parecido. Agrupar por padrão de erro aqui contradiria o eixo do artigo.
#   - `(Felzenszwalb, SLIC, Line Felz, Line SLIC)` também é clado de suporte
#     1,00, mas se divide em dois sub-clados igualmente sólidos -- segmentação e
#     grafo de linha são operações diferentes sobre o mesmo RAG.
#   - `Image descriptors` fica sozinha: o nó que a juntava à série temporal tem
#     suporte **0,57**, ou seja, em 43% das reamostragens a topologia ali é
#     outra.
FAMILIES = {
    "Space-filling curve": ["Hilbert", "Random walk", "HVG"],
    "Image statistics": ["Descriptors"],
    "Region graphs": ["Felzenszwalb", "SLIC", "L(G_felz)", "Line graph"],
    "Similarity graph": ["Similarity graph"],
    "Spectral & text": ["Fourier", "Gabor", "Text"],
}

# NOTA DE NOMENCLATURA, para não passar batido: "Spectral & text" é um clado
# empírico, não uma família de medida. O que une Fourier, Gabor e Text é serem
# as três views mais fracas e acertarem as mesmas classes fáceis -- Text é a
# única fonte FORA da imagem e não tem parentesco com as outras duas. O nome
# composto é deliberado: agrupar por padrão de erro é o critério aqui, e o texto
# do artigo tem de dizer isso em vez de deixar "Spectral" sugerir que Text mede
# algo espectral.

def family_of(name: str) -> str:
    for family, members in FAMILIES.items():
        if name in members:
            return family
    raise KeyError(f"view sem família: {name}")


# ---------------------------------------------------------------------------
# Paleta canônica por view (1/set)
# ---------------------------------------------------------------------------
#
# Antes disto havia **quatro** mapas view->cor incompatíveis em `experiments/`:
# `plot_modality_cost_by_resolution` (8 cores independentes),
# `plot_ablation_overview` (cor por família, 5 slots),
# `plot_standalone_by_classifier` (rampas por família) e
# `plot_redundancy_two_spaces` (slots 1-7 em ordem de bloco). A mesma view saía
# de cores diferentes em figuras vizinhas do mesmo artigo.
#
# A regra é **uma matiz por família, um tom por view dentro da família**. Isso
# atende os dois consumidores com uma fonte só: a figura que quer agrupar usa
# `family_colour`, a que precisa distinguir as doze usa `colour`. E de graça a
# estrutura de famílias fica legível em toda figura -- as quatro views de grafo
# de região aparecem como quatro azuis, e o leitor vê o clado sem legenda.
#
# As matizes vêm da paleta categórica de 8 slots já usada no projeto, e a
# atribuição às famílias é a de `plot_ablation_overview.py`, que é a que declara
# a intenção de "as figuras do paper ficarem consistentes entre si". O único
# slot novo é o do grafo de similaridade, que não existia lá.
SLOTS = {
    1: "#2a78d6",  # azul
    2: "#eb6834",  # laranja
    3: "#1baf7a",  # aqua
    4: "#eda100",  # âmbar
    5: "#e87ba4",  # rosa
    6: "#008300",  # verde
    7: "#4a3aa7",  # violeta
    8: "#e34948",  # vermelho
}

# A atribuição não é estética: foi ESCOLHIDA por busca exaustiva sobre as 8P5
# permutações de slot, maximizando a menor distância CIELAB **entre famílias** no
# pior dos quatro cenários de visão (normal + protanopia, deuteranopia e
# tritanopia simuladas por Machado, Oliveira & Fernandes 2009, severidade 1,0).
#
# A primeira versão, que seguia `plot_ablation_overview.py`, tinha
# `Felzenszwalb x Fourier` a dE 9,7 sob protanopia -- azul e violeta colapsam
# para deficiência vermelho-verde, que é a comum (~8% dos homens). Esta
# atribuição leva o pior par entre famílias para **dE 12,9**.
#
# Duas âncoras foram fixadas na busca em vez de otimizadas: `Region graphs` no
# azul (é a família maior e a convenção mais usada nas figuras existentes) e
# `Text` no vermelho (ele é o outlier do banco, a única fonte fora da imagem).
#
# **Segunda restrição, e ela veio de uma revisão de olho, não da métrica.** A
# versão que maximizava dE punha `Image statistics` no violeta (slot 7), e o
# usuário apontou em 1/set que aquilo "estava perto demais dos grafos". A métrica
# discordava -- contra os dois vizinhos na figura dava dE 30,8 e 45,7, muito
# acima do piso. Ele estava certo e a métrica estava respondendo outra pergunta:
# **dE mede discriminabilidade, não categoria**. Violeta escuro entre azuis é
# *nomeado* "um azul mais escuro", e a função desta paleta é justamente dizer
# "outra família".
#
# Medido depois da observação: o slot 7 está a **22,9° de matiz do azul**, contra
# 76-156° de todos os outros slots. É o único abaixo de 60°. A regra que fica é
# **nenhuma família dentro de 60° do azul**, porque azul é a família de quatro
# views que domina as figuras. Proibir o slot 7 custa 0,4 de dE (13,3 -> 12,9) e
# leva a separação de matiz de 22,9° para 76,3°.
#
# De brinde, `Image statistics` volta para o aqua `#1baf7a`, que é o que
# `plot_ablation_overview.py` já usava -- a única view cuja memória visual
# existente é preservada.
FAMILY_SLOT = {
    "Region graphs": 1,
    "Similarity graph": 6,
    "Image statistics": 3,
    "Space-filling curve": 4,
    "Spectral & text": 5,
}

# Text sai da matiz da família dele: o clado "Spectral & text" é empírico (por
# padrão de erro), e o próprio módulo já avisa acima que Text não mede nada
# espectral. Pintá-lo de violeta junto com Fourier e Gabor faria a figura
# afirmar um parentesco que o artigo nega. Fica no slot 8, sozinho.
OVERRIDE_SLOT = {"Text": 8}


def _shade(hex_colour: str, amount: float) -> str:
    """Clareia em direção ao branco. `amount` 0 = a matiz, 1 = branco.

    Teto de 0,62 na tabela abaixo: acima disso o tom não tem contraste
    suficiente contra o fundo claro das figuras (#fbfbfa).
    """
    base = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in base)


# Ordem de força MEDIDA (§5.9d, coluna do MLP em 125 classes), usada só para
# decidir qual membro de cada família fica com a matiz cheia e quais clareiam.
# É deliberadamente separada de `FAMILIES`: aquilo é taxonomia, derivada do
# dendrograma de concordância, e reordenar a taxonomia para servir a cor
# trocaria o significado pela decoração. Aqui a rampa desce junto com as barras
# numa figura ordenada por desempenho, o que é a razão de existir.
#
# `Similarity graph` e `Text` não estão na lista porque são filhos únicos da
# matiz deles; quem faltar cai no fim, com o tom mais claro.
STRENGTH = ["Felzenszwalb", "SLIC", "Descriptors", "L(G_felz)", "Line graph",
            "Gabor", "Hilbert", "Random walk", "Fourier", "HVG"]
# Passos por tamanho de família, também escolhidos pela auditoria. Espaçar mais
# NÃO ajuda: clarear demais aproxima todas as matizes do branco e escurecer
# aproxima do preto, então alargar a rampa melhora o contraste dentro da família
# e estraga o de fora. Medido: rampa 0..0,72 derruba o pior par entre famílias
# de 13,3 para 4,2.
_STEPS = {1: [0.0], 2: [0.0, 0.24], 3: [0.0, 0.31, 0.62],
          4: [0.0, 0.24, 0.44, 0.62]}

PALETTE = {}
for _family, _members in FAMILIES.items():
    _ranked = sorted(_members, key=lambda v: (STRENGTH.index(v)
                                              if v in STRENGTH else len(STRENGTH)))
    for _rank, _view in enumerate(_ranked):
        _slot = OVERRIDE_SLOT.get(_view, FAMILY_SLOT[_family])
        _steps = _STEPS[len([m for m in _ranked if m not in OVERRIDE_SLOT])]
        _step = 0.0 if _view in OVERRIDE_SLOT else _steps[min(_rank, len(_steps) - 1)]
        PALETTE[_view] = _shade(SLOTS[_slot], _step)

if len(set(PALETTE.values())) != len(PALETTE):
    raise RuntimeError("duas views com a mesma cor na paleta canônica")


def colour(name: str, default: str = "#b9b8b5") -> str:
    """Cor canônica da view. O default cinza é para blocos fora da taxonomia.

    `Caption embedding` cai no default de propósito: ele não é uma view do banco
    (marginal −0,09pp, §5.9) e não deve ganhar identidade visual de uma.
    """
    return PALETTE.get(name, default)


def family_colour(name: str) -> str:
    """A matiz cheia da família, para figuras que agrupam em vez de distinguir."""
    return SLOTS[OVERRIDE_SLOT.get(name) or FAMILY_SLOT[family_of(name)]]


# ---------------------------------------------------------------------------
# Hachura por família (1/set)
# ---------------------------------------------------------------------------
#
# Canal REDUNDANTE com a matiz, e ele existe porque a matiz sozinha não fecha.
# Seis famílias num círculo de matiz dão no máximo 60° de separação média, então
# exigir 60° entre todos os pares é impossível por construção -- e de fato quatro
# pares ficam abaixo, o pior sendo `Similarity graph x Image statistics` a 24,9°
# (dois verdes, apontado pelo usuário em 1/set). Recolorir só empurra o aperto de
# um par para outro: mover o grafo para um teal no vão de 119° sobe o mínimo para
# 33,6° e o par binding vira `Spectral x Text`.
#
# Hachura sai desse jogo de soma zero. Ela dá um segundo eixo de identidade que
# não é cor nenhuma, então também sobrevive a **impressão em cinza**, que a rampa
# de tom sozinha não garante.
#
# `Region graphs` fica SEM hachura de propósito: são quatro das dez views e
# hachurar todas deixaria a figura carregada onde ela já é clara (o azul está a
# 76-144° de todo mundo). A hachura marca justamente as famílias de uma view só,
# que é onde a cor aperta.
#
# A atribuição segue os pares mais próximos em matiz, dando padrões
# maximamente diferentes a eles: `Similarity graph` e `Image statistics` (24,9°)
# ficam com `xxx` e `|||`; `Spectral` e `Text` (33,6°) com `\\\\` e `ooo`.
#
# `|||` em vez de `///` no `Image statistics` foi decidido olhando a versão em
# CINZA: `///` e o `\\\\` do Spectral são diagonais espelhadas e, sem cor, viram
# "as duas hachuradas na diagonal". Vertical não colide com nenhuma das duas e
# também não briga com as linhas de grade, que são horizontais.
FAMILY_HATCH = {
    "Region graphs": "",
    "Image statistics": "|||",
    "Similarity graph": "xxx",
    "Space-filling curve": "...",
    "Spectral & text": "\\\\",
}
OVERRIDE_HATCH = {"Text": "ooo"}


def hatch(name: str) -> str:
    """Padrão de hachura da família da view. String vazia = barra lisa."""
    if name in OVERRIDE_HATCH:
        return OVERRIDE_HATCH[name]
    try:
        return FAMILY_HATCH[family_of(name)]
    except KeyError:
        return ""


def edge(name: str, amount: float = 0.42) -> str:
    """Versão escurecida da cor da view, para desenhar a hachura e o contorno.

    Hachura em branco some nas barras claras (o `Line SLIC` é `#aeccef`) e
    hachura em preto vira ruído nas escuras. Escurecer a própria cor mantém a
    identidade cromática da barra e dá contraste em qualquer tom da rampa.
    """
    base = [int(colour(name)[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c * (1 - amount)):02x}" for c in base)
