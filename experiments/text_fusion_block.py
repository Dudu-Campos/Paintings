"""Bloco de texto aninhado, compartilhado pelos dois produtores de posterior.

O §5.18 fixa que o `build_text_scores` produz OOF SIMPLES sobre a coorte
inteira: para o fold externo k, os escores das linhas de treino vieram de
modelos que viram os rotulos de k. E o mecanismo do SNF (§6) diluido. Por isso
o bloco tem de ser refeito por fold externo, com OOF interno -- e isso e o que
`nested_text_block` faz.

Existe um modulo so porque a logica de split JA divergiu em silencio entre os
dois produtores de fusao uma vez (o comentario do
`evaluate_decision_fusion_official.py:148` registra o caso). Aqui o que e
compartilhado e o bloco, nao o split: cada produtor continua dono do seu.

O cache e por (dataset, fold) e guarda o HASH DOS STEMS junto. Sem isso, uma
coorte reordenada reusaria um bloco desalinhado em silencio, que e exatamente o
erro de -2,92pp que o §5.18b registra -- `load_captions` casa por stem e nao
reclama, entao nada na saida denuncia.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_text_scores import make_stop_words  # noqa: E402
from evaluate_text_marginal_multitask import load_captions, nested_text_block  # noqa: E402

# O `CAPTIONS` do `evaluate_text_marginal_multitask` e fixo no multitask (guarda
# Path, nao template), entao o template por corpus fica aqui -- e o mesmo do
# `evaluate_logit_add_text_multitask.py`, que ja roda os sete.
CAPTIONS = {"estudante": "results/{dataset}_student_captions.jsonl",
            "qwen": "results/{dataset}_qwen_captions.jsonl"}

# Corpora cujo TESTE e coorte separada no registry, logo com arquivo de legenda
# proprio. §7.14: no nct_crc o teste sao os 7.180 tiles de PACIENTES diferentes.
TEST_COHORT = {"nct_crc": "nct_crc_val"}


def captioned_stems(path: Path) -> set[str]:
    """Stems que TEM legenda, sem carregar o texto -- serve de guarda antes de
    `load_captions`, que falha alto por design."""
    if not path.exists():
        raise SystemExit(f"legendas ausentes: {path}")
    out: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            out.add(str(json.loads(line)["stem"]))
    return out


def stem_digest(stems: np.ndarray) -> str:
    return hashlib.sha1("\n".join(map(str, stems)).encode()).hexdigest()[:16]


def load_texts(dataset: str, stems: np.ndarray, held_stems: np.ndarray | None = None,
               captions: str = "estudante") -> np.ndarray:
    """Legendas na ORDEM DA COORTE, concatenando a coorte de teste quando ela e
    separada. Falha alto se faltar alguma (§8: quatro extratores ja gravaram
    menos imagens que o esperado sem erro nenhum)."""
    path = ROOT / CAPTIONS[captions].format(dataset=dataset)
    if not path.exists():
        raise SystemExit(f"legendas ausentes: {path}")
    texts = load_captions(path, stems)
    if held_stems is not None:
        held_path = ROOT / CAPTIONS[captions].format(dataset=TEST_COHORT[dataset])
        if not held_path.exists():
            raise SystemExit(f"legendas da coorte de teste ausentes: {held_path}")
        texts = np.concatenate([texts, load_captions(held_path, held_stems)])
    return texts


def make_text_fn(dataset: str, stems: np.ndarray, labels_str: np.ndarray,
                 classes: np.ndarray, cache_dir: Path, texts: np.ndarray):
    """Devolve `fn(fold, train_idx, test_idx) -> bloco de comprimento total`."""
    # As stop words de NOME DE ESTILO sao a defesa do §5.18 contra o captioner
    # nomear o rotulo. Lista vazia removeria a defesa em silencio.
    stop_words = list(make_stop_words(classes))
    index_of = {c: i for i, c in enumerate(classes)}
    digest = stem_digest(stems)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fn(fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        cache = cache_dir / f"fold_{fold:02d}.npz"
        if cache.exists():
            blob = np.load(cache)
            if str(blob["digest"]) == digest:
                return blob["block"].astype(np.float32)
            print(f"  [aviso] cache de texto do fold {fold} e de outra coorte; refazendo",
                  flush=True)
        block = nested_text_block(texts, labels_str, classes, index_of, stop_words,
                                  train_idx, test_idx, fold)
        with cache.open("wb") as handle:
            np.savez_compressed(handle, block=block.astype(np.float32), digest=digest)
        return block.astype(np.float32)

    return fn
