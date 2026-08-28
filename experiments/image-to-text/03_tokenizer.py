"""Etapa 03 — tokenizer.

Treina (ou serializa) o vocabulário que o estudante usa. Um BPE ajustado sobre
as legendas do professor mantém o vocabulário pequeno e alinhado ao domínio das
descrições afetivas. Alternativa: reusar um tokenizer pronto (GPT-2) — daí esta
etapa só registra os IDs dos tokens especiais e valida cobertura.

Nota: pode ser dispensável se optar por um tokenizer pré-treinado; mantida como
etapa explícita da pipeline para deixar a dependência visível (o estudante NÃO
pode ser treinado antes que <bos>/<eos>/<pad>/<img> estejam fixados).

Entrada:
    artifacts/teacher_captions.csv      (etapa 02)

Saída:
    artifacts/tokenizer/                vocab + merges + special_tokens_map
        (com <pad>=0, <bos>, <eos>, <img> em índices estáveis)

Uso:
    python 03_tokenizer.py --vocab-size 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BOS,
    EOS,
    IMG,
    PAD,
    SPECIAL_TOKENS,
    TEACHER_CAPTIONS_CSV,
    TOKENIZER_DIR,
    VOCAB_SIZE,
    log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, default=TEACHER_CAPTIONS_CSV)
    parser.add_argument("--out", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="descarta merges muito raros (ruído no corpus)")
    return parser.parse_args()


def strip_eos(text: str) -> str:
    """As legendas já vêm com ' <eos>'; o tokenizer aprende o texto puro e
    o <eos> entra como token especial (não como sequência de bytes)."""
    return str(text).replace(EOS, "").strip()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    captions = pd.read_csv(args.captions)
    corpus = [strip_eos(c) for c in captions["caption"] if str(c).strip()]
    log(f"Treinando BPE em {len(corpus)} legendas "
        f"(vocab-alvo={args.vocab_size}, min_freq={args.min_frequency})")

    # SPECIAL_TOKENS = [PAD, BOS, EOS, IMG] -> índices 0,1,2,3 estáveis.
    # Passar a lista ao trainer reserva esses IDs no início do vocabulário.
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.save_model(str(args.out))                 # vocab.json + merges.txt
    tokenizer.save(str(args.out / "tokenizer.json"))    # arquivo único portátil

    special_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
    (args.out / "special_tokens_map.json").write_text(
        json.dumps({
            "pad_token": PAD, "bos_token": BOS,
            "eos_token": EOS, "img_token": IMG,
            "ids": special_ids,
        }, indent=2))

    final_vocab = tokenizer.get_vocab_size()
    log(f"Salvo tokenizer em {args.out}")
    log(f"Vocabulário final: {final_vocab} tokens")
    log(f"IDs especiais: {special_ids}")
    assert special_ids[PAD] == 0, f"<pad> deveria ser 0, é {special_ids[PAD]}"

    # sanity — round-trip encode/decode
    sample = corpus[0]
    enc = tokenizer.encode(sample)
    dec = tokenizer.decode(enc.ids)
    log(f"round-trip OK={dec.strip() == sample.strip()} "
        f"| {len(enc.ids)} tokens p/ {len(sample.split())} palavras")
    lengths = [len(tokenizer.encode(c).ids) for c in corpus]
    log(f"tokens por legenda: min={min(lengths)} "
        f"mediana={sorted(lengths)[len(lengths) // 2]} max={max(lengths)}")


if __name__ == "__main__":
    main()
