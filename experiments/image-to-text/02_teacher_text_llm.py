"""Etapa 02 — geração de texto com o professor (Qwen2-VL via vLLM).

Para cada imagem do image_index, o VLM gera uma descrição com ênfase no tom
emocional do que é *visível* (cor, luz, composição, expressões) — sem inventar
narrativa. Essas legendas são os alvos de destilação do estudante.

Pontos de atenção (início/fim da descrição):
  - prompt pede N frases e "termine com ponto";
  - `max_tokens` como teto duro;
  - pós-processamento corta na última sentença completa e ANEXA <eos> explícito
    (o estudante só aprende a parar se todo alvo terminar em <eos>);
  - filtrar recusas, repetições e legendas fora da faixa de comprimento.

Entrada:
    artifacts/image_index.csv      (etapa 01)

Saída:
    artifacts/teacher_captions.csv   image_id, path, split, caption, n_tokens
    artifacts/teacher_captions.jsonl mesmo conteúdo + prompt + raw (auditoria)

Uso:
    python 02_teacher_text_llm.py --limit 30000 --tp 1 --max-tokens 96
"""

from __future__ import annotations

import argparse

# from config import (TEACHER_MODEL, TEACHER_MAX_NEW_TOKENS, TEACHER_TEMPERATURE,
#                     IMAGE_INDEX_CSV, TEACHER_CAPTIONS_CSV, TEACHER_CAPTIONS_JSONL,
#                     EOS, MAX_CAPTION_TOKENS, log)

PROMPT = (
    "Describe what is visibly present in this image in two or three complete "
    "sentences. Convey the emotional tone evoked by the scene's colors, light, "
    "composition and expressions, but only from what is visible. Do not invent "
    "a story or name any style, artist or place. End with a period."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--index", type=Path, default=IMAGE_INDEX_CSV)
    # parser.add_argument("--model", default=TEACHER_MODEL)
    # parser.add_argument("--limit", type=int, default=0)
    # parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    # parser.add_argument("--max-tokens", type=int, default=TEACHER_MAX_NEW_TOKENS)
    # parser.add_argument("--temperature", type=float, default=TEACHER_TEMPERATURE)
    # parser.add_argument("--resume", action="store_true", help="pular já feitos")
    return parser.parse_args()


def postprocess(text: str) -> str:
    """Corta na última sentença completa e garante terminação com <eos>."""
    # TODO: strip; cortar após o último '.', '!' ou '?'
    # TODO: rejeitar recusas ("I cannot", "As an AI")
    # TODO: colapsar repetição; validar faixa de comprimento
    # TODO: retornar caption + EOS (ou "" se reprovada)
    raise NotImplementedError


def main() -> None:
    args = parse_args()  # noqa: F841
    # TODO: carregar image_index; (opcional) filtrar por --limit / --resume
    # TODO: subir vLLM com o VLM (LLM(model=..., limit_mm_per_prompt={"image":1}))
    # TODO: montar mensagens multimodais (imagem + PROMPT) em lote
    # TODO: gerar com SamplingParams(temperature, max_tokens, stop=[EOS/"."])
    # TODO: postprocess() em cada saída; descartar reprovadas
    # TODO: escrever teacher_captions.{csv,jsonl} (jsonl com prompt+raw)
    # TODO: log de taxa de descarte e histograma de n_tokens
    raise NotImplementedError("scaffold — etapa 02 ainda não implementada")


if __name__ == "__main__":
    main()
