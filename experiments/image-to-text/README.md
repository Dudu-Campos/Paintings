# image-to-text — destilação Qwen → decoder compacto (afeto)

Pipeline de *knowledge distillation* para captioning com ênfase em sentimento.
Um VLM professor (Qwen2-VL) descreve imagens do **COCO** com foco no tom
emocional do que é visível; um decoder-only compacto (não-LLM, treinado do
zero) aprende a reproduzir essas descrições. Como o estudante só vê COCO
(fotos, zero conteúdo de história da arte), ele funciona como um **controle
limpo com sabor de afeto** quando aplicado ao WikiArt: carrega vocabulário
emocional sem poder vazar rótulo de estilo/artista.

## Fluxo de dados

```
                 (COCO train2017)
                       │
   01_prepare.py  ─────┤  varre imagens + splits, resolve caminhos
                       ▼
              image_index.csv ............. tabela mestre (1 linha por imagem)
                       │
   02_teacher_text_llm.py  ── Qwen2-VL via vLLM, prompt afetivo, parada por EOS
                       ▼
              teacher_captions.{csv,jsonl} .. imagem → legenda do professor
                       │
   03_tokenizer.py  ── treina/serializa BPE sobre as legendas do professor
                       ▼
              tokenizer/ (vocab + merges) ... <bos> <eos> <pad> <img>
                       │
   04_student.py  ──── treina decoder-only (prefixo visual), loga loss + ROUGE
                       ▼
              student_ckpt/ + train_log.csv .. checkpoint + curva de loss
                       │
   05_evaluate.py  ─── métricas no teste COCO + comparação vs CATR no WikiArt
                       ▼
              eval_report.{csv,md}
```

Cada etapa lê o(s) artefato(s) da anterior e escreve o seu — o `image_index.csv`
é a referência consultada por todas. Constantes compartilhadas (dimensões,
tokens especiais, caminhos) ficam em `config.py`.

## Ordem de execução

```bash
python experiments/image-to-text/01_prepare.py
python experiments/image-to-text/02_teacher_text_llm.py --limit 30000
python experiments/image-to-text/03_tokenizer.py
python experiments/image-to-text/04_student.py
python experiments/image-to-text/05_evaluate.py
```

## Decisões de projeto em aberto (ver docstrings de cada arquivo)

- `512` = `d_model` **e** `max_seq_len` do estudante (proposto).
- Encoder de imagem do estudante: CLIP ViT-B/32 congelado vs. features
  ResNet/DINOv2 já extraídas no projeto.
- Escopo da geração: subset (~30k, piloto) vs. COCO inteiro (~118k).
- Parada: teto de tokens + `<eos>` explícito no professor; máscara de `<pad>`
  na loss; `no_repeat_ngram`/`repetition_penalty` na inferência do estudante.

> Esboço inicial: os scripts são scaffolds (contrato de I/O + TODOs), sem
> implementação ainda.
