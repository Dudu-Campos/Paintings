# SLM image description (COCO 2017 / ArtEmis)

Pipeline autônomo para gerar descrições de imagens com um SLM multimodal via vLLM.
Cada servidor baixa sua própria cópia do dataset localmente (sem transacionar dados
entre hosts) e processa as imagens em batch.

## Componentes

- `download_coco.py` — baixa e configura o COCO 2017 em `--data_dir`, extrai os
  zips e escreve `manifest.csv` (`image_id, file_name, split, path`).
- `download_artemis.py` — baixa ArtEmis (labels) **+** WikiArt (imagens) do Kaggle
  via `kagglehub` e gera um `manifest.csv` mergeado. O ArtEmis só tem anotações; as
  imagens vêm do WikiArt. O merge é pelo `stem` do filename (id do ArtEmis ==
  `<stem>` da imagem WikiArt `<art_style>/<stem>.jpg`). `image_id = stem`,
  `split = art_style`.
- `vllm_service.py` — `VLLMMultimodalService`: wrapper assíncrono do `AsyncLLMEngine`
  do vLLM com suporte a imagem (`multi_modal_data`); batch que preserva a ordem +
  variante streaming com callback. Extensão multimodal do script base text-only de
  `aspect-implies-issue-validation/scripts/vllm_service.py`.
- `generate_descriptions.py` — lê o manifest (ou um diretório), gera uma descrição
  por imagem em batch e grava JSONL resumível.

## Setup

```bash
pip install -r src/slm/requirements.txt
```

## Uso

```bash
# 1. Baixar e configurar o COCO 2017 (val2017 por padrão)
python src/slm/download_coco.py --data_dir data/coco --splits val2017

# 2. Gerar descrições com o SLM (unsloth/Qwen2.5-VL-7B-Instruct por padrão)
python src/slm/generate_descriptions.py \
    --manifest data/coco/manifest.csv \
    --split val2017 \
    --batch_size 8

# Variante 4-bit da Unsloth (menos VRAM)
python src/slm/generate_descriptions.py \
    --manifest data/coco/manifest.csv --split val2017 \
    --model unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit \
    --quantization bitsandbytes

# Teste rápido em poucas imagens
python src/slm/generate_descriptions.py \
    --manifest data/coco/manifest.csv --split val2017 --limit 16
```

### ArtEmis + WikiArt (Kaggle)

```bash
# 1. Baixar ArtEmis (labels) + WikiArt (imagens) e gerar o manifest mergeado
#    (requer credenciais Kaggle em ~/.kaggle/kaggle.json)
python src/slm/download_artemis.py --data_dir data/artemis

# 2. Gerar descrições com output próprio e rastreável (distinto do COCO)
python src/slm/generate_descriptions.py \
    --manifest data/artemis/manifest.csv \
    --output outputs/slm_descriptions/artemis_descriptions.jsonl
```

Saída COCO:    `outputs/slm_descriptions/coco_descriptions.jsonl`
Saída ArtEmis: `outputs/slm_descriptions/artemis_descriptions.jsonl`
(`{"image_id", "file_name", "split", "description", "model"}`).

Modelo trocável via `--model` (qualquer VLM suportado pelo vLLM, ex.:
`HuggingFaceTB/SmolVLM-Instruct`, `Qwen/Qwen2-VL-2B-Instruct`). Para os repos
quantizados `-bnb-4bit` da Unsloth, passe `--quantization bitsandbytes`.
