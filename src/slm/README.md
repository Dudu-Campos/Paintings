# SLM image description (COCO 2017)

Pipeline autônomo para gerar descrições de imagens com um SLM multimodal via vLLM.
Cada servidor baixa sua própria cópia do COCO localmente (sem transacionar dados
entre hosts) e processa as imagens em batch.

## Componentes

- `download_coco.py` — baixa e configura o COCO 2017 em `--data_dir`, extrai os
  zips e escreve `manifest.csv` (`image_id, file_name, split, path`).
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

# 2. Gerar descrições com o SLM (Qwen2.5-VL-3B por padrão)
python src/slm/generate_descriptions.py \
    --manifest data/coco/manifest.csv \
    --split val2017 \
    --batch_size 8

# Teste rápido em poucas imagens
python src/slm/generate_descriptions.py \
    --manifest data/coco/manifest.csv --split val2017 --limit 16
```

Saída: `outputs/slm_descriptions/coco_descriptions.jsonl`
(`{"image_id", "file_name", "split", "description", "model"}`).

Modelo trocável via `--model` (qualquer VLM suportado pelo vLLM, ex.:
`HuggingFaceTB/SmolVLM-Instruct`, `Qwen/Qwen2-VL-2B-Instruct`).
