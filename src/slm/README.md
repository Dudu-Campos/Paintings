# Qwen image descriptions for paintings

Pipeline autônomo para gerar descrições de pinturas com um SLM multimodal via
vLLM. As descrições do Qwen são um controle textual/importado para a discussão
do artigo; elas não fazem parte do método visual principal.

## Componentes

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

## Uso com ArtEmis + WikiArt

```bash
# 1. Baixar ArtEmis (labels) + WikiArt (imagens) e gerar o manifest mergeado
#    (requer credenciais Kaggle em ~/.kaggle/kaggle.json)
python src/slm/download_artemis.py --data_dir data/artemis

# 2. Gerar descrições Qwen, retomáveis e rastreáveis
python src/slm/generate_descriptions.py \
    --manifest data/artemis/manifest.csv \
    --output outputs/slm_descriptions/artemis_descriptions.jsonl
```

Saída: `outputs/slm_descriptions/artemis_descriptions.jsonl`
(`{"image_id", "file_name", "split", "description", "model"}`).

O modelo é trocável via `--model` (qualquer VLM suportado pelo vLLM, ex.:
`HuggingFaceTB/SmolVLM-Instruct`, `Qwen/Qwen2-VL-2B-Instruct`). Para os repos
quantizados `-bnb-4bit` da Unsloth, passe `--quantization bitsandbytes`.
