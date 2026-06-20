"""
Serviço vLLM multimodal para geração em batch.

Extensão do script base text-only (aspect-implies-issue-validation/scripts/vllm_service.py):
mantém o mesmo padrão assíncrono (generate_batch preserva a ordem; a variante
streaming chama callback assim que cada item fica pronto), mas aceita pares
(texto, imagem) e injeta a imagem via multi_modal_data no AsyncLLMEngine.

O prompt final é montado com o chat template do AutoProcessor do próprio modelo.
"""
import asyncio
import uuid
from dataclasses import fields
from typing import Callable

from PIL import Image
from transformers import AutoProcessor
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


class VLLMMultimodalService:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 8,
        temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.temperature = temperature
        self.engine = None
        self.processor = None

    async def init(self):
        """Inicializa o engine assíncrono do vLLM e o processor do modelo."""
        engine_kwargs = dict(
            model=self.model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
            limit_mm_per_prompt={'image': 1},
            disable_log_stats=False,
        )
        valid = {f.name for f in fields(AsyncEngineArgs)}
        if 'disable_log_requests' in valid:
            engine_kwargs['disable_log_requests'] = True
        elif 'enable_log_requests' in valid:
            engine_kwargs['enable_log_requests'] = False
        engine_args = AsyncEngineArgs(**engine_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _build_prompt(self, text: str, image: Image.Image) -> dict:
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': text},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {'prompt': prompt, 'multi_modal_data': {'image': image}}

    async def _generate_one(self, text, image, request_id, params):
        prompt = self._build_prompt(text, image)
        generator = self.engine.generate(prompt, params, request_id)

        final_output = None
        async for output in generator:
            final_output = output

        return final_output.outputs[0].text

    def _sampling_params(self):
        return SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature)

    async def generate_batch(self, texts: list[str], images: list[Image.Image]) -> list[str]:
        """Recebe textos + imagens e retorna outputs na MESMA ordem dos inputs."""
        if self.engine is None:
            raise RuntimeError('Você precisa chamar await service.init() primeiro.')

        params = self._sampling_params()
        tasks = [
            asyncio.create_task(self._generate_one(text, image, str(uuid.uuid4()), params))
            for text, image in zip(texts, images)
        ]
        return await asyncio.gather(*tasks)

    async def generate_streaming_batch(
        self,
        texts: list[str],
        images: list[Image.Image],
        index: list,
        callback: Callable[[object, str], None],
    ):
        """Processa cada item do batch e chama callback(idx, output) assim que pronto."""
        if self.engine is None:
            raise RuntimeError('Você precisa chamar await service.init() primeiro.')

        params = self._sampling_params()

        async def wrap(idx, text, image):
            output = await self._generate_one(text, image, str(uuid.uuid4()), params)
            callback(idx, output)

        tasks = [
            asyncio.create_task(wrap(idx, text, image))
            for idx, text, image in zip(index, texts, images)
        ]
        await asyncio.gather(*tasks)
