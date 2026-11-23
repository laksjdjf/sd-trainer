import torch.nn as nn
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from .base import BaseTextModel, BaseTextOutput


class QwenImage21TextModel(BaseTextModel):
    """Use the official prompt template and pre-final-norm encoder output."""

    def __init__(self, processor, text_encoder):
        super().__init__()
        self.processor = processor
        self.tokenizers = [processor.tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self._prompt_pipeline = QwenImage21Pipeline(
            scheduler=None, vae=None, text_encoder=text_encoder,
            processor=processor, transformer=None,
        )

    def forward(self, prompts):
        embeddings, mask, image_mask = self._prompt_pipeline._get_qwen_prompt_embeds(
            prompts, device=self.device,
        )
        if image_mask.any():
            raise ValueError("Image tokens in captions are unsupported in text-to-image training.")
        return BaseTextOutput(embeddings, attention_mask=mask.bool())

    @classmethod
    def from_pretrained(cls, path, revision=None, torch_dtype=None, variant=None):
        processor = Qwen3VLProcessor.from_pretrained(path, subfolder="processor", revision=revision)
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            path, subfolder="text_encoder", revision=revision, torch_dtype=torch_dtype,
            **({"variant": variant} if variant else {}),
        )
        return cls(processor, encoder)
