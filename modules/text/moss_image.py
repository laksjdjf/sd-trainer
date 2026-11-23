import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .base import BaseTextModel


class MOSSImageTextModel(BaseTextModel):
    """Qwen3.5 final hidden states, evaluated without padding as upstream does."""

    def __init__(self, tokenizer, text_encoder, max_length=4096):
        super().__init__()
        self.tokenizers = [tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self.max_length = max_length

    def tokenize(self, texts):
        tokens = []
        for text in texts:
            prompt = self.tokenizers[0].apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            tokens.append(self.tokenizers[0](
                prompt, return_tensors="pt", truncation=True, max_length=self.max_length,
            ).to(self.device))
        return tokens

    def get_hidden_states(self, tokens):
        rows = [self.text_encoders[0](**token, use_cache=False, return_dict=True).last_hidden_state[0]
                for token in tokens]
        length = max(row.shape[0] for row in rows)
        embeddings = torch.stack([torch.nn.functional.pad(row, (0, 0, 0, length - row.shape[0])) for row in rows])
        mask = torch.arange(length, device=embeddings.device)[None] < torch.tensor(
            [row.shape[0] for row in rows], device=embeddings.device,
        )[:, None]
        return embeddings, None, mask

    @classmethod
    def from_pretrained(cls, path, revision=None, torch_dtype=None, variant=None, max_length=4096):
        tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer", revision=revision)
        encoder = AutoModel.from_pretrained(
            path, subfolder="text_encoder", revision=revision, torch_dtype=torch_dtype, variant=variant,
        )
        return cls(tokenizer, encoder, max_length)
