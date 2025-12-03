import torch.nn as nn
from transformers import Qwen3Model, Qwen2Tokenizer
from .base import BaseTextModel

class HDMTextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer:Qwen2Tokenizer, 
        text_encoder:Qwen3Model,
        clip_skip:int=-1
    ):
        super().__init__()
        self.tokenizers = [tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self.clip_skip = clip_skip

    def tokenize(self, texts):
        if all(len(text) == 0 for text in texts):
            texts = [self.tokenizers[0].pad_token] * len(texts)
        text_inputs = self.tokenizers[0](
            texts, 
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        text_inputs = {
            "input_ids": text_inputs.input_ids.to(self.text_encoders[0].device),
            "attention_mask": text_inputs.attention_mask.to(self.text_encoders[0].device),
        }
        return [text_inputs]

    def get_hidden_states(self, tokens):
        text_output = self.text_encoders[0](tokens[0]["input_ids"], attention_mask=tokens[0]["attention_mask"])
        encoder_hidden_states = text_output.last_hidden_state
        return encoder_hidden_states, None, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, variant=None, max_length=256):
        tokenizer = Qwen2Tokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = Qwen3Model.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)
        tokenizer.model_max_length = max_length

        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
