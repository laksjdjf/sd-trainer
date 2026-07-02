import torch.nn as nn
from transformers import GemmaTokenizer, Gemma2Model
from .base import BaseTextModel


class Lumina2TextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer: GemmaTokenizer, 
        text_encoder: Gemma2Model,
        clip_skip: int = -1
    ):
        super().__init__()
        self.tokenizers = [tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self.clip_skip = clip_skip

    def tokenize(self, texts):
        text_inputs = self.tokenizers[0](
            texts, 
            max_length=self.tokenizers[0].model_max_length, 
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
        text_output = self.text_encoders[0](tokens[0]["input_ids"], attention_mask=tokens[0]["attention_mask"], output_hidden_states=True)
        encoder_hidden_states = text_output.hidden_states[self.clip_skip]

        return encoder_hidden_states, None, tokens[0]["attention_mask"]
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, variant=None, max_length=256):
        tokenizer = GemmaTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = Gemma2Model.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)
        tokenizer.model_max_length = max_length

        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
