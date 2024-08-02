from transformers import CLIPTokenizer, CLIPTextModel, T5EncoderModel, T5TokenizerFast
import os
import torch
import torch.nn as nn
from modules.text_model import TextModel

class FluxTextModel(TextModel):
    def get_hidden_states(self, tokens, tokens_2):
        pooled_output = self.text_encoder(tokens, output_hidden_states=False).pooler_output
        encoder_hidden_states = self.text_encoder_2(tokens_2, output_hidden_states=False)[0]

        return encoder_hidden_states, pooled_output

    def forward(self, prompts):
        tokens, tokens_2 = self.tokenize(prompts)
        encoder_hidden_states, pooled_output = self.get_hidden_states(tokens, tokens_2)
        return encoder_hidden_states, pooled_output
    
    @classmethod
    def from_pretrained(cls, path, revision=None, torch_dtype=None, clip_skip=-2):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision, torch_dtype=torch_dtype)
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype)

        tokenizer_2 = T5TokenizerFast.from_pretrained(path, subfolder='tokenizer_2', revision=revision, torch_dtype=torch_dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype)

        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)