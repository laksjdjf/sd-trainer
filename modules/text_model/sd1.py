import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from .base import BaseTextModel

class SD1TextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer:CLIPTokenizer, 
        text_encoder:CLIPTextModel, 
        clip_skip:int=-1
    ):
        super().__init__()
        self.tokenizers = [tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self.clip_skip = clip_skip

    def get_hidden_states(self, tokens):
        encoder_hidden_states = self.text_encoders[0](tokens[0], output_hidden_states=True).hidden_states[self.clip_skip]
        encoder_hidden_states = self.text_encoders[0].text_model.final_layer_norm(encoder_hidden_states)
        return encoder_hidden_states, None, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-1, revision=None, torch_dtype=None, variant=None):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)
        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
