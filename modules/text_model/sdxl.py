import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from .base import BaseTextModel

class SDXLTextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer:CLIPTokenizer, 
        tokenizer_2:CLIPTokenizer, 
        text_encoder:CLIPTextModel, 
        text_encoder_2:CLIPTextModelWithProjection, 
        clip_skip:int=-1
    ):
        super().__init__()
        self.tokenizers = [tokenizer, tokenizer_2]
        self.text_encoders = nn.ModuleList([text_encoder, text_encoder_2])
        self.clip_skip = clip_skip

    def get_hidden_states(self, tokens):
        encoder_hidden_states = self.text_encoders[0](tokens[0], output_hidden_states=True).hidden_states[self.clip_skip]
        encoder_output_2 = self.text_encoders[1](tokens[1], output_hidden_states=True)
        last_hidden_state = encoder_output_2.last_hidden_state

        # calculate pooled_output
        eos_token_index = torch.where(tokens[1] == self.tokenizers[1].eos_token_id)[1].to(device=last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
        pooled_output = self.text_encoders[1].text_projection(pooled_output)

        encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

        # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

        # pooled_output is zero vector for empty text            
        for i, token in enumerate(tokens[1]):
            if token[1].item() == self.tokenizers[1].eos_token_id:
                pooled_output[i] = 0

        return encoder_hidden_states, pooled_output, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-1, revision=None, torch_dtype=None, variant=None):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)
        tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2', revision=revision)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype, variant=variant)
        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
