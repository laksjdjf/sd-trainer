import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from .base import BaseTextModel

class FluxTextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer:CLIPTokenizer, 
        tokenizer_2:T5TokenizerFast, 
        text_encoder:CLIPTextModel, 
        text_encoder_2:T5EncoderModel, 
        clip_skip:int=-1
    ):
        super().__init__()
        self.tokenizers = [tokenizer, tokenizer_2]
        self.text_encoders = nn.ModuleList([text_encoder, text_encoder_2])
        self.clip_skip = clip_skip

    def get_hidden_states(self, tokens):
        pooled_output = self.text_encoders[0](tokens[0], output_hidden_states=False).pooler_output
        encoder_hidden_states = self.text_encoders[1](tokens[1], output_hidden_states=False)[0]

        return encoder_hidden_states, pooled_output, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, variant=None, max_length=128):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)

        tokenizer_2 = T5TokenizerFast.from_pretrained(path, subfolder='tokenizer_2', revision=revision)
        text_encoder_2 = T5EncoderModel.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype, variant=variant)
        tokenizer_2.model_max_length = max_length

        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
    
    def prepare_fp8(self, autocast_dtype):
        for text_encoder in self.text_encoders:
            if hasattr(text_encoder, 'text_model'):
                text_encoder.text_model.embeddings.to(autocast_dtype)

        def forward_hook(modules):
            def forward(hidden_states):
                hidden_gelu = modules.act(modules.wi_0(hidden_states))
                hidden_linear = modules.wi_1(hidden_states)
                hidden_states = hidden_gelu * hidden_linear
                hidden_states = modules.dropout(hidden_states)

                hidden_states = modules.wo(hidden_states)
                return hidden_states
            return forward

        for modules in self.text_encoders[1].modules():
            if modules.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                modules.to(autocast_dtype)
            if modules.__class__.__name__ in ["T5DenseGatedActDense"]:
                modules.forward = forward_hook(modules) 
