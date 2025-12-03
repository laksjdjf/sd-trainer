import torch.nn as nn
from transformers import LlamaTokenizerFast, UMT5EncoderModel
from .base import BaseTextModel

class AuraFlowTextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer:LlamaTokenizerFast, 
        text_encoder:UMT5EncoderModel,
        clip_skip:int=-1
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

        text_inputs = {k: v.to(self.text_encoders[0].device) for k, v in text_inputs.items()}
        return [text_inputs]

    def get_hidden_states(self, tokens):
        encoder_hidden_states = self.text_encoders[0](**tokens[0])[0]
        attention_mask = tokens[0]["attention_mask"].unsqueeze(-1).expand(encoder_hidden_states.shape)
        encoder_hidden_states = encoder_hidden_states * attention_mask
        return encoder_hidden_states, None, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, variant=None, max_length=128):
        tokenizer = LlamaTokenizerFast.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = UMT5EncoderModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)
        tokenizer.model_max_length = max_length

        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
