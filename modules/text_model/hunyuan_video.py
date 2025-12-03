import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, LlamaTokenizerFast, LlamaModel
from .base import BaseTextModel

class HunyuanVideoTextModel(BaseTextModel):

    DEFAULT_PROMPT_TEMPLATE = {
        "template": (
            "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
            "4. background environment, light, style and atmosphere."
            "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        ),
        "crop_start": 95,
    }
    
    def __init__(
        self, 
        tokenizer:LlamaTokenizerFast, 
        tokenizer_2:CLIPTokenizer, 
        text_encoder:LlamaModel, 
        text_encoder_2:CLIPTextModel, 
        clip_skip:int=-3
    ):
        super().__init__()
        self.tokenizers = [tokenizer, tokenizer_2]
        self.text_encoders = nn.ModuleList([text_encoder, text_encoder_2])
        self.clip_skip = clip_skip

    def tokenize(self, texts):
        tokens = []

        texts_with_template = [self.DEFAULT_PROMPT_TEMPLATE["template"].format(text) for text in texts]
        text_inputs = self.tokenizers[0](
            texts_with_template,
            max_length=256 + self.DEFAULT_PROMPT_TEMPLATE["crop_start"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        token = text_inputs.input_ids.to(device=self.text_encoders[0].device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=self.text_encoders[0].device)
        tokens.append((token, prompt_attention_mask))

        token = self.tokenizers[1](
            texts, 
            max_length=self.tokenizers[1].model_max_length, 
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(self.text_encoders[1].device)
        tokens.append(token)
        return tokens

    def get_hidden_states(self, tokens):
        token, prompt_attention_mask = tokens[0]
        prompt_embeds = self.text_encoders[0](
            input_ids=token,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[self.clip_skip]
        prompt_embeds = prompt_embeds[:, self.DEFAULT_PROMPT_TEMPLATE["crop_start"]:]
        prompt_attention_mask = prompt_attention_mask[:, self.DEFAULT_PROMPT_TEMPLATE["crop_start"]:]
        pooled_output = self.text_encoders[1](tokens[1], output_hidden_states=False).pooler_output

        return prompt_embeds, pooled_output, prompt_attention_mask
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, variant=None, quantization_config=None):
        tokenizer = LlamaTokenizerFast.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = LlamaModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=quantization_config)

        tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2', revision=revision)
        text_encoder_2 = CLIPTextModel.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=quantization_config)

        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
    
    def prepare_fp8(self, autocast_dtype):
        self.text_encoders[1].text_model.embeddings.to(torch.float16)
        for modules in self.text_encoders[0].modules():
            if modules.__class__.__name__ in ["LlamaRMSNorm", "Embedding"]:
                modules.to(autocast_dtype)
