import torch.nn as nn
from transformers import Qwen2Tokenizer, Qwen3Model
from .base import BaseTextModel


class ZImageTextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer: Qwen2Tokenizer, 
        text_encoder: Qwen3Model,
        clip_skip: int = -1
    ):
        super().__init__()
        self.tokenizers = [tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self.clip_skip = clip_skip

    def tokenize(self, texts):
        prompts = []
        for i, prompt_item in enumerate(texts):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizers[0].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(prompt_item)

        text_inputs = self.tokenizers[0](
            prompts, 
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        text_inputs = {
            "input_ids": text_inputs.input_ids.to(self.text_encoders[0].device),
            "attention_mask": text_inputs.attention_mask.to(self.text_encoders[0].device).bool(),
        }
        return [text_inputs]

    def get_hidden_states(self, tokens):
        text_output = self.text_encoders[0](
            tokens[0]["input_ids"],
            attention_mask=tokens[0]["attention_mask"],
            output_hidden_states=True,
        )
        encoder_hidden_states = text_output.hidden_states[self.clip_skip]
        
        embeddings_list = []
        for i in range(len(encoder_hidden_states)):
            embeddings_list.append(encoder_hidden_states[i][tokens[0]["attention_mask"][i]])
        return embeddings_list, None, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, variant=None, max_length=256):
        tokenizer = Qwen2Tokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = Qwen3Model.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)
        tokenizer.model_max_length = max_length

        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
