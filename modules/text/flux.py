import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, Qwen2Tokenizer, Qwen3ForCausalLM
from .base import BaseTextModel


class FluxTextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer: CLIPTokenizer, 
        tokenizer_2: T5TokenizerFast, 
        text_encoder: CLIPTextModel, 
        text_encoder_2: T5EncoderModel, 
        clip_skip: int = -1
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


class Flux2KleinTextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer: Qwen2Tokenizer, 
        text_encoder: Qwen3ForCausalLM,
        clip_skip: int = -1
    ):
        super().__init__()
        self.tokenizers = [tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self.clip_skip = clip_skip

    def tokenize(self, texts):
        all_input_ids = []
        all_attention_masks = []

        for single_prompt in texts:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizers[0].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizers[0](
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.text_encoders[0].device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.text_encoders[0].device)

        text_inputs = {
            "input_ids": input_ids.to(self.text_encoders[0].device),
            "attention_mask": attention_mask.to(self.text_encoders[0].device),
        }
        return [text_inputs]

    def get_hidden_states(self, tokens):
        text_output = self.text_encoders[0](
            tokens[0]["input_ids"],
            attention_mask=tokens[0]["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        out = torch.stack([text_output.hidden_states[k] for k in (9, 18, 27)], dim=1)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds, None, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, variant=None, max_length=256):
        tokenizer = Qwen2Tokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision)
        text_encoder = Qwen3ForCausalLM.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype, variant=variant)

        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
