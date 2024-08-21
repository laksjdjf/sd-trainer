from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast
import os
import torch
import torch.nn as nn

class BaseTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = None
        self.negative_prompt = None

    def tokenize(self, texts):
        tokens = []
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            token = tokenizer(
                texts, 
                max_length=tokenizer.model_max_length, 
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(text_encoder.device)
            tokens.append(token)
        return tokens
    
    def get_hidden_states(self, tokens):
        raise NotImplementedError
    
    def forward(self, prompts):
        tokens = self.tokenize(prompts)
        encoder_hidden_states, pooled_output = self.get_hidden_states(tokens)
        return encoder_hidden_states, pooled_output
    
    def enable_gradient_checkpointing(self, enable=True):
        for text_encoder in self.text_encoders:
            if enable:
                text_encoder.gradient_checkpointing_enable()
            else:
                text_encoder.gradient_checkpointing_disable()
    
    def save_pretrained(self, save_directory):
        for i, (text_encoder, tokenizer) in enumerate(zip(self.text_encoders, self.tokenizers)):
            surfix = "" if i == 0 else f"_{i+1}"
            text_encoder.save_pretrained(os.path.join(save_directory, f"text_encoder{surfix}"))
            tokenizer.save_pretrained(os.path.join(save_directory, f"tokenizer{surfix}"))
    
    def prepare_fp8(self, autocast_dtype):
        for text_encoder in self.text_encoders:
            if hasattr(text_encoder, 'text_model'):
                text_encoder.text_model.embeddings.to(autocast_dtype)
    
    @torch.no_grad()
    def cache_uncond(self):
        uncond_hidden_state, uncond_pooled_output = self([""])
        self.uncond_hidden_state = uncond_hidden_state.detach().float().cpu()
        self.uncond_pooled_output = uncond_pooled_output.detach().float().cpu() if uncond_pooled_output is not None else None
    
    @torch.no_grad()
    def cache_sample(self, prompt, negative_prompt):
        encoder_hidden_states, pooled_output = self([prompt])
        self.encoder_hidden_states = encoder_hidden_states.detach().float().cpu()
        self.pooled_output = pooled_output.detach().float().cpu() if pooled_output is not None else None
        
        encoder_hidden_states, pooled_output = self([negative_prompt])
        self.negative_encoder_hidden_states = encoder_hidden_states.detach().float().cpu()
        self.negative_pooled_output = pooled_output.detach().float().cpu() if pooled_output is not None else None
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    @classmethod
    def from_pretrained(cls, path, revision=None, torch_dtype=None):
        raise NotImplementedError
    
    @property
    def device(self):
        return self.text_encoders[0].device

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

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
        return encoder_hidden_states, None
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-1, revision=None, torch_dtype=None):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision, torch_dtype=torch_dtype)
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype)
        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
    
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

        return encoder_hidden_states, pooled_output
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-1, revision=None, torch_dtype=None):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision, torch_dtype=torch_dtype)
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype)
        tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2', revision=revision, torch_dtype=torch_dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype)
        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)

class SD3TextModel(BaseTextModel):
    def __init__(
        self, 
        tokenizer:CLIPTokenizer, 
        tokenizer_2:CLIPTokenizer, 
        text_encoder:CLIPTextModelWithProjection, 
        text_encoder_2:CLIPTextModelWithProjection, 
        clip_skip:int=-1
    ):
        super().__init__()
        self.tokenizers = [tokenizer, tokenizer_2]
        self.text_encoders = nn.ModuleList([text_encoder, text_encoder_2])
        self.clip_skip = clip_skip

    def get_hidden_states(self, tokens):
        encoder_output = self.text_encoders[0](tokens[0], output_hidden_states=True)
        last_hidden_state = encoder_output.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens[0] == self.tokenizers[0].eos_token_id)[1][0].to(device=last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]
        pooled_output = self.text_encoders[0].text_projection(pooled_output)

        encoder_hidden_states = encoder_output.hidden_states[self.clip_skip]

        encoder_output_2 = self.text_encoders[1](tokens[1], output_hidden_states=True)
        last_hidden_state = encoder_output_2.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens[1] == self.tokenizers[1].eos_token_id)[1].to(device=last_hidden_state.device)
        pooled_output_2 = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
        pooled_output_2 = self.text_encoders[1].text_projection(pooled_output_2)

        encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

        # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

        # pad
        encoder_hidden_states = torch.cat([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=2)
        encoder_hidden_states = torch.cat([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=1) # t5

        pooled_output = torch.cat([pooled_output, pooled_output_2], dim=1)

        return encoder_hidden_states, pooled_output

    @classmethod
    def from_pretrained(cls, path, clip_skip=-1, revision=None, torch_dtype=None):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision, torch_dtype=torch_dtype)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype)
        tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2', revision=revision, torch_dtype=torch_dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype)
        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
    
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

        return encoder_hidden_states, pooled_output
    
    @classmethod
    def from_pretrained(cls, path, clip_skip=-2, revision=None, torch_dtype=None, max_length=128):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision, torch_dtype=torch_dtype)
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype)

        tokenizer_2 = T5TokenizerFast.from_pretrained(path, subfolder='tokenizer_2', revision=revision, torch_dtype=torch_dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype)
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