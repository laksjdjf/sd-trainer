import os
import torch
import torch.nn as nn

class BaseTextOutput:
    def __init__(self, encoder_hidden_states, pooled_output=None, attention_mask=None):
        self.encoder_hidden_states = encoder_hidden_states
        self.pooled_output = pooled_output
        self.attention_mask = attention_mask

    def to(self, device=None, dtype=None):
        self.encoder_hidden_states = self.encoder_hidden_states.to(device=device, dtype=dtype)
        if self.pooled_output is not None:
            self.pooled_output = self.pooled_output.to(device=device, dtype=dtype)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device=device)
        return self
    
    def __getitem__(self, key):
        encoder_hidden_states = self.encoder_hidden_states[key]
        pooled_output = None if self.pooled_output is None else self.pooled_output[key]
        attention_mask = None if self.attention_mask is None else self.attention_mask[key]

        return BaseTextOutput(encoder_hidden_states, pooled_output, attention_mask)
    
    def __len__(self):
        return self.encoder_hidden_states.size(0)

    def clone(self):
        encoder_hidden_states = self.encoder_hidden_states.clone()
        pooled_output = None if self.pooled_output is None else self.pooled_output.clone()
        attention_mask = None if self.attention_mask is None else self.attention_mask.clone()

        return BaseTextOutput(encoder_hidden_states, pooled_output, attention_mask)
    
    def detach(self):
        encoder_hidden_states = self.encoder_hidden_states.detach()
        pooled_output = None if self.pooled_output is None else self.pooled_output.detach()
        attention_mask = None if self.attention_mask is None else self.attention_mask.detach()

        return BaseTextOutput(encoder_hidden_states, pooled_output, attention_mask)
    
    def repeat(self, n):
        encoder_hidden_states = self.encoder_hidden_states.repeat(n, 1, 1)
        pooled_output = None if self.pooled_output is None else self.pooled_output.repeat(n, 1).clone()
        attention_mask = None if self.attention_mask is None else self.attention_mask.repeat(n, 1)

        return BaseTextOutput(encoder_hidden_states, pooled_output, attention_mask)
    
    @classmethod
    def cat(cls, outputs):
        encoder_hidden_states = torch.cat([output.encoder_hidden_states for output in outputs], dim=0)
        if outputs[0].pooled_output is None:
            pooled_output = None
        else:
            pooled_output = torch.cat([output.pooled_output for output in outputs], dim=0)
        if outputs[0].attention_mask is None:
            attention_mask = None
        else:
            attention_mask = torch.cat([output.attention_mask for output in outputs], dim=0)
        
        return cls(encoder_hidden_states, pooled_output, attention_mask)

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
        encoder_hidden_states, pooled_output, attention_mask = self.get_hidden_states(tokens)
        return BaseTextOutput(encoder_hidden_states, pooled_output, attention_mask)
    
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
        self.uncond_output = self([""])
        self.uncond_output = self.uncond_output.detach().to("cpu", torch.float32)
    
    @torch.no_grad()
    def cache_sample(self, prompt, negative_prompt):
        if prompt.split(".")[-1] == "txt":
            with open(prompt, "r") as f:
                prompt = f.read()
                
        if negative_prompt.split(".")[-1] == "txt":
            with open(negative_prompt, "r") as f:
                negative_prompt = f.read()

        self.positive_output = self([prompt])
        self.positive_output.detach().to("cpu", torch.float32)
        
        self.negative_output = self([negative_prompt])
        self.negative_output.detach().to("cpu", torch.float32)

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
