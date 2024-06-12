from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import os
import torch
import torch.nn as nn

class SD3TextModel(nn.Module):
    def __init__(
        self, 
        tokenizer:CLIPTokenizer, 
        tokenizer_2:CLIPTokenizer, 
        text_encoder:CLIPTextModelWithProjection, 
        text_encoder_2:CLIPTextModelWithProjection, 
        clip_skip:int=-1
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.clip_skip = clip_skip
        self.sdxl = tokenizer_2 is not None

    def tokenize(self, texts):
        tokens = self.tokenizer(
            texts, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(self.text_encoder.device)

        tokens_2 = self.tokenizer_2(
            texts, 
            max_length=self.tokenizer_2.model_max_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        ).input_ids.to(self.text_encoder_2.device)

        return tokens, tokens_2

    def get_hidden_states(self, tokens, tokens_2):
        encoder_output = self.text_encoder(tokens, output_hidden_states=True)
        last_hidden_state = encoder_output.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens == self.tokenizer.eos_token_id)[1][0].to(device=last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]
        pooled_output = self.text_encoder.text_projection(pooled_output)

        encoder_hidden_states = encoder_output.hidden_states[self.clip_skip]

        encoder_output_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
        last_hidden_state = encoder_output_2.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens_2 == self.tokenizer_2.eos_token_id)[1].to(device=last_hidden_state.device)
        pooled_output_2 = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
        pooled_output_2 = self.text_encoder_2.text_projection(pooled_output_2)

        encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

        # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

        # pad
        encoder_hidden_states = torch.cat([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=2)
        encoder_hidden_states = torch.cat([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=1) # t5

        print(pooled_output.shape, pooled_output_2.shape)
        pooled_output = torch.cat([pooled_output, pooled_output_2], dim=1)

        return encoder_hidden_states, pooled_output

    def forward(self, prompts):
        tokens, tokens_2 = self.tokenize(prompts)
        encoder_hidden_states, pooled_output = self.get_hidden_states(tokens, tokens_2)
        return encoder_hidden_states, pooled_output

    def enable_gradient_checkpointing(self, enable=True):
        if enable:
            self.text_encoder.gradient_checkpointing_enable()
            self.text_encoder_2.gradient_checkpointing_enable()
        else:
            self.text_encoder.gradient_checkpointing_disable()
            self.text_encoder_2.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(cls, path, revision=None, torch_dtype=None, clip_skip=-2):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer', revision=revision, torch_dtype=torch_dtype)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder', revision=revision, torch_dtype=torch_dtype)

        tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2', revision=revision, torch_dtype=torch_dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder_2', revision=revision, torch_dtype=torch_dtype)

        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
    
    def save_pretrained(self, save_directory):
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))

        self.text_encoder_2.save_pretrained(os.path.join(save_directory, "text_encoder_2"))
        self.tokenizer_2.save_pretrained(os.path.join(save_directory, "tokenizer_2"))
    
    def set_embedding_dtype(self, dtype):
        self.text_encoder.text_model.embeddings.to(dtype)
        self.text_encoder_2.text_model.embeddings.to(dtype)