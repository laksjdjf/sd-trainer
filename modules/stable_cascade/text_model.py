from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import os
import torch
import torch.nn as nn

class CascadeTextModel(nn.Module):
    def __init__(
        self, 
        tokenizer:CLIPTokenizer, 
        text_encoder:CLIPTextModelWithProjection, 
        clip_skip:int=-1
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.clip_skip = clip_skip

    def tokenize(self, texts):
        tokens = self.tokenizer(
            texts, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(self.text_encoder.device)

        return tokens

    def get_hidden_states(self, tokens):
        encoder_output = self.text_encoder(tokens, output_hidden_states=True)

        pooled_output = encoder_output.text_embeds.unsqueeze(1)
        encoder_hidden_states = encoder_output.hidden_states[self.clip_skip]

        return encoder_hidden_states, pooled_output

    def forward(self, prompts):
        tokens = self.tokenize(prompts)
        encoder_hidden_states, pooled_output = self.get_hidden_states(tokens)
        return encoder_hidden_states, pooled_output

    def enable_gradient_checkpointing(self, enable=True):
        if enable:
            self.text_encoder.gradient_checkpointing_enable()
        else:
            self.text_encoder.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(cls, path, clip_skip=-1):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer')
        text_encoder = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder')

        return cls(tokenizer, text_encoder, clip_skip=clip_skip)
    
    def save_pretrained(self, save_directory):
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
    
    def set_embedding_dtype(self, dtype):
        self.text_encoder.text_model.embeddings.to(dtype)