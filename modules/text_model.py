from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import os
import torch
import torch.nn as nn

class TextModel(nn.Module):
    def __init__(
        self, 
        tokenizer:CLIPTokenizer, 
        tokenizer_2:CLIPTokenizer, 
        text_encoder:CLIPTextModel, 
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

        if self.sdxl:
            tokens_2 = self.tokenizer_2(
                texts, 
                max_length=self.tokenizer_2.model_max_length, 
                padding="max_length",
                truncation=True, 
                return_tensors='pt'
            ).input_ids.to(self.text_encoder_2.device)
        else:
            tokens_2 = None

        return tokens, tokens_2

    def get_hidden_states(self, tokens, tokens_2=None):
        encoder_hidden_states = self.text_encoder(tokens, output_hidden_states=True).hidden_states[self.clip_skip]
        if self.sdxl:
            encoder_output_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            last_hidden_state = encoder_output_2.last_hidden_state

            # calculate pooled_output
            eos_token_index = torch.where(tokens_2 == self.tokenizer_2.eos_token_id)[1].to(device=last_hidden_state.device)
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
            pooled_output = self.text_encoder_2.text_projection(pooled_output)

            encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

            # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

            # pooled_output is zero vector for empty text            
            for i, token in enumerate(tokens_2):
                if token[1].item() == self.tokenizer_2.eos_token_id: # 二番目がEOSなら空文
                    pooled_output[i] = 0
        else:
            encoder_hidden_states = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
            pooled_output = None

        return encoder_hidden_states, pooled_output

    def forward(self, prompts):
        tokens, tokens_2 = self.tokenize(prompts)
        encoder_hidden_states, pooled_output = self.get_hidden_states(tokens, tokens_2)
        return encoder_hidden_states, pooled_output

    def enable_gradient_checkpointing(self, enable=True):
        if enable:
            self.text_encoder.gradient_checkpointing_enable()
            if self.sdxl:
                self.text_encoder_2.gradient_checkpointing_enable()
        else:
            self.text_encoder.gradient_checkpointing_disable()
            if self.sdxl:
                self.text_encoder_2.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(cls, path, sdxl=False, clip_skip=-1):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder')
        if sdxl:
            tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2')
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder_2')
        else:
            tokenizer_2 = None
            text_encoder_2 = None
        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
    
    def save_pretrained(self, save_directory):
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        if self.sdxl:
            self.text_encoder_2.save_pretrained(os.path.join(save_directory, "text_encoder_2"))
            self.tokenizer_2.save_pretrained(os.path.join(save_directory, "tokenizer_2"))
    
    def set_embedding_dtype(self, dtype):
        self.text_encoder.text_model.embeddings.to(dtype)
        if self.sdxl:
            if hasattr(self.text_encoder_2, 'text_model'):
                self.text_encoder_2.text_model.embeddings.to(dtype)
            else:
                for modules in self.text_encoder_2.modules():
                    if modules.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                        modules.to(dtype)