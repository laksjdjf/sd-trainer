from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import os
import torch
import torch.nn as nn


def load_model(path, sdxl=False):
    if sdxl:
        if os.path.isfile(path):
            pipe = StableDiffusionXLPipeline.from_single_file(path, scheduler_type="ddim")
            tokenizer = pipe.tokenizer
            tokenizer_2 = pipe.tokenizer_2
            text_encoder = pipe.text_encoder
            text_encoder_2 = pipe.text_encoder_2
            unet = pipe.unet
            vae = pipe.vae
            scheduler = pipe.scheduler
            del pipe
        else:
            tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer')
            text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder')
            tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2')
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder_2')
            unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet')
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae')
            scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler')
    else:
        if os.path.isfile(path):
            pipe = StableDiffusionPipeline.from_single_file(path, scheduler_type="ddim")
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
            unet = pipe.unet
            vae = pipe.vae
            scheduler = pipe.scheduler
            del pipe
        else:
            tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer')
            text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder')
            unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet')
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae')
            scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler')
        text_encoder_2, tokenizer_2 = None, None
    text_model = TextModel(tokenizer, tokenizer_2, text_encoder, text_encoder_2)
    return text_model, vae, unet, scheduler


class TextModel(nn.Module):
    def __init__(self, tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=-1):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.clip_skip = clip_skip
        self.sdxl = tokenizer_2 is not None

    def tokenize(self, texts):
        tokens = self.tokenizer(texts, max_length=self.tokenizer.model_max_length, padding="max_length",
                                truncation=True, return_tensors='pt').input_ids.to(self.text_encoder.device)
        if self.sdxl:
            tokens_2 = self.tokenizer_2(texts, max_length=self.tokenizer_2.model_max_length, padding="max_length",
                                        truncation=True, return_tensors='pt').input_ids.to(self.text_encoder_2.device)
            empty_text = []
            for text in texts:
                if text == "":
                    empty_text.append(True)
                else:
                    empty_text.append(False)
        else:
            tokens_2 = None
            empty_text = None

        return tokens, tokens_2, empty_text

    def forward(self, tokens, tokens_2=None, empty_text=None):
        encoder_hidden_states = self.text_encoder(tokens, output_hidden_states=True).hidden_states[self.clip_skip]
        if self.sdxl:
            encoder_output_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            pooled_output = encoder_output_2[0]
            encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

            # pooled_output is zero vector for empty text
            if empty_text is not None:
                for i, empty in enumerate(empty_text):
                    if empty:
                        pooled_output[i] = torch.zeros_like(pooled_output[i])
        else:
            encoder_hidden_states = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
            pooled_output = None

        return encoder_hidden_states, pooled_output
    
    def gradient_checkpointing_enable(self, enable=True):
        if enable:
            self.text_encoder.gradient_checkpointing_enable()
            if self.sdxl:
                self.text_encoder_2.gradient_checkpointing_enable()
        else:
            self.text_encoder.gradient_checkpointing_disable()
            if self.sdxl:
                self.text_encoder_2.gradient_checkpointing_disable()
