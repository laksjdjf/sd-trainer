from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import os

def load_model(path, sdxl = False):
    if sdxl:
        if os.path.isfile(path):
            pipe = StableDiffusionXLPipeline.from_single_file(path, scheduler_type = "ddim")
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
        return tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae, unet, scheduler
    else:
        if os.path.isfile(path):
            pipe = StableDiffusionPipeline.from_single_file(path, scheduler_type = "ddim")
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
        return tokenizer, None, text_encoder, None, vae, unet, scheduler