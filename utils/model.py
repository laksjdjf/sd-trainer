from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os

def load_model(path):
    if os.path.isfile(path):
        pipe = StableDiffusionPipeline.from_ckpt(path, scheduler_type = "ddim")
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
    return tokenizer, text_encoder, vae, unet, scheduler