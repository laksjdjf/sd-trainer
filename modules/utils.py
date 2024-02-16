import importlib
import os
from safetensors.torch import load_file, save_file
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from modules.text_model import TextModel

# データローダー用の関数
def collate_fn(x):
    return x[0]

# モデルのロード用の関数
def load_sd(file):
    if os.path.splitext(file)[1] == ".safetensors":
        return load_file(file)
    else:
        return torch.load(file, map_location="cpu")
    
# モデルのセーブ用の関数
def save_sd(state_dict, file):
    if os.path.splitext(file)[1] == '.safetensors':
        save_file(state_dict, file)
    else:
        torch.save(state_dict, file)

# 文字列からモジュールを取得
def get_attr_from_config(config_text: str):
    if config_text is None:
        return None
    module = ".".join(config_text.split(".")[:-1])
    attr = config_text.split(".")[-1]
    return getattr(importlib.import_module(module), attr)


def load_model(path, sdxl=False, clip_skip=-1):
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
            text_model = TextModel(tokenizer, tokenizer_2, text_encoder, text_encoder_2)
            del pipe
        else:
            text_model = TextModel.from_pretrained(path, sdxl=True)
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
            text_model = TextModel(tokenizer, None, text_encoder, None)
            del pipe
        else:
            text_model = TextModel.from_pretrained(path)
            unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet')
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae')
            scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler')
            
    text_model.clip_skip = clip_skip
    return text_model, vae, unet, scheduler