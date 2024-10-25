import importlib
import os
from safetensors.torch import load_file, save_file
import torch
import math
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline, SD3Transformer2DModel, FluxTransformer2DModel, AuraFlowTransformer2DModel
from modules.diffusion_model import DiffusionModel, SD3DiffusionModel, FluxDiffusionModel, AuraFlowDiffusionModel
from modules.text_model import SD1TextModel, SDXLTextModel, SD3TextModel, FluxTextModel, AuraFlowTextModel
from modules.scheduler import BaseScheduler, FlowScheduler

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


def load_model(path, model_type="sd1", clip_skip=-1, revision=None, torch_dtype=None, variant=None, nf4=False):

    if nf4:
        from diffusers import BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        nf4_config = None

    if model_type == "sdxl":
        if os.path.isfile(path):
            pipe = StableDiffusionXLPipeline.from_single_file(path, scheduler_type="ddim")
            tokenizer = pipe.tokenizer
            tokenizer_2 = pipe.tokenizer_2
            text_encoder = pipe.text_encoder
            text_encoder_2 = pipe.text_encoder_2
            unet = pipe.unet
            vae = pipe.vae
            diffusers_scheduler = pipe.scheduler
            text_model = SDXLTextModel(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
            del pipe
        else:
            text_model = SDXLTextModel.from_pretrained(path, clip_skip=clip_skip, revision=revision, torch_dtype=torch_dtype, variant=variant)
            unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=nf4_config)
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)
            diffusers_scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler', revision=revision)
        diffusion = DiffusionModel(unet, sdxl=True)
        scheduler = BaseScheduler(diffusers_scheduler.config.prediction_type == "v_prediction")
    elif model_type == "sd1":
        if os.path.isfile(path):
            pipe = StableDiffusionPipeline.from_single_file(path, scheduler_type="ddim")
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
            unet = pipe.unet
            vae = pipe.vae
            diffusers_scheduler = pipe.scheduler
            text_model = SD1TextModel(tokenizer, text_encoder, clip_skip=clip_skip)
            del pipe
        else:
            text_model = SD1TextModel.from_pretrained(path, clip_skip=clip_skip, revision=revision, torch_dtype=torch_dtype, variant=variant)
            unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=nf4_config)
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)
            diffusers_scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler', revision=revision)
        scheduler = BaseScheduler(diffusers_scheduler.config.prediction_type == "v_prediction")
        diffusion = DiffusionModel(unet)
    elif model_type == "sd3":
        if os.path.isfile(path):
            NotImplementedError("from_single_file is not implemented for SD3")
        else:
            text_model = SD3TextModel.from_pretrained(path, clip_skip=clip_skip, revision=revision, torch_dtype=torch_dtype, variant=variant)
            unet = SD3Transformer2DModel.from_pretrained(path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=nf4_config)
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)
            diffusers_scheduler = None
        scheduler = FlowScheduler()
        diffusion = SD3DiffusionModel(unet)
    elif model_type == "flux":
        if os.path.isfile(path):
            NotImplementedError("from_single_file is not implemented for Flux")
        else:
            text_model = FluxTextModel.from_pretrained(path, revision=revision, torch_dtype=torch_dtype, variant=variant)
            unet = FluxTransformer2DModel.from_pretrained(path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=nf4_config)
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)
            diffusers_scheduler = None
        scheduler = FlowScheduler(shift=math.exp(1.15))
        diffusion = FluxDiffusionModel(unet)
    elif model_type == "auraflow":
        if os.path.isfile(path):
            NotImplementedError("from_single_file is not implemented for AuraFlow")
        else:
            text_model = AuraFlowTextModel.from_pretrained(path, revision=revision, torch_dtype=torch_dtype, variant=variant)
            unet = AuraFlowTransformer2DModel.from_pretrained(path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=nf4_config)
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)
            diffusers_scheduler = None
        scheduler = FlowScheduler(shift=1.73)
        diffusion = AuraFlowDiffusionModel(unet)

    text_model.clip_skip = clip_skip
    return text_model, vae, diffusion, diffusers_scheduler, scheduler