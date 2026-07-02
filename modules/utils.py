import importlib
import os
from safetensors.torch import load_file, save_file
import torch
from modules.model_registry import get_model_spec

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

def load_model(path, model_type="sd1", clip_skip=-1, revision=None, torch_dtype=None, variant=None, nf4=False, taesd=False):

    if nf4:
        from diffusers import BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        nf4_config = None

    spec = get_model_spec(model_type)
    text_model, vae, diffusion, diffusers_scheduler, scheduler = spec.load(
        spec, path, clip_skip, revision, torch_dtype, variant, nf4_config, taesd
    )

    text_model.clip_skip = clip_skip
    return text_model, vae, diffusion, diffusers_scheduler, scheduler
