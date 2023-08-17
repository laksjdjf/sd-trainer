import importlib
import os
from safetensors.torch import load_file, save_file
import torch

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
    module = ".".join(config_text.split(".")[:-1])
    attr = config_text.split(".")[-1]
    return getattr(importlib.import_module(module), attr)

def default(dic, key, default_value):
    if hasattr(dic, key) and getattr(dic, key) is not None:
        return getattr(dic, key)
    else:
        return default_value