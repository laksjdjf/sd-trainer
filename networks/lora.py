# This code is based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import torch

import contextlib
import os

from utils.functions import get_attr_from_config, load_sd, save_sd

UNET_TARGET_REPLACE_MODULE_TRANSFORMER = ["Transformer2DModel"]
UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'
LORA_PREFIX_TEXT_ENCODER_1 = 'lora_te1'
LORA_PREFIX_TEXT_ENCODER_2 = 'lora_te2'

def get_rank_and_alpha_from_state_dict(state_dict):
    modules_rank = {}
    modules_alpha = {}
    ema = False
    for key, value in state_dict.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "ema" in key:
            ema = True
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            rank = value.size()[0]
            modules_rank[lora_name] = rank

    # support old LoRA without alpha
    for key in modules_rank.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_rank[key]

    return modules_rank, modules_alpha, ema

class LoRANetwork(torch.nn.Module):
    def __init__(
        self,
        text_model,
        unet,
        up_only,
        train_encoder=False,
        rank=4,
        conv_rank=None,
        multiplier=1.0,
        alpha=1.0,
        conv_alpha=1.0,
        module=None,
        mode="apply",  # "apply" or "merge"
        forward_mode=None,  # None or "merge"
        state_dict=None,
        ema=False,
    ) -> None:

        super().__init__()
        self.multiplier = multiplier
        self.lora_rank = rank
        self.alpha = alpha
        self.forward_mode = forward_mode

        if isinstance(self.lora_rank, dict):
            self.conv_lora_rank = self.lora_rank
            self.conv_alpha = self.alpha
        else:
            self.conv_lora_rank = conv_rank
            self.conv_alpha = conv_alpha

        self.target_block = "up_blocks" if up_only else ""

        if module == "loha":
            self.module = get_attr_from_config("networks.lora_modules.LohaModule")
        elif module is None:
            self.module = get_attr_from_config("networks.lora_modules.LoRAModule")
        else:
            self.module = get_attr_from_config(module)

        # text encoderのloraを作る
        if train_encoder:
            if text_model.sdxl:
                self.text_encoder_loras = self.create_modules(
                    LORA_PREFIX_TEXT_ENCODER_1, text_model.text_encoder, TEXT_ENCODER_TARGET_REPLACE_MODULE, self.lora_rank)
                self.text_encoder_loras += self.create_modules(
                    LORA_PREFIX_TEXT_ENCODER_2, text_model.text_encoder_2, TEXT_ENCODER_TARGET_REPLACE_MODULE, self.lora_rank)
                print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")
            else:
                self.text_encoder_loras = self.create_modules(
                    LORA_PREFIX_TEXT_ENCODER, text_model.text_encoder, TEXT_ENCODER_TARGET_REPLACE_MODULE, self.lora_rank)
                print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")
        else:
            self.text_encoder_loras = []

        # unetのloraを作る
        self.unet_loras = []
        if self.lora_rank is not None:
            self.unet_loras += self.create_modules(
                LORA_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE_TRANSFORMER, self.lora_rank, self.alpha)
        if self.conv_lora_rank is not None:
            self.unet_loras += self.create_modules(
                LORA_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE_CONV, self.conv_lora_rank, self.conv_alpha)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion 名前の被りがないか確認しているようだ
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        # loraを適用する
        for lora in self.text_encoder_loras + self.unet_loras:
            self.add_module(lora.lora_name, lora)
        
        if ema:
            self.update_ema()

        if state_dict is not None:
            self.load_state_dict(state_dict, False)

        if mode == "apply":
            self.apply_to()
        elif mode == "merge":
            self.merge_to()

    @classmethod
    def from_file(cls, text_model, unet, path, multiplier=1.0, module=None, mode="apply"):
        state_dict = load_sd(path)
        modules_rank, modules_alpha, ema = get_rank_and_alpha_from_state_dict(state_dict)
        network = cls(text_model, unet, False, True, rank=modules_rank, alpha=modules_alpha,
                      multiplier=multiplier, module=module, mode=mode, state_dict=state_dict, ema=ema)
        return network

    def apply_to(self, multiplier=None):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to(multiplier)

    def unapply_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unapply_to()

    def merge_to(self, multiplier=None):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.merge_to(multiplier)

    def restore_from(self, multiplier=None):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.restore_from(multiplier)

    def update_ema(self, decay=0.999):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.update_ema(decay)

    def create_modules(self, prefix, root_module: torch.nn.Module, target_replace_modules, modules_rank=None, modules_alpha=1.0) -> list:
        loras = []
        is_dict = isinstance(modules_rank, dict)
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules and self.target_block in name:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if is_dict:
                            if lora_name in modules_rank:
                                rank = modules_rank[lora_name]
                                alpha = modules_alpha[lora_name]
                            else:
                                continue
                        else:
                            rank = modules_rank
                            alpha = modules_alpha
                        lora = self.module(
                            lora_name, child_module, self.multiplier, rank, alpha, forward_mode=self.forward_mode)
                        loras.append(lora)
        return loras

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.text_encoder_loras]
            param_data = {'params': params}
            if text_encoder_lr is not None:
                param_data['lr'] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {'params': params}
            if unet_lr is not None:
                param_data['lr'] = unet_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=torch.float16):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == '':
            file += '.safetensors'

        save_sd(state_dict, file)

    def load_weights(self, file):
        weights_sd = load_sd(file)
        info = self.load_state_dict(weights_sd, False)
        return info

    @contextlib.contextmanager
    def set_temporary_multiplier(self, multiplier):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = multiplier
        yield
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = 1.0

    @contextlib.contextmanager
    def set_temporary_ema(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.ema = True
        yield
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.ema = False
