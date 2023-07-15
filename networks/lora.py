# This code is based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import torch
import math
import os
from networks.loha import LohaModule
import contextlib

UNET_TARGET_REPLACE_MODULE_TRANSFORMER = ["Transformer2DModel"]
UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'


class LoRAModule(torch.nn.Module):
    # replaces forward method of the original Linear, instead of replacing the original Linear module.

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == 'Linear':
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            if lora_dim == "dynamic":
                lora_dim = min(math.ceil(in_dim ** 0.5),
                               math.ceil(out_dim ** 0.5)) * 2
                self.lora_dim = lora_dim
            self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            if lora_dim == "dynamic":
                lora_dim = min(math.ceil(in_dim ** 0.5),
                               math.ceil(out_dim ** 0.5)) * 2
                self.lora_dim = lora_dim

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(
                self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(
            alpha))                    # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module                  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        if self.multiplier == 0.0:
            return self.org_forward(x)
        else:
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
            
class LoRANetwork(torch.nn.Module):
    def __init__(self, text_encoder, unet, up_only, train_encoder=False, rank=4, conv_rank=None, multiplier=1.0, alpha=1, module=None) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = rank
        self.conv_lora_dim = conv_rank
        self.alpha = alpha
        self.target_block = "up_blocks" if up_only else ""

        if module == "loha":
            self.module = LohaModule
        else:
            self.module = LoRAModule

        # text encoderのloraを作る
        if train_encoder:
            self.text_encoder_loras = self.create_modules(
                LORA_PREFIX_TEXT_ENCODER, text_encoder, TEXT_ENCODER_TARGET_REPLACE_MODULE, self.lora_dim)
            print(
                f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")
        else:
            self.text_encoder_loras = []

        # unetのloraを作る
        self.unet_loras = []
        if self.lora_dim is not None:
            self.unet_loras += self.create_modules(
                LORA_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE_TRANSFORMER, self.lora_dim)
        if self.conv_lora_dim is not None:
            self.unet_loras += self.create_modules(
                LORA_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE_CONV, self.conv_lora_dim)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion 名前の被りがないか確認しているようだ
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        # loraを適用する
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        self.requires_grad_(True)

    # 見づらいのでメソッドにしちゃう
    def create_modules(self, prefix, root_module: torch.nn.Module, target_replace_modules, rank) -> list:
        loras = []
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules and self.target_block in name:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear","Conv2d"]:
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        lora = self.module(
                            lora_name, child_module, self.multiplier, rank, self.alpha)
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

        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import save_file
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)
            
    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    #重みの監視用だが使ってない
    def weight_log(self):
        state_dict = self.state_dict()
        means = {k: v.float().abs().mean() for k, v in state_dict.items()}
        target_keys = ["lora_up", "lora_down",
                       "down_blocks", "mid_block", "up_blocks",
                       "to_q", "to_k", "to_v", "to_out",
                       "ff_net_0", "ff_net_2",
                       "attn1", "attn2"
                       ]
        logs = {}
        for target_key in target_keys:
            logs[target_key] = torch.stack(
                [means[key] for key in means.keys() if target_key in key]).mean().item()
        return logs
    
    @contextlib.contextmanager
    def set_temporary_multiplier(self, multiplier):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = multiplier
        yield
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = 1.0

def add_lora(file, unet, text_encoder=None, multiplier=1.0):
    state_dict = load(file)
    keys = state_dict.keys()
    count = 0
    for name, module in unet.named_modules():
        if module.__class__.__name__ in UNET_TARGET_REPLACE_MODULE_TRANSFORMER + UNET_TARGET_REPLACE_MODULE_CONV:
            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ in ["Linear","Conv2d"]:
                    lora_name = LORA_PREFIX_UNET + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    target_keys = [key for key in keys if lora_name in key]
                    if len(target_keys) != 0:
                        count += 1
                        up = state_dict[[key for key in target_keys if "lora_up" in key][0]].to(unet.device,dtype = unet.dtype)
                        down = state_dict[[key for key in target_keys if "lora_down" in key][0]].to(unet.device,dtype = unet.dtype)
                        alpha = state_dict[[key for key in target_keys if "alpha" in key][0]].to(unet.device,dtype = unet.dtype)
                        target_shape = child_module.weight.shape
                        rank = up.shape[1]
                        up = up.view(-1,rank)
                        down = down.view(rank,-1)
                        weight = (up @ down).view(target_shape)
                        child_module.weight += weight * alpha * multiplier / rank
    print(f"Applied lora {count} modules in unet")
    count = 0
    
    if text_encoder is not None:
        for name, module in text_encoder.named_modules():
            if module.__class__.__name__ in TEXT_ENCODER_TARGET_REPLACE_MODULE:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear","Conv2d"]:
                        lora_name = LORA_PREFIX_TEXT_ENCODER + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        target_keys = [key for key in keys if lora_name in key]
                        if len(target_keys) != 0:
                            count += 1
                            up = state_dict[[key for key in target_keys if "lora_up" in key][0]].to(text_encoder.device,dtype = text_encoder.dtype)
                            down = state_dict[[key for key in target_keys if "lora_down" in key][0]].to(text_encoder.device,dtype = text_encoder.dtype)
                            alpha = state_dict[[key for key in target_keys if "alpha" in key][0]].to(text_encoder.device,dtype = text_encoder.dtype)
                            target_shape = child_module.weight.shape
                            rank = up.shape[1]
                            up = up.view(-1,rank)
                            down = down.view(rank,-1)
                            weight = (up @ down).view(target_shape)
                            child_module.weight += weight * alpha * multiplier / rank
    print(f"Applied lora {count} modules in text encoder")
