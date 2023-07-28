# This code is based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import torch
import torch.nn.functional as F
import math

from networks.loha import LohaModule
import contextlib

from safetensors.torch import load_file
import os

UNET_TARGET_REPLACE_MODULE_TRANSFORMER = ["Transformer2DModel"]
UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'
LORA_PREFIX_TEXT_ENCODER_1 = 'lora_te1'
LORA_PREFIX_TEXT_ENCODER_2 = 'lora_te2'


def load(file):
    if os.path.splitext(file)[1] == ".safetensors":
        return load_file(file)
    else:
        return torch.load(file, map_location="cpu")


def get_rank_and_alpha_from_state_dict(state_dict):
    modules_rank = {}
    modules_alpha = {}
    for key, value in state_dict.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            rank = value.size()[0]
            modules_rank[lora_name] = rank

    # support old LoRA without alpha
    for key in modules_rank.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_rank[key]

    return modules_rank, modules_alpha


class LoRAModule(torch.nn.Module):
    # replaces forward method of the original Linear, instead of replacing the original Linear module.

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_rank=4, alpha=1, forward_mode=None):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_rank = lora_rank
        self.forward_mode = forward_mode

        if org_module.__class__.__name__ == 'Linear':
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            if lora_rank == "dynamic":
                lora_rank = min(math.ceil(in_dim ** 0.5),
                                math.ceil(out_dim ** 0.5)) * 2
                self.lora_rank = lora_rank
            self.lora_down = torch.nn.Linear(in_dim, lora_rank, bias=False)
            self.lora_up = torch.nn.Linear(lora_rank, out_dim, bias=False)
            self.op = F.linear
            self.extra_args = {}
            kernel_size = (1, 1) # 便宜上の定義

        elif org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            if lora_rank == "dynamic":
                lora_rank = min(math.ceil(in_dim ** 0.5),
                                math.ceil(out_dim ** 0.5)) * 2
                self.lora_rank = lora_rank

            self.lora_rank = min(self.lora_rank, in_dim, out_dim)
            if self.lora_rank != lora_rank:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_rank}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_rank, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(
                self.lora_rank, out_dim, (1, 1), (1, 1), bias=False)
            
            self.op = F.conv2d
            self.extra_args = {
                "stride": stride,
                "padding": padding
            }

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nm = self.in_dim * self.out_dim * kernel_size[0] * kernel_size[1]
        self.nplusm = self.in_dim * kernel_size[0] * kernel_size[1] + self.out_dim
        self.shape = org_module.weight.shape

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_rank if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_rank
        self.register_buffer('alpha', torch.tensor(alpha))

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = [org_module] # moduleにならないようにlistに入れる

    def apply_to(self, multiplier=None):
        if multiplier is not None:
            self.multiplier = multiplier
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def unapply_to(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=None, sign=1):
        lora_weight = self.get_weight(multiplier) * sign

        # get org weight
        org_sd = self.org_module[0].state_dict()
        org_weight = org_sd["weight"]
        weight = org_weight + lora_weight.to(org_weight.device, dtype=org_weight.dtype)

        # set weight to org_module
        org_sd["weight"] = weight
        self.org_module[0].load_state_dict(org_sd)

    def restore_from(self, multiplier=None):
        self.merge_to(multiplier=multiplier, sign=-1)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight# out_dim, rank, [1, 1]
        down_weight = self.lora_down.weight # rank, in_dim, [kernel, kernel]
        
        lora_weight = up_weight.view(-1, self.lora_rank) @ down_weight.view(self.lora_rank, -1)  # out_dim, in_dim*kernel*kernel
        lora_weight = lora_weight.view(self.shape)  # out_dim, in_dim, [kernel, kernel]

        return lora_weight * multiplier * self.scale

    def forward(self, x):
        if self.multiplier == 0.0:
            return self.org_forward(x)
        else:
            if self.forward_mode == "merge":
                if len(x.shape) == 4:
                    b = x.shape[0] * x.shape[2] * x.shape[3]
                elif len(x.shape) == 3:
                    b = x.shape[0] * x.shape[1]
                else:
                    b = x.shape[0]
                #if self.nm < self.nplusm * b:
                if self.nm < b * (self.lora_rank + 2 * self.out_dim):
                    weight = self.get_weight() + self.org_module[0].weight
                    bias = None if self.org_module[0].bias is None else self.org_module[0].bias
                    return self.op(x, weight, bias, **self.extra_args)
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

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
            self.module = LohaModule
        else:
            self.module = LoRAModule

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

        if state_dict is not None:
            self.load_state_dict(state_dict, False)

        if mode == "apply":
            self.apply_to()
        elif mode == "merge":
            self.merge_to()

    @classmethod
    def from_file(cls, text_model, unet, path, multiplier=1.0, module=None, mode="apply"):
        state_dict = load(path)
        modules_rank, modules_alpha = get_rank_and_alpha_from_state_dict(state_dict)
        network = cls(text_model, unet, False, True, rank=modules_rank, alpha=modules_alpha,
                      multiplier=multiplier, module=module, mode=mode, state_dict=state_dict)
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

    def create_modules(self, prefix, root_module: torch.nn.Module, target_replace_modules, modules_rank=None, modules_alpha=1.0) -> list:
        loras = []
        is_dict = isinstance(modules_rank, dict)
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules and self.target_block in name:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"]:
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

        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import save_file
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        weights_sd = load(file)
        info = self.load_state_dict(weights_sd, False)
        return info

    @contextlib.contextmanager
    def set_temporary_multiplier(self, multiplier):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = multiplier
        yield
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = 1.0
