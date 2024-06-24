import torch
import torch.nn as nn
from modules.utils import get_attr_from_config
from modules.utils import save_sd, load_sd
import os
import contextlib
import logging

logger = logging.getLogger("ネットワークちゃん")

UNET_TARGET_REPLACE_MODULE_TRANSFORMER = ["Transformer2DModel"]
UNET_TARGET_REPLACE_MODULE_ATTENTION = ["Transformer2DModel"]
MMDIT_TARGET_REPLACE_MODULE = ["JointTransformerBlock"]
UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_MMDIT = 'lora_transformer'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'
LORA_PREFIX_TEXT_ENCODER_1 = 'lora_te1'
LORA_PREFIX_TEXT_ENCODER_2 = 'lora_te2'

def is_key_allowed(key, key_filters):
    if key_filters is None:
        return True
    else:
        return any(filter in key for filter in key_filters)
    
def detect_module_from_state_dict(state_dict):
    for key in state_dict.keys():
        if "lora_up" in key:
            return "networks.lora.LoRAModule"
    return ValueError("cannot detect module from state_dict")

class NetworkManager(nn.Module):
    def __init__(
        self, 
        text_model,
        unet, 
        module=None,
        file_name=None,
        module_args=None,
        unet_key_filters=None,
        conv_module_args=None,
        text_module_args=None,
        multiplier=1.0,
        mode="apply", # select "apply" or "merge"
    ):
        super().__init__()
        self.multiplier = multiplier

        if file_name is not None:
            state_dict = load_sd(file_name)
            module = detect_module_from_state_dict(state_dict)
        else:
            state_dict = None
            
        keys = [] if state_dict is None else set([key.split(".")[0] for key in state_dict.keys()])
        unet_keys = [key for key in keys if LORA_PREFIX_UNET in key]
        te_keys = [key for key in keys if LORA_PREFIX_TEXT_ENCODER in key]
        te1_keys = [key for key in keys if LORA_PREFIX_TEXT_ENCODER_1 in key]
        te2_keys = [key for key in keys if LORA_PREFIX_TEXT_ENCODER_2 in key]

        self.module = get_attr_from_config(module)

        if hasattr(conv_module_args, "same") and conv_module_args.same:
            conv_module_args = module_args
        if hasattr(text_module_args, "same") and text_module_args.same:
            text_module_args = module_args
        
        # unetのloraを作る
        self.unet_modules = []
        if state_dict or module_args is not None:
            self.unet_modules += self.create_modules(LORA_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE_TRANSFORMER, state_dict, module_args, unet_key_filters, unet_keys)
            self.unet_modules += self.create_modules(LORA_PREFIX_MMDIT, unet, MMDIT_TARGET_REPLACE_MODULE, state_dict, module_args, unet_key_filters, unet_keys)
        if state_dict or conv_module_args is not None:
            self.unet_modules += self.create_modules(LORA_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE_CONV, state_dict, conv_module_args, unet_key_filters, unet_keys)
        if state_dict or text_module_args is not None:
            self.text_encoder_modules = []
            if text_model.sdxl:
                self.text_encoder_modules += self.create_modules(LORA_PREFIX_TEXT_ENCODER_1, text_model.text_encoder, TEXT_ENCODER_TARGET_REPLACE_MODULE, state_dict, text_module_args, None, te1_keys)
                self.text_encoder_modules += self.create_modules(LORA_PREFIX_TEXT_ENCODER_2, text_model.text_encoder_2, TEXT_ENCODER_TARGET_REPLACE_MODULE, state_dict, text_module_args, None, te2_keys)
            else:
                self.text_encoder_modules += self.create_modules(LORA_PREFIX_TEXT_ENCODER, text_model.text_encoder, TEXT_ENCODER_TARGET_REPLACE_MODULE, state_dict, text_module_args, None, te_keys)
        else:
            self.text_encoder_modules = []

        self.apply_te = len(self.text_encoder_modules) > 0
        
        logger.info(f"UNetのモジュールは: {len(self.unet_modules)}個だよ。")
        logger.info(f"TextEncoderのモジュールは: {len(self.text_encoder_modules)}個だよ。")

        for lora in self.text_encoder_modules + self.unet_modules:
            self.add_module(lora.lora_name, lora)

        if mode == "apply":
            self.apply_to()
        elif mode == "merge":
            self.merge_to()
        else:
            raise ValueError(f"mode {mode} is not supported.")

    def apply_to(self, multiplier=None):
        for lora in self.text_encoder_modules + self.unet_modules:
            lora.apply_to(multiplier)

    def unapply_to(self):
        for lora in self.text_encoder_modules + self.unet_modules:
            lora.unapply_to()

    def merge_to(self, multiplier=None):
        for lora in self.text_encoder_modules + self.unet_modules:
            lora.merge_to(multiplier)

    def restore_from(self, multiplier=None):
        for lora in self.text_encoder_modules + self.unet_modules:
            lora.restore_from(multiplier)

    def create_modules(self, prefix, root_module, target_replace_modules, state_dict=None, module_args=None, unet_limited_keys=None, target_keys=None) -> list:
        modules = []
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if state_dict is not None:
                            if target_keys and lora_name in target_keys:
                                lora = self.module(lora_name, child_module, self.multiplier, state_dict)
                                modules.append(lora)
                        else:
                            if is_key_allowed(lora_name, unet_limited_keys):
                                lora = self.module(lora_name, child_module, self.multiplier, state_dict, **module_args)
                                modules.append(lora)
        return modules

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_modules:
            params = []
            [params.extend(lora.parameters()) for lora in self.text_encoder_modules]
            param_data = {'params': params}
            if text_encoder_lr is not None:
                param_data['lr'] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_modules:
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_modules]
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
        for lora in self.text_encoder_modules + self.unet_modules:
            lora.multiplier = multiplier
        yield
        for lora in self.text_encoder_modules + self.unet_modules:
            lora.multiplier = 1.0
