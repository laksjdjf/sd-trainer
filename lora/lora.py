#### This code is based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import torch
import math
import os

UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"] #Attentionはいらないのでは？
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'

#LoRAModule (W=W0 + ΔW = W0 + αBA^T)
class LoRAModule(torch.nn.Module):
#replaces forward method of the original Linear, instead of replacing the original Linear module.

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        
        #とりまLinearだけ
        if org_module.__class__.__name__ == 'Linear':
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))                    # 定数として扱える

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
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
    
#LoRANetwork
class LoRANetwork(torch.nn.Module):
    def __init__(self, text_encoder, unet, lora_dim=4, target_block = "" ,multiplier=1.0, alpha=1) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.target_block = target_block
        
        #text encoderのloraを作る
        if text_encoder is not None:
            self.text_encoder_loras = self.create_modules(LORA_PREFIX_TEXT_ENCODER,text_encoder,TEXT_ENCODER_TARGET_REPLACE_MODULE)
            print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")
        else:
            self.text_encoder_loras = []
        
        #unetのloraを作る
        self.unet_loras = self.create_modules(LORA_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE)
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
        

    #見づらいのでメソッドにしちゃう
    def create_modules(self,prefix, root_module: torch.nn.Module, target_replace_modules) -> list:
        loras = []
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules and self.target_block in name:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear":
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim, self.alpha)
                        loras.append(lora)
        return loras
    
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.text_encoder_loras] #loraの全パラメータ
            param_data = {'params': params}
            if text_encoder_lr is not None:
                param_data['lr'] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras] #loraの全パラメータ
            param_data = {'params': params}
            if unet_lr is not None:
                param_data['lr'] = unet_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import save_file
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)
            
    def weight_log(self):
        state_dict = self.state_dict()
        
        means = {k:v.float().abs().mean() for k,v in state_dict.items()} 
        
        target_keys = ["lora_up","lora_down",
                       #up onlyに対応できない"down_blocks","mid_block","up_blocks",
                       "to_q","to_k","to_v","to_out",
                       "ff_net_0","ff_net_2",
                       "attn1","attn2"
                      ]
        
        logs = {}
        
        for target_key in target_keys:
            logs[target_key] = torch.stack([means[key] for key in means.keys() if target_key in key]).mean().item()
        return logs
