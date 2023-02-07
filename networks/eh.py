#### This code is based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import torch
import os
from diffusers import UNet2DConditionModel

UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"] #Attentionはいらないのでは？
EH_PREFIX_UNET = 'EH_unet'

#EhModule
class EHModule(torch.nn.Module):
#replaces forward method of the original Linear, instead of replacing the original Linear module.

    def __init__(self, eh_name, org_module: torch.nn.Module, num_groups:int = 4, multiplier:float = 1.0):
        super().__init__()
        self.eh_name = eh_name
        self.num_groups = num_groups
        self.multiplier = multiplier
        
        #とりまLinearだけ
        if org_module.__class__.__name__ == 'Linear':
            in_dim = org_module.in_features 
            out_dim = org_module.out_features
            self.linear = torch.nn.Linear(in_dim // num_groups, out_dim // num_groups, bias=False)

        torch.nn.init.zeros_(self.linear.weight)
        self.org_module = org_module                  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module
    
    def merge_to(self, path):
        row, column = self.org_module.weight.shape
        new_weight = torch.zeros((row, column))
        
        #ごみ、良い方法おせーて
        for i in range(self.num_groups):
            new_weight[i * row // self.num_groups:(i+1) * row // self.num_groups,i * column // self.num_groups:(i+1) * column // self.num_groups] = self.linear.weight
            
        self.org_module.weight += new_weight * self.multiplier

    def forward(self, x):
        #(B:バッチサイズ,Np:トークン数,D:埋め込み次元/チャンネル)
        shape = x.shape
        
        #(B,Np,in_D) -> (B,Np,out_D)
        y = self.org_forward(x)
        
        #(B,Np,in_D) -> (B,Np,n,in_D/n)
        z = x.reshape(shape[0],shape[1],self.num_groups,-1)
        #(B,Np,n,in_D/n) -> (B,Np,n,out_D/n)
        z = self.linear(z)
        #(B,Np,n,out_D/n) -> (B,Np,out_D)
        z = z.reshape(shape[0],shape[1],-1)
        
        return y + z * self.multiplier
    
class EHNetwork(torch.nn.Module):
    def __init__(self, unet:UNet2DConditionModel, num_groups:int = 4, multiplier:float = 1.0, merge:bool = False) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.multiplier = multiplier
        
        
        #unetのEHを作る
        self.unet_ehs = self.create_modules(EH_PREFIX_UNET, unet, UNET_TARGET_REPLACE_MODULE)
        print(f"create EH for U-Net: {len(self.unet_ehs)} modules.")
        
        # ehを適用する
        for eh in self.unet_ehs:
            self.add_module(eh.eh_name, eh)
            if merge:
                eh.merge_to() #重みのマージ
            else:
                eh.apply_to() #モジュールのあぷらい
        self.requires_grad_(True)
        

    #見づらいのでメソッドにしちゃう
    def create_modules(self,prefix, root_module: torch.nn.Module, target_replace_modules) -> list:
        ehs = []
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear":
                        eh_name = prefix + '.' + name + '.' + child_name
                        eh_name = eh_name.replace('.', '_')
                        eh = EHModule(eh_name, child_module, self.num_groups, self.multiplier)
                        ehs.append(eh)
        return ehs
    
    def prepare_optimizer_params(self, unet_lr):
        self.requires_grad_(True)
        all_params = []

        params = []
        [params.extend(eh.parameters()) for eh in self.unet_ehs] #ehの全パラメータ
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
