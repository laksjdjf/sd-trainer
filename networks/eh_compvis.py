# reference:https://github.com/kohya-ss/sd-webui-additional-networks/blob/main/scripts/lora_compvis.py

import copy
import math
import re
from typing import NamedTuple
import torch


class EHInfo(NamedTuple):
    eh_name: str
    module_name: str
    module: torch.nn.Module
    multiplier: float
    dim: int


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
    
    def merge_to(self):
        print(f"merging {self.eh_name}")
        row, column = self.org_module.weight.shape
        new_weight = torch.zeros((row, column))
        
        #ごみ、良い方法おせーて
        for i in range(self.num_groups):
            new_weight[i * row // self.num_groups:(i+1) * row // self.num_groups,i * column // self.num_groups:(i+1) * column // self.num_groups] = self.linear.weight
            
        self.org_module.weight = torch.nn.Parameter(self.org_module.weight + new_weight * self.multiplier)

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

# text encoder？知らん。
def create_network_and_apply_compvis(du_state_dict, multiplier, unet, **kwargs):
    # get device and dtype from unet
    for module in unet.modules():
        if module.__class__.__name__ == "Linear":
            param: torch.nn.Parameter = module.weight
            # device = param.device
            dtype = param.dtype
            break
    
    #個人的に出力側の層を学習しないメリットを感じないので、もうこのweightがあること前提にしてしまう。
    num_groups = 320 // du_state_dict["EH_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_k.linear.weight"].shape[0]
    
    # create, apply and load weights
    network = EHNetworkCompvis(unet, num_groups, multiplier=multiplier)
    state_dict = network.apply_eh_modules(du_state_dict)    # some weights are applied to text encoder
    network.to(dtype)  # with this, if error comes from next line, the model will be used
    info = network.load_state_dict(state_dict, strict=False)
    
    
    """
    # とりあえずわからんから無効化
    # remove redundant warnings
    if len(info.missing_keys) > 4:
        missing_keys = []
        alpha_count = 0
        for key in info.missing_keys:
            if 'alpha' not in key:
                missing_keys.append(key)
            else:
                if alpha_count == 0:
                    missing_keys.append(key)
                alpha_count += 1
        if alpha_count > 1:
            missing_keys.append(
                    f"... and {alpha_count-1} alphas. The model doesn't have alpha, use dim (rannk) as alpha. You can ignore this message.")

        info = torch.nn.modules.module._IncompatibleKeys(missing_keys, info.unexpected_keys)
    
    """

    return network, info


class EHNetworkCompvis(torch.nn.Module):
    # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    # TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    
    #え？なんでこっち？
    UNET_TARGET_REPLACE_MODULE = ["SpatialTransformer"]    # , "Attention"]
    #TEXT_ENCODER_TARGET_REPLACE_MODULE = ["ResidualAttentionBlock", "CLIPAttention", "CLIPMLP"]

    EH_PREFIX_UNET = 'EH_unet'
    
    #diffusersのkeyからcompviのkeyに置き換える
    @classmethod
    def convert_diffusers_name_to_compvis(cls, v2, du_name):
        """
        convert diffusers's EH name to CompVis
        """
        cv_name = None
        if "EH_unet_" in du_name:
            m = re.search(r"_down_blocks_(\d+)_attentions_(\d+)_(.+)", du_name)
            if m:
                du_block_index = int(m.group(1))
                du_attn_index = int(m.group(2))
                du_suffix = m.group(3)

                cv_index = 1 + du_block_index * 3 + du_attn_index            # 1,2, 4,5, 7,8
                cv_name = f"EH_unet_input_blocks_{cv_index}_1_{du_suffix}"
            else:
                m = re.search(r"_mid_block_attentions_(\d+)_(.+)", du_name)
                if m:
                    du_suffix = m.group(2)
                    cv_name = f"EH_unet_middle_block_1_{du_suffix}"
                else:
                    m = re.search(r"_up_blocks_(\d+)_attentions_(\d+)_(.+)", du_name)
                    if m:
                        du_block_index = int(m.group(1))
                        du_attn_index = int(m.group(2))
                        du_suffix = m.group(3)

                        cv_index = du_block_index * 3 + du_attn_index            # 3,4,5, 6,7,8, 9,10,11
                        cv_name = f"EH_unet_output_blocks_{cv_index}_1_{du_suffix}"
                        
        '''
        elif "eh_te_" in du_name:
            m = re.search(r"_model_encoder_layers_(\d+)_(.+)", du_name)
            if m:
                du_block_index = int(m.group(1))
                du_suffix = m.group(2)

                cv_index = du_block_index
                if v2:
                    if 'mlp_fc1' in du_suffix:
                        cv_name = f"eh_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc1', 'mlp_c_fc')}"
                    elif 'mlp_fc2' in du_suffix:
                        cv_name = f"eh_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('mlp_fc2', 'mlp_c_proj')}"
                    elif 'self_attn':
                        # handled later
                        cv_name = f"eh_te_wrapped_model_transformer_resblocks_{cv_index}_{du_suffix.replace('self_attn', 'attn')}"
                else:
                    cv_name = f"eh_te_wrapped_transformer_text_model_encoder_layers_{cv_index}_{du_suffix}"
        '''

        assert cv_name is not None, f"conversion failed: {du_name}. the model may not be trained by `sd-scripts`."
        return cv_name
    
    #diffusersの辞書からcompvisの辞書に置き換えて返す
    @classmethod
    def convert_state_dict_name_to_compvis(cls, v2, state_dict):
        """
        convert keys in state dict to load it by load_state_dict
        """
        new_sd = {}
        for key, value in state_dict.items():
            tokens = key.split('.')
            compvis_name = EHNetworkCompvis.convert_diffusers_name_to_compvis(v2, tokens[0])
            new_key = compvis_name + '.' + '.'.join(tokens[1:])

            new_sd[new_key] = value

        return new_sd

    def __init__(self, unet, num_groups = 4, multiplier = 1.0) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.num_groups = num_groups

        # create module instances
        self.v2 = False

        def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules, multiplier):
            ehs = []
            replaced_modules = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == "Linear":
                            eh_name = prefix + '.' + name + '.' + child_name
                            eh_name = eh_name.replace('.', '_')
                            if '_resblocks_23_' in eh_name:        # ignore last block in StabilityAi Text Encoder
                                break
                            eh = EHModule(eh_name, child_module, self.num_groups, multiplier)
                            ehs.append(eh)

                            replaced_modules.append(child_module)
                        elif child_module.__class__.__name__ == "MultiheadAttention":
                            # make four modules: not replacing forward method but merge weights
                            self.v2 = True
                            for suffix in ['q', 'k', 'v', 'out']:
                                module_name = prefix + '.' + name + '.' + child_name            # ~.attn
                                module_name = module_name.replace('.', '_')
                                if '_resblocks_23_' in module_name:              # ignore last block in StabilityAi Text Encoder
                                    break
                                eh_name = module_name + '_' + suffix
                                eh_info = EHInfo(eh_name, module_name, child_module, multiplier, self.num_groups)
                                ehs.append(eh_info)

                                replaced_modules.append(child_module)
            return ehs, replaced_modules

        self.unet_ehs, unet_rep_modules = create_modules(
                EHNetworkCompvis.EH_PREFIX_UNET, unet, EHNetworkCompvis.UNET_TARGET_REPLACE_MODULE, self.multiplier)
        print(f"create EH for U-Net: {len(self.unet_ehs)} modules.")

        # make backup of original forward/weights, if multiple modules are applied, do in 1st module only
        backed_up = False                                         # messaging purpose only
        for rep_module in unet_rep_modules:
            if rep_module.__class__.__name__ == "MultiheadAttention":    # multiple MHA modules are in list, prevent to backed up forward
                if not hasattr(rep_module, "_eh_org_weights"):
                    # avoid updating of original weights. state_dict is reference to original weights
                    rep_module._eh_org_weights = copy.deepcopy(rep_module.state_dict())
                    backed_up = True
            elif not hasattr(rep_module, "_eh_org_forward"):
                rep_module._eh_org_forward = rep_module.forward
                backed_up = True
        if backed_up:
            print("original forward/weights is backed up.")

        # assertion
        names = set()
        for eh in self.unet_ehs:
            assert eh.eh_name not in names, f"duplicated eh name: {eh.eh_name}"
            names.add(eh.eh_name)

    def restore(self, text_encoder, unet):
        # restore forward/weights from property for all modules
        restored = False              # messaging purpose only
        modules = []
        modules.extend(unet.modules())
        for module in modules:
            if hasattr(module, "_eh_org_forward"):
                module.forward = module._eh_org_forward
                del module._eh_org_forward
                restored = True
            if hasattr(module, "_eh_org_weights"):   # module doesn't have forward and weights at same time currently, but supports it for future changing
                module.load_state_dict(module._eh_org_weights)
                del module._eh_org_weights
                restored = True

        if restored:
            print("original forward/weights is restored.")

    def apply_eh_modules(self, du_state_dict):
        # conversion 1st step: convert names in state_dict
        state_dict = EHNetworkCompvis.convert_state_dict_name_to_compvis(self.v2, du_state_dict)

        # add modules to network: this makes state_dict can be got from EHNetwork
        mha_ehs = {}
        for eh in self.unet_ehs:
            if type(eh) == EHModule:
                eh.apply_to()        # ensure remove reference to original Linear: reference makes key of state_dict
                self.add_module(eh.eh_name, eh)
            else:
                # SD2.x MultiheadAttention merge weights to MHA weights
                eh_info: EHInfo = eh
                if eh_info.module_name not in mha_ehs:
                    mha_ehs[eh_info.module_name] = {}

                eh_dic = mha_ehs[eh_info.module_name]
                eh_dic[eh_info.eh_name] = eh_info
                if len(eh_dic) == 4:
                    # calculate and apply
                    w_q = state_dict.get(eh_info.module_name + '_q_proj.linear.weight')
                    if w_q is not None:          # corresponding LoRa module exists
                        w_k = state_dict[eh_info.module_name + '_k_proj.linear.weight']
                        w_v = state_dict[eh_info.module_name + '_v_proj.linear.weight']
                        w_out = state_dict[eh_info.module_name + '_out_proj.linear.weight']

                        sd = eh_info.module.state_dict()
                        qkv_weight = sd['in_proj_weight']
                        out_weight = sd['out_proj.weight']
                        dev = qkv_weight.device
                        
                        def merge_weights(weight, eh_weight):
                            dtype = weight.dtype
                            row, column = weight.weight.shape
                            new_weight = torch.zeros((row, column))

                            #ごみ、良い方法おせーて
                            for i in range(self.num_groups):
                                new_weight[i * row // self.num_groups:(i+1) * row // self.num_groups,i * column // self.num_groups:(i+1) * column // self.num_groups] = eh_weight

                            weight = weight.float() + eh_info.multiplier * new_weight.to(dev, dtype=torch.float)
                            weight = weight.to(dtype)
                            return weight

                        q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3)
                        if q_weight.size()[1] == w_q.size()[0]:
                            q_weight = merge_weights(q_weight, w_q)
                            k_weight = merge_weights(k_weight, w_k)
                            v_weight = merge_weights(v_weight, w_v)
                            qkv_weight = torch.cat([q_weight, k_weight, v_weight])

                            out_weight = merge_weights(out_weight, w_out)

                            sd['in_proj_weight'] = qkv_weight.to(dev)
                            sd['out_proj.weight'] = out_weight.to(dev)

                            eh_info.module.load_state_dict(sd)
                        else:
                            # different dim, version mismatch
                            print(f"shape of weight is different: {eh_info.module_name}. SD version may be different")

                        for t in ["q", "k", "v", "out"]:
                            del state_dict[f"{eh_info.module_name}_{t}_proj.linear.weight"]
                    else:
                        # corresponding weight not exists: version mismatch
                        pass

        # conversion 2nd step: convert weight's shape (and handle wrapped)
        state_dict = self.convert_state_dict_shape_to_compvis(state_dict)

        return state_dict
    
    def convert_state_dict_shape_to_compvis(self, state_dict):
        # shape conversion
        current_sd = self.state_dict()      # to get target shape
        wrapped = False
        count = 0
        for key in list(state_dict.keys()):
            if key not in current_sd:
                continue           # might be error or another version
            if "wrapped" in key:
                wrapped = True

            value: torch.Tensor = state_dict[key]
            if value.size() != current_sd[key].size():
                # print(f"convert weights shape: {key}, from: {value.size()}, {len(value.size())}")
                count += 1
                if len(value.size()) == 4:
                    value = value.squeeze(3).squeeze(2)
                else:
                    value = value.unsqueeze(2).unsqueeze(3)
                state_dict[key] = value
            if tuple(value.size()) != tuple(current_sd[key].size()):
                print(
                        f"weight's shape is different: {key} expected {current_sd[key].size()} found {value.size()}. SD version may be different")
                del state_dict[key]
        print(f"shapes for {count} weights are converted.")

        # convert wrapped
        if not wrapped:
            print("remove 'wrapped' from keys")
            for key in list(state_dict.keys()):
                if "_wrapped_" in key:
                    new_key = key.replace("_wrapped_", "_")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        return state_dict
