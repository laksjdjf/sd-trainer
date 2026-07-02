import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from networks.lora import LoRAModule

class iLoRAModule(LoRAModule):
    def forward(self, x, scale = None):
        if self.multiplier == 0.0:
            if hasattr(self, "num_data"):
                self.num_data += x.shape[0]
                imat_in = x.pow(2).mean(dim=(0,1)).norm(dim=-1).unsqueeze(0)
                imat_in = imat_in / imat_in.norm(dim=-1).unsqueeze(0) / math.sqrt(x.shape[-1]) * 0.1
                self.lora_down.weight.data = self.lora_down.weight.data * (self.num_data - x.shape[0]) / self.num_data + imat_in / self.num_data
            else:
                self.num_data = x.shape[0]
                self.lora_down.weight.data = x.pow(2).mean(dim=(0,1)).unsqueeze(0)
            y =  self.org_forward(x)

            imat_out = y.pow(2).mean(dim=(0,1)).unsqueeze(-1)
            imat_out = imat_out / imat_out.norm(dim=-1).unsqueeze(-1) / math.sqrt(y.shape[-1]) * 0.1
            self.lora_up.weight.data = self.lora_up.weight.data * (self.num_data - x.shape[0]) / self.num_data + imat_out / self.num_data

            return y

        if self.forward_mode == "sequential":
            return self.org_forward(x) + self.lora_forward(x)
        elif self.forward_mode == "merge":
            weight = self.org_module[0].state_dict()["weight"]
            bias = None if "bias" not in self.org_module[0].state_dict() else self.org_module[0].state_dict()["bias"]
            return self.functional(x, weight + self.get_weight(), bias, **self.functional_args)