import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def apply_to(self, multiplier=None):
        if multiplier is not None:
            self.multiplier = multiplier
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def unapply_to(self):
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=None, sign=1):
        lora_weight = self.get_weight(multiplier) * sign

        # get org weight
        org_sd = self.org_module[0].state_dict()
        org_weight = org_sd["weight"]
        weight = org_weight + lora_weight.to(org_weight)

        # set weight to org_module
        org_sd["weight"] = weight
        self.org_module[0].load_state_dict(org_sd)

    def restore_from(self, multiplier=None):
        self.merge_to(multiplier=multiplier, sign=-1)

    def get_weight(self, multiplier=None):
        raise NotImplementedError

class LoRAModule(BaseModule):

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, state_dict=None, rank=4, alpha=1, forward_mode="sequential"):
        super().__init__()
        self.lora_name = lora_name
        self.forward_mode = forward_mode

        if state_dict is not None:
            up_weight = state_dict[f"{lora_name}.lora_up.weight"]
            down_weight = state_dict[f"{lora_name}.lora_down.weight"]
            alpha = state_dict[f"{lora_name}.alpha"]
            rank = up_weight.shape[1]

        self.rank = rank

        if 'Linear' in org_module.__class__.__name__: # ["Linear", "LoRACompatibleLinear"]
            in_dim = org_module.in_features
            out_dim = org_module.out_features

            self.lora_down = torch.nn.Linear(in_dim, rank, bias=False)
            self.lora_up = torch.nn.Linear(rank, out_dim, bias=False)

            self.functional = torch.nn.functional.linear
            self.functional_args = {}

        elif 'Conv' in org_module.__class__.__name__: # ["Conv2d", "LoRACompatibleConv"]
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.rank = min(self.rank, in_dim, out_dim)
            if self.rank != rank:
                print(f"{lora_name} dim (rank) is changed to: {self.rank} because of in_dim or out_dim is smaller than rank")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.rank, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(
                self.rank, out_dim, (1, 1), (1, 1), bias=False)
            
            self.functional = torch.nn.functional.conv2d
            self.functional_args = {
                "stride": stride,
                "padding": padding,
            }

        self.shape = org_module.weight.shape

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = rank if alpha is None or alpha == 0 else alpha

        self.register_buffer('alpha', torch.tensor(alpha))

        if state_dict is not None:
            self.lora_down.weight = torch.nn.Parameter(down_weight.float())
            self.lora_up.weight = torch.nn.Parameter(up_weight.float())
        else:  # same as microsoft's
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = [org_module] # moduleにならないようにlistに入れる

    # calculate lora weight (delta W)
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.view(-1, self.rank) # out_dim, rank
        down_weight = self.lora_down.weight.view(self.rank, -1) # rank, in_dim*kernel*kernel
        
        lora_weight = up_weight @ down_weight  # out_dim, in_dim*kernel*kernel
        lora_weight = lora_weight.view(self.shape)  # out_dim, in_dim, [kernel, kernel]

        return lora_weight * multiplier * self.alpha / self.rank

    def lora_forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.multiplier * self.alpha / self.rank

    def forward(self, x, scale = None):
        if self.multiplier == 0.0:
            return self.org_forward(x)
        if self.forward_mode == "sequential":
            return self.org_forward(x) + self.lora_forward(x)
        elif self.forward_mode == "merge":
            weight = self.org_module[0].state_dict()["weight"]
            bias = None if "bias" not in self.org_module[0].state_dict() else self.org_module[0].state_dict()["bias"]
            return self.functional(x, weight + self.get_weight(), bias, **self.functional_args)