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
        raise NotImplementedError

    def update_ema(self, decay=0.999):
        raise NotImplementedError

class LoRAModule(BaseModule):
    # replaces forward method of the original Linear, instead of replacing the original Linear module.

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_rank=4, alpha=1, forward_mode=None):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_rank = lora_rank
        self.forward_mode = forward_mode
        self.ema = False

        if 'Linear' in org_module.__class__.__name__:
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

        elif 'Conv' in org_module.__class__.__name__:
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

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        if self.ema:
            up_weight = self.ema_up
            down_weight = self.ema_down
        else:
            up_weight = self.lora_up.weight# out_dim, rank, [1, 1]
            down_weight = self.lora_down.weight # rank, in_dim, [kernel, kernel]
        
        lora_weight = up_weight.view(-1, self.lora_rank) @ down_weight.view(self.lora_rank, -1)  # out_dim, in_dim*kernel*kernel
        lora_weight = lora_weight.view(self.shape)  # out_dim, in_dim, [kernel, kernel]

        return lora_weight * multiplier * self.scale
    
    def update_ema(self, decay=0.999):
        self.decay = decay
        if not hasattr(self,"ema_up"):
            self.register_buffer('ema_up', self.lora_up.weight.detach())
            self.register_buffer('ema_down', self.lora_down.weight.detach())
        else:
            self.ema_up = self.ema_up * decay + self.lora_up.weight * (1 - decay)
            self.ema_down = self.ema_down * decay + self.lora_down.weight * (1 - decay)

    def lora_forward(self, x):
        if self.ema:
            x = self.op(x, self.ema_down, **self.extra_args)
            x = self.op(x, self.ema_up)
            return x * self.multiplier * self.scale
        else:
            return self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

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
            return self.org_forward(x) + self.lora_forward(x)
        
class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orig_weight, w1a, w1b, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a@w1b)*(w2a@w2b)) * scale
        return orig_weight.reshape(diff_weight.shape) + diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out*(w2a@w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a@w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        del temp
        return grad_out, grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


class HadaWeightCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1a, w1b, t2, w2a, w2b, scale)

        rebuild1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1b, w1a)
        rebuild2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2b, w2a)

        return orig_weight + rebuild1*rebuild2*scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1a, w1b, t2, w2a, w2b, scale) = ctx.saved_tensors

        grad_out = grad_out*scale

        temp = torch.einsum('i j k l, j r -> i r k l', t2, w2b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w2a)

        grad_w = rebuild*grad_out
        del rebuild

        grad_w1a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w1a.T)
        del grad_w, temp

        grad_w1b = torch.einsum('i r k l, i j k l -> r j', t1, grad_temp)
        grad_t1 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w1b.T)
        del grad_temp

        temp = torch.einsum('i j k l, j r -> i r k l', t1, w1b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w1a)

        grad_w = rebuild*grad_out
        del rebuild

        grad_w2a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w2a.T)
        del grad_w, temp

        grad_w2b = torch.einsum('i r k l, i j k l -> r j', t2, grad_temp)
        grad_t2 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w2b.T)
        del grad_temp
        return grad_out, grad_t1, grad_w1a, grad_w1b, grad_t2, grad_w2a, grad_w2b, None


def make_weight(orig_weight, w1a, w1b, w2a, w2b, scale):
    return HadaWeight.apply(orig_weight, w1a, w1b, w2a, w2b, scale)


def make_weight_cp(orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale):
    return HadaWeightCP.apply(orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale)


class LohaModule(BaseModule):
    """
    Hadamard product Implementaion for Low Rank Adaptation
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0, lora_dim=4, alpha=1, dropout=0.,
        use_cp=False,
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.cp = False

        self.shape = org_module.weight.shape
        if "Conv" in org_module.__class__.__name__:
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels

            if lora_dim == "dynamic":
                lora_dim = min(math.ceil(in_dim ** 0.5),
                               math.ceil(out_dim ** 0.5))
            self.lora_dim = lora_dim
            self.cp = use_cp and k_size != (1, 1)
            if self.cp:
                shape = (out_dim, in_dim, *k_size)
            else:
                shape = (out_dim, in_dim*k_size[0]*k_size[1])
            self.op = F.conv2d
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

            if lora_dim == "dynamic":
                lora_dim = min(math.ceil(in_dim ** 0.5),
                               math.ceil(out_dim ** 0.5))
            self.lora_dim = lora_dim
            shape = (out_dim, in_dim)
            self.op = F.linear
            self.extra_args = {}

        if self.cp:
            self.hada_t1 = nn.Parameter(torch.empty(
                lora_dim, lora_dim, shape[2], shape[3]))
            self.hada_w1_a = nn.Parameter(torch.empty(
                lora_dim, shape[0]))  # out_dim, 1-mode
            self.hada_w1_b = nn.Parameter(torch.empty(
                lora_dim, shape[1]))  # in_dim , 2-mode

            self.hada_t2 = nn.Parameter(torch.empty(
                lora_dim, lora_dim, shape[2], shape[3]))
            self.hada_w2_a = nn.Parameter(torch.empty(
                lora_dim, shape[0]))  # out_dim, 1-mode
            self.hada_w2_b = nn.Parameter(torch.empty(
                lora_dim, shape[1]))  # in_dim , 2-mode
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1]))

            self.hada_w2_a = nn.Parameter(torch.empty(shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1]))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))  # 定数として扱える

        # Need more experiences on init method
        if self.cp:
            torch.nn.init.normal_(self.hada_t1, std=0.1)
            torch.nn.init.normal_(self.hada_t2, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w2_b, std=0.01)
        torch.nn.init.normal_(self.hada_w1_a, std=1)
        torch.nn.init.constant_(self.hada_w2_a, 0)

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.grad_ckpt = False

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        d_weight = self.hada_w1_a @ self.hada_w1_b
        d_weight *= self.hada_w2_a @ self.hada_w2_b
        return (d_weight).reshape(self.shape) * multiplier * self.scale

    @torch.enable_grad()
    def forward(self, x):
        if self.multiplier == 0.0:
            return self.org_module[0](x)
        # print(torch.mean(torch.abs(self.orig_w1a.to(x.device) - self.hada_w1_a)), end='\r')
        if self.cp:
            weight = make_weight_cp(
                self.org_module[0].weight.data,
                self.hada_t1, self.hada_w1_a, self.hada_w1_b,
                self.hada_t1, self.hada_w2_a, self.hada_w2_b,
                scale=torch.tensor(self.scale*self.multiplier),
            )
        else:
            weight = make_weight(
                self.org_module[0].weight.data,
                self.hada_w1_a, self.hada_w1_b,
                self.hada_w2_a, self.hada_w2_b,
                scale=torch.tensor(self.scale*self.multiplier),
            )

        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        return self.op(
            x,
            weight.view(self.shape),
            bias,
            **self.extra_args
        )
