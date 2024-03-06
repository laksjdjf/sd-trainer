from networks.manager import NetworkManager
import torch
import torch.nn as nn
import torch.nn.functional as F

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

# https://github.com/huggingface/diffusers/blob/687bc2772721af584d649129f8d2a28ca56a9ad8/src/diffusers/models/controlnet.py#L66
class ControlNetConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class LoRTnoCManager(NetworkManager):
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
        in_channels=3,
    ):
        super().__init__(
            text_model,
            unet, 
            module,
            file_name,
            module_args,
            unet_key_filters,
            conv_module_args,
            text_module_args,
            multiplier,
            mode,
        )
        self.in_channels = in_channels
        self.hidden_channels = unet.conv_in.out_channels
        
        self.conditioning_embedding = ControlNetConditioningEmbedding(self.hidden_channels, in_channels)
        self.org_conv_in = [unet.conv_in]

        if mode == "apply":
            self.org_conv_in_forward = self.org_conv_in[0].forward
            self.org_conv_in[0].forward = self.forward_hook(self.org_conv_in_forward)
        elif mode == "merge":
            raise NotImplementedError("merge mode is not supported yet for LoRTnoCManager.")
        else:
            raise ValueError(f"mode {self.mode} is not supported.")
            
    def set_controlnet_hint(self, hint):
        self.hint = hint

    def forward_hook(self, forward):
        def hook(x):
            hint = self.conditioning_embedding(self.hint)
            return forward(x) + hint
        return hook

    def apply_to(self, multiplier=None):
        super().apply_to(multiplier)
        if hasattr(self, "org_conv_in"):
            if not hasattr(self, "org_conv_in_forward"):
                self.org_conv_in_forward = self.org_conv_in[0].forward
            self.org_conv_in[0].forward = self.forward_hook(self.org_conv_in_forward)

    def unapply_to(self):
        super().unapply_to()
        self.org_conv_in[0].forward = self.org_conv_in_forward

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        optimizer_params = super().prepare_optimizer_params(text_encoder_lr, unet_lr)
        optimizer_params += [{"params": self.conditioning_embedding.parameters(), "lr": unet_lr}]
        return optimizer_params
