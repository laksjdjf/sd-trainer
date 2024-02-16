import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class DiffusionModel(nn.Module):
    def __init__(
        self, 
        unet:UNet2DConditionModel,
        sdxl:bool=False,
    ):
        super().__init__()
        self.unet = unet
        self.sdxl = sdxl
    
    def forward(self, latents, timesteps, encoder_hidden_states, pooled_output, size_condition=None):
        if self.sdxl:
            if size_condition is None:
                h, w = latents.shape[2] * 8, latents.shape[3] * 8
                size_condition = torch.tensor([h, w, 0, 0, h, w]) # original_h/w. crop_top/left, target_h/w
                size_condition = size_condition.repeat(latents.shape[0], 1).to(latents)
            added_cond_kwargs = {"text_embeds": pooled_output, "time_ids": size_condition}
        else:
            added_cond_kwargs = None

        model_output = self.unet(
            latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        return model_output
    
    def enable_gradient_checkpointing(self, enable:bool=True):
        if enable:
            self.unet.enable_gradient_checkpointing()
        else:
            self.unet.disable_gradient_checkpointing()
        