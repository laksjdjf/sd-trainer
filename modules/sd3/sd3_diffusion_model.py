import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

class SD3DiffusionModel(nn.Module):
    def __init__(
        self, 
        unet:UNet2DConditionModel,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = None
        self.sdxl = None
    
    def forward(self, latents, timesteps, encoder_hidden_states, pooled_output, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        model_output = self.unet(
            latents,
            encoder_hidden_states,
            pooled_output,
            timesteps,
        ).sample

        return model_output
    
    def create_controlnet(self, config):
        return
    
    def enable_gradient_checkpointing(self, enable:bool=True):
        if enable:
            self.unet.enable_gradient_checkpointing()
        else:
            self.unet.disable_gradient_checkpointing()
        