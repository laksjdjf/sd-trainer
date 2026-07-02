import torch
from .base import DiffusionModel


class ZImageDiffusionModel(DiffusionModel):
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        timesteps = (1000 - timesteps.to(latents)) / 1000
        latents = latents.unsqueeze(2)  # add frame dimension
        latents = list(latents.unbind(dim=0))
        
        model_output = self.unet(
            x=latents,
            t=timesteps,
            cap_feats=text_output.encoder_hidden_states,
        )[0]

        return - torch.stack(model_output).squeeze(2)  # remove frame dimension
