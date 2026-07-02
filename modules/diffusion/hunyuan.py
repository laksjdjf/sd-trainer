import torch
from .base import DiffusionModel


class HunyuanVideoDiffusionModel(DiffusionModel):
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        guidance = torch.tensor([6.0] * latents.shape[0]).to(latents) * 1000.0

        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0)).to(latents)
            
        model_output = self.unet(
            hidden_states=latents,
            timestep=timesteps,
            guidance=guidance,
            pooled_projections=text_output.pooled_output,
            encoder_hidden_states=text_output.encoder_hidden_states,
            encoder_attention_mask=text_output.attention_mask,

            return_dict=False,
        )[0]

        return model_output
    
    def prepare_fp8(self, autocast_dtype):
        for modules in self.unet.modules():
            if modules.__class__.__name__ in ["RMSNorm"]:
                modules.to(autocast_dtype)
