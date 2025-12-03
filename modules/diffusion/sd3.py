from .base import DiffusionModel

class SD3DiffusionModel(DiffusionModel):
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        model_output = self.unet(
            latents,
            text_output.encoder_hidden_states,
            text_output.pooled_output,
            timesteps,
        ).sample

        return model_output
    
    def prepare_fp8(self, autocast_dtype):
        for modules in self.unet.modules():
            if modules.__class__.__name__ in ["PatchEmbed", "RMSNorm"]:
                modules.to(autocast_dtype)
