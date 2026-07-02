from .base import DiffusionModel


class AuraFlowDiffusionModel(DiffusionModel):
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        timesteps = timesteps.to(latents) / 1000
            
        model_output = self.unet(
            latents,
            encoder_hidden_states=text_output.encoder_hidden_states,
            timestep=timesteps,
            return_dict=False,
        )[0]

        return model_output
