from .base import DiffusionModel

class Lumina2DiffusionModel(DiffusionModel):
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        timesteps = 1 - (timesteps.to(latents) / 1000)
        
        model_output = self.unet(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=text_output.encoder_hidden_states,
            encoder_attention_mask=text_output.attention_mask,
            return_dict=False,
        )[0]

        return - model_output
