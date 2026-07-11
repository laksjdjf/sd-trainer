from .base import DiffusionModel


class AnimaDiffusionModel(DiffusionModel):
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))

        transformer_dtype = self.unet.dtype
        timestep = (timesteps.to(latents) / 1000).to(transformer_dtype)
        latent_model_input = latents.to(transformer_dtype)
        height = latents.shape[-2] * 8
        width = latents.shape[-1] * 8
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        prompt_embeds = text_output.encoder_hidden_states.to(
            device=latents.device,
            dtype=transformer_dtype,
        )

        model_output = self.unet(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]

        return model_output

    def prepare_fp8(self, autocast_dtype):
        pass
