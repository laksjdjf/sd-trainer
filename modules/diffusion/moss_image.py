from .base import DiffusionModel


class MOSSImageDiffusionModel(DiffusionModel):
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None,
                controlnet_hint=None, skip_layers=None):
        if controlnet_hint is not None:
            raise ValueError("MOSS-Image ControlNet is not implemented.")
        if text_output.attention_mask is None:
            raise ValueError("MOSS-Image requires a text attention mask; regenerate text caches.")
        # Upstream uses t in [0,1000] directly and predicts noise - clean.
        return self.unet(
            hidden_states=latents, timestep=timesteps,
            encoder_hidden_states=text_output.encoder_hidden_states,
            encoder_attention_mask=text_output.attention_mask,
            skip_layers=skip_layers, return_dict=False,
        )[0]
