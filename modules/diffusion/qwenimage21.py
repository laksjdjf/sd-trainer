import torch
from .base import DiffusionModel


class QwenImage21DiffusionModel(DiffusionModel):
    """Text-to-image adapter. KV caching stays off so gradients remain valid."""

    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if controlnet_hint is not None:
            raise ValueError("Qwen-Image-2.1 image conditioning is not implemented.")
        if latents.ndim != 5 or latents.shape[2] != 1:
            raise ValueError("Qwen-Image-2.1 expects image latents shaped B,C,1,H,W.")
        batch, channels, _, height, width = latents.shape
        if height % 2 or width % 2:
            raise ValueError("Qwen-Image-2.1 image dimensions must be multiples of 32 pixels.")
        hidden = latents.flatten(2).transpose(1, 2)
        text = text_output.encoder_hidden_states.to(hidden)
        mask = text_output.attention_mask
        if mask is not None:
            mask = mask.to(device=hidden.device, dtype=torch.bool)
        # Each VLM image slot expands into four latent tokens.
        img_mask = torch.cat([
            torch.zeros(batch, text.shape[1], device=hidden.device, dtype=torch.bool),
            torch.ones(batch, height * width // 4, device=hidden.device, dtype=torch.bool),
        ], dim=1)
        t = timesteps.to(hidden).reshape(-1).expand(batch) / 1000
        output = self.unet(
            hidden_states=hidden, encoder_hidden_states=text,
            encoder_hidden_states_mask=mask, timestep=t,
            img_shapes=[[(1, height, width)] for _ in range(batch)],
            img_mask=img_mask, return_dict=False,
        )[0]
        return output[:, -height * width:].transpose(1, 2).reshape(batch, channels, 1, height, width)
