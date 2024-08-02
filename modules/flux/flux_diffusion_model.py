import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

class FluxDiffusionModel(nn.Module):
    def __init__(
        self, 
        unet:UNet2DConditionModel,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = None
        self.sdxl = None

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def _unpack_latents(self, latents, height, width):
        batch_size, num_patches, channels = latents.shape

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents
    
    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
    
    def _prepare_text_ids(self, batch_size, num_prompt_tokens, device, dtype):
        text_ids = torch.zeros(batch_size, num_prompt_tokens, 3).to(device=device, dtype=dtype)
        return text_ids
    
    def forward(self, latents, timesteps, encoder_hidden_states, pooled_output, size_condition=None, controlnet_hint=None):
        batch_size, num_channels_latents, height, width = latents.shape

        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, latents.device, latents.dtype)
        text_ids = self._prepare_text_ids(batch_size, encoder_hidden_states.shape[1], latents.device, latents.dtype)
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        timesteps = timesteps.to(latents) / 1000

        guidance = torch.tensor([0.0]*latents.shape[0]).to(latents)
        model_output = self.unet(
            hidden_states=latents,
            timestep=timesteps,
            guidance=guidance,
            pooled_projections=pooled_output,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        model_output = self._unpack_latents(model_output, height, width)

        return model_output
    
    def create_controlnet(self, config):
        return
    
    def enable_gradient_checkpointing(self, enable:bool=True):
        if enable:
            self.unet.enable_gradient_checkpointing()
        else:
            self.unet.disable_gradient_checkpointing()
        