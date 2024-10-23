import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

class DiffusionModel(nn.Module):
    def __init__(
        self, 
        unet:UNet2DConditionModel,
        controlnet:ControlNetModel=None,
        sdxl:bool=False,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.sdxl = sdxl
    
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if self.sdxl:
            if size_condition is None:
                h, w = latents.shape[2] * 8, latents.shape[3] * 8
                size_condition = torch.tensor([h, w, 0, 0, h, w]) # original_h/w. crop_top/left, target_h/w
                size_condition = size_condition.repeat(latents.shape[0], 1).to(latents)
            added_cond_kwargs = {"text_embeds": text_output.pooled_output, "time_ids": size_condition}
        else:
            added_cond_kwargs = None

        if self.controlnet is not None:
            assert controlnet_hint is not None, "controlnet_hint is required when controlnet is enabled"
            down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                latents,
                timesteps,
                encoder_hidden_states=text_output.encoder_hidden_states,
                controlnet_cond=controlnet_hint,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        else:
            down_block_additional_residuals = None
            mid_block_additional_residual = None

        model_output = self.unet(
            latents,
            timesteps,
            text_output.encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample

        return model_output
    
    def create_controlnet(self, config):
        if config.resume is not None:
            pre_controlnet = ControlNetModel.from_pretrained(config.resume)
        else:
            pre_controlnet = ControlNetModel.from_unet(self.unet)  

        if config.transformer_layers_per_block is not None:
            down_block_types = tuple(["DownBlock2D" if l == 0 else "CrossAttnDownBlock2D" for l in config.transformer_layers_per_block])
            transformer_layers_per_block = tuple([int(x) for x in config.transformer_layers_per_block])
            self.controlnet = ControlNetModel.from_config(
                pre_controlnet.config,
                down_block_types=down_block_types,
                transformer_layers_per_block=transformer_layers_per_block,
            )
            self.controlnet.load_state_dict(pre_controlnet.state_dict(), strict=False)
            del pre_controlnet
        else:
            self.controlnet = pre_controlnet
        
        self.controlnet.config.global_pool_conditions = config.global_average_pooling

    def enable_gradient_checkpointing(self, enable:bool=True):
        for model in [self.unet, self.controlnet]:
            if model is not None:
                if enable:
                    model.enable_gradient_checkpointing()
                else:
                    model.disable_gradient_checkpointing()

    def prepare_fp8(self, autocast_dtype):
        pass

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


class FluxDiffusionModel(DiffusionModel):
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
    
    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        batch_size, num_channels_latents, height, width = latents.shape

        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, latents.device, latents.dtype)
        text_ids = self._prepare_text_ids(batch_size, text_output.encoder_hidden_states.shape[1], latents.device, latents.dtype)
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        timesteps = timesteps.to(latents) / 1000
        
        if self.unet.config.guidance_embeds:
            guidance = 3.0 if sample else 1.0 # torima
            guidance = torch.tensor([guidance]*latents.shape[0]).to(latents)
        else:
            guidance = None
            
        model_output = self.unet(
            hidden_states=latents,
            timestep=timesteps,
            guidance=guidance,
            pooled_projections=text_output.pooled_output,
            encoder_hidden_states=text_output.encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        model_output = self._unpack_latents(model_output, height, width)

        return model_output
    
    def prepare_fp8(self, autocast_dtype):
        for modules in self.unet.modules():
            if modules.__class__.__name__ in ["RMSNorm"]:
                modules.to(autocast_dtype)

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