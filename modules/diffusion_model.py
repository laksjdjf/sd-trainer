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
    
    def forward(self, latents, timesteps, encoder_hidden_states, pooled_output, size_condition=None, controlnet_hint=None):
        if self.sdxl:
            if size_condition is None:
                h, w = latents.shape[2] * 8, latents.shape[3] * 8
                size_condition = torch.tensor([h, w, 0, 0, h, w]) # original_h/w. crop_top/left, target_h/w
                size_condition = size_condition.repeat(latents.shape[0], 1).to(latents)
            added_cond_kwargs = {"text_embeds": pooled_output, "time_ids": size_condition}
        else:
            added_cond_kwargs = None

        if self.controlnet is not None:
            assert controlnet_hint is not None, "controlnet_hint is required when controlnet is enabled"
            down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
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
            encoder_hidden_states,
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
        