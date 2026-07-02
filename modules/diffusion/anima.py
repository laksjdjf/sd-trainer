import torch

from .base import DiffusionModel
from diffusers_anima.pipelines.anima.text_encoding import build_condition


class AnimaDiffusionModel(DiffusionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__enable_gradient_checkpointing()

    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))

        transformer_dtype = self.unet.dtype
        timestep = (timesteps.to(latents) / 1000).to(transformer_dtype)
        latent_model_input = latents.to(transformer_dtype)
        height = latents.shape[-2] * 8
        width = latents.shape[-1] * 8
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        prompt_embeds = build_condition(
            self.unet,
            qwen_hidden=text_output.encoder_hidden_states.to(device=latents.device, dtype=transformer_dtype),
            t5_ids=text_output.pooled_output.to(device=latents.device),
            t5_weights=text_output.attention_mask.to(device=latents.device, dtype=transformer_dtype),
        )
        prompt_embeds = prompt_embeds.clone()

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

    def __enable_gradient_checkpointing(self, enable=True):
        for module in self.unet.modules():
            if module.__class__.__name__ == "CosmosTransformerBlock":

                if enable:
                    if hasattr(module, "_original_forward"):
                        continue

                    module._original_forward = module.forward

                    def make_checkpointed_forward(m):
                        original_forward = m._original_forward

                        def checkpointed_forward(*args, **kwargs):

                            def custom_forward(*inputs):
                                return original_forward(*inputs, **kwargs)

                            return torch.utils.checkpoint.checkpoint(
                                custom_forward,
                                *args,
                                use_reentrant=False,
                            )

                        return checkpointed_forward

                    module.forward = make_checkpointed_forward(module)

                else:
                    if hasattr(module, "_original_forward"):
                        module.forward = module._original_forward
                        del module._original_forward