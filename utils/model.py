from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from modules.text_model import TextModel
import os
import torch
from typing import Optional, Dict, Any

def load_model(path, sdxl=False):
    if sdxl:
        if os.path.isfile(path):
            pipe = StableDiffusionXLPipeline.from_single_file(path, scheduler_type="ddim")
            tokenizer = pipe.tokenizer
            tokenizer_2 = pipe.tokenizer_2
            text_encoder = pipe.text_encoder
            text_encoder_2 = pipe.text_encoder_2
            unet = pipe.unet
            vae = pipe.vae
            scheduler = pipe.scheduler
            text_model = TextModel(tokenizer, tokenizer_2, text_encoder, text_encoder_2)
            del pipe
        else:
            text_model = TextModel.from_pretrained(path, sdxl=True)
            unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet')
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae')
            scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler')
    else:
        if os.path.isfile(path):
            pipe = StableDiffusionPipeline.from_single_file(path, scheduler_type="ddim")
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
            unet = pipe.unet
            vae = pipe.vae
            scheduler = pipe.scheduler
            text_model = TextModel(tokenizer, None, text_encoder, None)
            del pipe
        else:
            text_model = TextModel.from_pretrained(path)
            unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet')
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae')
            scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler')

    return text_model, vae, unet, scheduler

def patch_mid_block_checkpointing(mid_block):

    def forward(
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        if mid_block.training:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}  # 省略 if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(mid_block.resnets[0]),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            for attn, resnet in zip(mid_block.attentions, mid_block.resnets[1:]):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

        else:
            hidden_states = mid_block.resnets[0](hidden_states, temb)
            for attn, resnet in zip(mid_block.attentions, mid_block.resnets[1:]):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = resnet(hidden_states, temb)

        return hidden_states

    mid_block.forward = forward
    print("unet mid_blockにgradient checkpointingを無理やり適用しました。")
