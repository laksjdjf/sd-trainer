from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import os
import torch
import torch.nn as nn
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
            text_model = TextModel.from_pretrained(path)
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


class TextModel(nn.Module):
    def __init__(self, tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=-1):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.clip_skip = clip_skip
        self.sdxl = tokenizer_2 is not None

    def tokenize(self, texts):
        tokens = self.tokenizer(texts, max_length=self.tokenizer.model_max_length, padding="max_length",
                                truncation=True, return_tensors='pt').input_ids.to(self.text_encoder.device)
        if self.sdxl:
            tokens_2 = self.tokenizer_2(texts, max_length=self.tokenizer_2.model_max_length, padding="max_length",
                                        truncation=True, return_tensors='pt').input_ids.to(self.text_encoder_2.device)
            empty_text = []
            for text in texts:
                if text == "":
                    empty_text.append(True)
                else:
                    empty_text.append(False)
        else:
            tokens_2 = None
            empty_text = None

        return tokens, tokens_2, empty_text

    def forward(self, tokens, tokens_2=None, empty_text=None):
        encoder_hidden_states = self.text_encoder(tokens, output_hidden_states=True).hidden_states[self.clip_skip]
        if self.sdxl:
            encoder_output_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            pooled_output = encoder_output_2[0]
            encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

            # pooled_output is zero vector for empty text
            if empty_text is not None:
                for i, empty in enumerate(empty_text):
                    if empty:
                        pooled_output[i] = torch.zeros_like(pooled_output[i])
        else:
            encoder_hidden_states = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
            pooled_output = None

        return encoder_hidden_states, pooled_output
    
    def gradient_checkpointing_enable(self, enable=True):
        if enable:
            self.text_encoder.gradient_checkpointing_enable()
            if self.sdxl:
                self.text_encoder_2.gradient_checkpointing_enable()
        else:
            self.text_encoder.gradient_checkpointing_disable()
            if self.sdxl:
                self.text_encoder_2.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(cls, path, sdxl=False, clip_skip=-1):
        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(path, subfolder='text_encoder')
        if sdxl:
            tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder='tokenizer_2')
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder='text_encoder_2')
        else:
            tokenizer_2 = None
            text_encoder_2 = None
        return cls(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
    
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
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} # 省略 if is_torch_version(">=", "1.11.0") else {}
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