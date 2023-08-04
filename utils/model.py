from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import random

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

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class TextModel(nn.Module):
    def __init__(self, tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=-1):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_list = [tokenizer] if tokenizer_2 is None else [tokenizer, tokenizer_2]

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_list = [text_encoder] if text_encoder_2 is None else [text_encoder, text_encoder_2]

        self.clip_skip = clip_skip
        self.sdxl = tokenizer_2 is not None

        self.textual_inversion = False

    def tokenize(self, texts):
        if self.textual_inversion:
            texts = self.get_caption_from_template(num_captions=len(texts))
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
            
            # calculate pooled_output
            last_hidden_state = encoder_output_2.last_hidden_state
            eos_token_index = torch.where(tokens_2 == self.tokenizer_2.eos_token_id)[1].to(device=last_hidden_state.device)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                eos_token_index
            ]
            pooled_output = self.text_encoder_2.text_projection(pooled_output)

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

    def encode_text(self, texts):
        tokens, tokens_2, empty_text = self.tokenize(texts)
        encoder_hidden_states, pooled_output = self.forward(tokens, tokens_2, empty_text)
        return encoder_hidden_states, pooled_output

    def prepare_textual_inversion(self, initial_token, target_token, num_tokens=1, mode="style", initial_file=None):
        self.num_tokens = num_tokens
        self.target_tokens = [target_token] + [target_token + str(i) for i in range(1,num_tokens)]
        self.target_token = " ".join(self.target_tokens)
        if mode == "style":
            self.textual_inversion_template = imagenet_style_templates_small
        elif mode == "object":
            self.textual_inversion_template = imagenet_templates_small
        else:
            raise ValueError("mode must be 'style' or 'object'.")

        if initial_file is not None:
            init_embs = self.load_embeddings(initial_file)
        else:
            init_embs = [None] * len(self.tokenizer_list)
        
        self.token_embeds_list = []
        for tokenizer, text_encoder, init_emb in zip(self.tokenizer_list, self.text_encoder_list, init_embs):
            add_tokens = tokenizer.add_tokens(self.target_tokens)  # トークンを追加
            assert add_tokens == num_tokens, "既に登録されているトークンは使えません"

            text_encoder.resize_token_embeddings(len(tokenizer))  # embeddingを更新
            init_token_id = (tokenizer.convert_tokens_to_ids([initial_token]))[0]
            assert init_token_id != tokenizer.eos_token_id, "init tokenが1トークンではありません。"

            token_embeds = text_encoder.get_input_embeddings().weight.data
            for i in range(num_tokens):
                if init_emb is None:
                    token_embeds[-(i+1)] = token_embeds[init_token_id]
                else:
                    token_embeds[-(i+1)] = init_emb[-(i+1)]
            self.token_embeds_list.append(token_embeds.detach().clone())
        print("Textual Inversion用に新しいトークンが追加されました。")
        self.textual_inversion = True

    def get_caption_from_template(self, num_captions=1):
        captions = random.choices(self.textual_inversion_template, k=num_captions)
        captions = [caption.format(self.target_token) for caption in captions]
        return captions
    
    @torch.no_grad()
    def reset_unupdate_embedding(self):
        for i, text_encoder in enumerate(self.text_encoder_list):
            text_encoder.get_input_embeddings().weight[:-(self.num_tokens+1)] = self.token_embeds_list[i][:-(self.num_tokens+1)]

    def set_textual_inversion_mode(self):
        text_encoder_list = [self.text_encoder] if not self.sdxl else [self.text_encoder, self.text_encoder_2]
        for text_encoder in text_encoder_list:
            text_encoder.requires_grad_(True)
            text_encoder.text_model.encoder.requires_grad_(False)
            text_encoder.text_model.final_layer_norm.requires_grad_(False)
            text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
            text_encoder.train()

    def get_textual_inversion_params(self):
        params = [self.text_encoder.get_input_embeddings().parameters()]
        if self.sdxl:
            params.append(self.text_encoder_2.get_input_embeddings().parameters())
        return params

    def save_embeddings(self, file, save_dtype=torch.float32):
        if self.sdxl:
            state_dict = {
                "clip_l": self.text_encoder.get_input_embeddings().weight[-self.num_tokens:],
                "clip_g": self.text_encoder_2.get_input_embeddings().weight[-self.num_tokens:]
            }
        else:
            state_dict = {"emb_params": self.text_encoder.get_input_embeddings().weight[-self.num_tokens:]}

        if save_dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(save_dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == '':
            file += '.safetensors'

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)  # can be loaded in Web UI

    def load_embeddings(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            data = load_file(file)
        else:
            data = torch.load(file, map_location="cpu")

        if self.sdxl:
            return [data["clip_l"], data["clip_g"]]
        else:
            return [data["emb_params"]]

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
