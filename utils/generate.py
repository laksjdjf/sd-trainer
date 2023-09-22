# 検証画像用のシンプルな生成器
# pipelineを継承（したけどコンストラクタ以外はこのコードで完結）
# autocastは使う側でやらせる

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import __version__ as diffusers_version
from utils.model import TextModel
from PIL import Image
from tqdm import tqdm
import os
import json


class StableDiffusionGenerator:
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_model: TextModel,
        scheduler_config
    ):
        self.unet = unet
        self.vae = vae
        self.text_model = text_model
        self.scheduler = DDIMScheduler.from_config(scheduler_config)
        self.sdxl = text_model.text_encoder_2 is not None
        self.latent_scale = 0.13025 if self.sdxl else 0.18215

    @torch.no_grad()
    def encode_prompts(self, prompts):
        encoder_hidden_states, pooled_output = self.text_model.encode_text(prompts)                                                      
        return encoder_hidden_states, pooled_output

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents / self.latent_scale
        images = []
        # VRAM節約のためにバッチサイズ1でループ
        for i in range(latents.shape[0]):
            image = self.vae.decode(latents[i].unsqueeze(0)).sample
            images.append(image)
        images = torch.cat(images, dim=0)
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @torch.no_grad()
    def generate(
        self,
        prompts,
        negative_prompts,
        height: int = 896,
        width: int = 640,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        pfg_feature: torch.Tensor = None,
        controlnet=None,
        guide_image=None,
        text_embeds=None,
        additional_skip=None,
        seed=4545,
    ):
        
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if type(prompts) == str:
            prompts = [prompts]
        if type(negative_prompts) == str:
            negative_prompts = [negative_prompts]
        if len(negative_prompts) == 1:
            negative_prompts = negative_prompts * len(prompts)

        assert len(prompts) == len(negative_prompts), "プロンプトとネガティブプロンプトの数が一致していません"

        if text_embeds is not None:
            encoder_hidden_state, pooled_output = text_embeds
        else:
            encoder_hidden_state, pooled_output = self.encode_prompts(prompts+negative_prompts)

        if pfg_feature is not None:
            cond, uncond = encoder_hidden_state.chunk(2)
            # cat (b, n1, d) + (b, n2, d)
            cond = torch.cat(
                [cond, pfg_feature.repeat(cond.shape[0], 1, 1)], dim=1)

            # zero padding
            uncond = torch.cat([uncond, torch.zeros(uncond.shape[0], pfg_feature.shape[1],
                               uncond.shape[2]).to(uncond.device, dtype=uncond.dtype)], dim=1)
            encoder_hidden_state = torch.cat([cond, uncond], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device="cuda")
        timesteps = self.scheduler.timesteps

        latents = torch.randn((len(prompts), 4, height // 8, width // 8), device="cuda")
        latents = latents * self.scheduler.init_noise_sigma
        
        if self.sdxl:
            size_condition = list((height, width) + (0, 0) + (height, width))
            size_condition = torch.tensor([size_condition], dtype=latents.dtype, device=latents.device)
            size_condition = torch.cat([size_condition]*2).repeat(latents.shape[0], 1) # なにやってだこれ
            added_cond_kwargs = {"text_embeds": pooled_output, "time_ids": size_condition}
        else:
            added_cond_kwargs = None

        progress_bar = tqdm(range(num_inference_steps), desc="Total Steps", leave=False)
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if controlnet is not None:
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_state,
                    controlnet_cond=guide_image,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
            elif additional_skip is not None:
                down_block_res_samples, mid_block_res_sample = additional_skip.res(len(prompts)*2, height // 8, width // 8)
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=encoder_hidden_state,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

            # ネガティブプロンプト版CFG
            noise_pred_text, noise_pred_negative = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_text - noise_pred_negative)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            progress_bar.update(1)

        images = self.decode_latents(latents)

        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

        return images
    
    def save_pretrained(self, save_directory):
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.text_model.save_pretrained(save_directory)
        self.scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))

        model_index = {}
        model_index["unet"] = ["diffusers", "UNet2DConditionModel"]
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
        model_index["text_encoder"] = ["transformers", "CLIPTextModel"]
        model_index["tokenizer"] = ["transformers", "CLIPTokenizer"]
        model_index["scheduler"] = ["diffusers", "DDIMScheduler"]
        if self.sdxl:
            model_index["text_encoder_2"] = ["transformers", "CLIPTextModelWithProjection"]
            model_index["tokenizer_2"] = ["transformers", "CLIPTokenizer"]
            model_index["_class_name"] = "StableDiffusionXLPipeline"
            model_index["force_zeros_for_empty_prompt"] = True
        else:
            model_index["_class_name"] = "StableDiffusionPipeline"

        model_index["_diffusers_version"] = diffusers_version

        with open(os.path.join(save_directory, "model_index.json"), "w") as f:
            json.dump(model_index, f)

