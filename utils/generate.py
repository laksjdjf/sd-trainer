# 検証画像用のシンプルな生成器
# pipelineを継承（したけどコンストラクタ以外はこのコードで完結）
# autocastは使う側でやらせる

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm import tqdm


class WrapStableDiffusionPipeline(StableDiffusionPipeline):
    def encode_prompts(self, prompts):
        with torch.no_grad():
            tokens = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding="max_length",
                                    truncation=True, return_tensors='pt').input_ids.to(self.device)
            encoder_hidden_state = self.text_encoder(tokens, output_hidden_states=True).last_hidden_state
        return encoder_hidden_state

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        images = []
        with torch.no_grad():
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

    def generate(self,
                 prompts,
                 negative_prompts,
                 height: int = 896,
                 width: int = 640,
                 guidance_scale: float = 7.0,
                 num_inference_steps: int = 50,
                 batch=None,
                 pfg_feature: torch.Tensor = None,
                 controlnet=None,
                 seed=4545,
                 ):

        torch.manual_seed(seed)

        if type(prompts) == str:
            prompts = [prompts]
        if type(negative_prompts) == str:
            negative_prompts = [negative_prompts]
        if len(negative_prompts) == 1:
            negative_prompts = negative_prompts * len(prompts)

        assert len(prompts) == len(negative_prompts), "プロンプトとネガティブプロンプトの数が一致していません"

        encoder_hidden_state = self.encode_prompts(prompts+negative_prompts)

        if pfg_feature is not None:
            cond, uncond = encoder_hidden_state.chunk(2)
            # cat (b, n1, d) + (b, n2, d)
            cond = torch.cat(
                [cond, pfg_feature.repeat(cond.shape[0], 1, 1)], dim=1)

            # zero padding
            uncond = torch.cat([uncond, torch.zeros(uncond.shape[0], pfg_feature.shape[1], uncond.shape[2]).to(uncond.device,dtype=uncond.dtype)], dim=1)
            encoder_hidden_state = torch.cat([cond, uncond], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = torch.randn((len(prompts), 4, height // 8, width // 8), device=self.device)
        latents = latents * self.scheduler.init_noise_sigma

        progress_bar = tqdm(range(num_inference_steps), desc="Total Steps", leave=False)
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if controlnet is not None:
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_state,
                    controlnet_cond=batch["control"].to(latents.device),
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=encoder_hidden_state,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

            # ネガティブプロンプト版CFG
            noise_pred_text, noise_pred_negative = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_text - noise_pred_negative)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            progress_bar.update(1)

        images = self.decode_latents(latents)

        torch.manual_seed(torch.seed())

        return images
