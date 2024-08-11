import torch.nn as nn
from tqdm import tqdm
from modules.trainer import BaseTrainer
from modules.flux.flux_diffusion_model import FluxDiffusionModel
from modules.flux.flux_text_model import FluxTextModel
from modules.flux.flux_scheduler import FluxScheduler
from networks.manager import NetworkManager
from diffusers import AutoencoderKL, FluxTransformer2DModel
import logging
import torch
logger = logging.getLogger("トレーナーちゃん")

def load_model(path, sdxl=False, clip_skip=-2, revision=None, torch_dtype=torch.bfloat16):
    text_model = FluxTextModel.from_pretrained(path, revision=revision, torch_dtype=torch_dtype)
    unet = FluxTransformer2DModel.from_pretrained(path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype)
    scheduler = None
            
    text_model.clip_skip = clip_skip
    return text_model, vae, unet, scheduler

class FluxTrainer(BaseTrainer):
    def __init__(self, config, diffusion:FluxTransformer2DModel, text_model:FluxTextModel, vae:AutoencoderKL, scheduler, network:NetworkManager):
        self.config = config
        self.diffusion = diffusion
        self.text_model = text_model
        self.vae = vae
        self.network = network
        self.diffusers_scheduler = scheduler # モデルのセーブ次にのみ利用
        self.scheduler = FluxScheduler()
        self.sdxl = text_model.sdxl
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159
        self.input_channels = 16
        
        if config is not None and config.merging_loras:
            for lora in config.merging_loras:
                NetworkManager(
                    text_model=self.text_model,
                    unet=self.diffusion.unet,
                    file_name=lora,
                    mode="merge"
                )
    
    @torch.no_grad()
    def sample(
        self, 
        prompt="", 
        negative_prompt="", 
        batch_size=1, 
        height=768, 
        width=768, 
        num_inference_steps=30, 
        guidance_scale=7.0, 
        denoise=1.0, 
        seed=4545, 
        images=None, 
        controlnet_hint=None, 
        **kwargs
    ):
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if guidance_scale != 1.0:
            prompt = [negative_prompt] * batch_size + [prompt] * batch_size
        else:
            prompt = [prompt] * batch_size

        timesteps = self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = timesteps[int(num_inference_steps*(1-denoise)):]

        if images is None:
            latents = torch.ones(batch_size, self.input_channels, height // 8, width // 8, device=self.device, dtype=self.autocast_dtype)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype):
                latents = self.encode_latents(images)
            latents.to(dtype=self.autocast_dtype)
        latents = (latents - self.shift_factor) * self.scaling_factor

        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        self.text_model.to("cuda")
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            encoder_hidden_states, pooled_output = self.text_model(prompt)
        self.text_model.to(self.te_device)

        if controlnet_hint is not None:
            if isinstance(controlnet_hint, str):
                controlnet_hint = Image.open(controlnet_hint).convert("RGB")
                controlnet_hint = transforms.ToTensor()(controlnet_hint).unsqueeze(0)
            controlnet_hint = controlnet_hint.to(self.device)
            if guidance_scale != 1.0:
                controlnet_hint = torch.cat([controlnet_hint] *2)
            
            if hasattr(self.network, "set_controlnet_hint"):
                self.network.set_controlnet_hint(controlnet_hint)

        progress_bar = tqdm(timesteps, desc="Sampling", leave=False, total=len(timesteps))

        for i, t in enumerate(timesteps):
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                latents_input = torch.cat([latents] * (2 if guidance_scale != 1.0 else 1), dim=0)
                model_output = self.diffusion(latents_input, t, encoder_hidden_states, pooled_output, controlnet_hint=controlnet_hint, guidance=3.0)

            if guidance_scale != 1.0:
                uncond, cond = model_output.chunk(2)
                model_output = uncond + guidance_scale * (cond - uncond)
            
            if i+1 < len(timesteps):
                latents = self.scheduler.step(latents, model_output, t, timesteps[i+1])
            else:
                latents = self.scheduler.pred_original_sample(latents, model_output, t)
            progress_bar.update(1)

        with torch.autocast("cuda", dtype=self.vae_dtype):
            images = self.decode_latents(latents / self.scaling_factor + self.shift_factor)
        
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        
        return images
    
    @classmethod
    def from_pretrained(cls, path, sdxl=False, clip_skip=None, config=None, network=None, revision=None, torch_dtype=torch.bfloat16):
        if clip_skip is None:
            clip_skip = -2 if sdxl else -1
        text_model, vae, unet, scheduler = load_model(path, sdxl=False, clip_skip=-2, revision=revision, torch_dtype=torch_dtype)
        diffusion = FluxDiffusionModel(unet)
        return cls(config, diffusion, text_model, vae, scheduler, network)